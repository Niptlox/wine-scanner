# -*- coding: utf-8 -*-
"""FastAPI-сервис сканера вин «Своё Вино».

  POST /v1/eval/predict      → {"slug": "..."}  поле multipart «image» (скрипт кейсодержателя)
  POST /predict              → то же самое
  POST /api/scan             → карточка + top5 + уверенность + аналоги (для фронтенда)
  GET  /api/wine/{slug}      → карточка
  GET  /api/analogs/{slug}   → аналоги из других виноделен
  POST /api/sommelier        → подбор вин по ответам «Цифрового сомелье»
  GET  /                     → мобильная страница сканера
"""
import os
import sys
import tempfile
import threading
import time
import traceback
import faulthandler

# Windows: при падении в C-коде (torch/cuda/opencv) печатаем стек, а не молча выходим
faulthandler.enable()
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # конфликт двух OpenMP (torch + numpy/opencv)
os.environ.setdefault("PYTHONUNBUFFERED", "1")

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import wine_engine as we  # noqa: E402
from catalog import SlugRecognizer  # noqa: E402

app = FastAPI(title="Своё Вино — сканер этикеток")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

REC = None
LOCK = threading.Lock()  # одна модель на GPU — запросы обрабатываем по очереди


def _log(msg):
    print(msg, flush=True)


@app.on_event("startup")
def load():
    global REC
    t = time.time()
    try:
        REC = SlugRecognizer(use_ocr=os.environ.get("USE_OCR", "1") == "1", log=_log)
    except Exception:
        _log("ОШИБКА при загрузке моделей:\n" + traceback.format_exc())
        raise
    _log(f"Готово за {time.time() - t:.0f} с: {len(REC.slugs)} вин, OCR={'on' if REC.ocr else 'off'}, "
         f"YOLO={getattr(REC, 'det_name', 'off') if REC.det else 'off'}, устройство={we.get_device()}")
    _log("Сервис слушает запросы. Страница: http://localhost:8080  Проверка: /health")


async def _save_upload(request: Request, file: UploadFile | None):
    """Принимаем multipart (любое имя поля) или «сырые» байты изображения в теле."""
    data = None
    if file is not None:
        data = await file.read()
    else:
        ctype = request.headers.get("content-type", "")
        if "multipart/form-data" in ctype:
            form = await request.form()
            for v in form.values():
                if hasattr(v, "read"):
                    data = await v.read()
                    break
        else:
            data = await request.body()
    if not data:
        raise HTTPException(400, "Нет изображения: отправьте файл (multipart) или байты картинки")
    fd, path = tempfile.mkstemp(suffix=".img")
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    return path


def _run(path):
    try:
        with LOCK:
            return REC.recognize(path)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


@app.post("/v1/eval/predict")
@app.post("/predict")
async def predict(request: Request, file: UploadFile | None = None):
    res = _run(await _save_upload(request, file))
    return JSONResponse({"slug": res["slug"]})


@app.post("/api/scan")
async def scan(request: Request, file: UploadFile | None = None):
    res = _run(await _save_upload(request, file))
    res["card"] = REC.card(res["slug"])
    res["analogs"] = REC.analogs(res["slug"], n=6)
    if not res["found"]:
        res["similar"] = [REC.card(t["slug"]) for t in res["top5"]]
    return res


@app.get("/api/wine/{slug}")
def wine(slug: str):
    if slug not in REC.slug2i:
        raise HTTPException(404, "Нет такого вина")
    return REC.card(slug)


@app.get("/api/analogs/{slug}")
def analogs(slug: str, n: int = 6):
    if slug not in REC.slug2i:
        raise HTTPException(404, "Нет такого вина")
    return REC.analogs(slug, n=n)


@app.post("/api/sommelier")
async def sommelier(request: Request):
    q = await request.json()
    return REC.sommelier(dish=q.get("dish"), color=q.get("color"), sweetness=q.get("sweetness"),
                         sparkling=q.get("sparkling"), like_slug=q.get("like_slug"),
                         region=q.get("region"), n=int(q.get("n", 5)))


@app.get("/img/{name}")
def img(name: str):
    p = os.path.join(we.IMAGES_DIR, os.path.basename(name))
    if not os.path.exists(p):
        raise HTTPException(404)
    return FileResponse(p)


@app.get("/health")
def health():
    return {"ok": REC is not None, "wines": len(REC.slugs) if REC else 0}


STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
