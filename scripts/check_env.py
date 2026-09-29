# -*- coding: utf-8 -*-
"""Пошаговая проверка окружения: python scripts/check_env.py
Грузит компоненты по одному и печатает время и память. Если процесс молча
завершается — последняя напечатанная строка показывает, на чём он упал."""
import faulthandler
import os
import sys
import time

faulthandler.enable()
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "app"))


def say(msg):
    print(msg, flush=True)


def ram():
    try:
        import psutil
        vm = psutil.virtual_memory()
        return f"ОЗУ: занято процессом {psutil.Process().memory_info().rss / 2**30:.1f} ГБ, свободно {vm.available / 2**30:.1f} ГБ"
    except Exception:
        return "ОЗУ: (pip install psutil, чтобы видеть)"


def step(name, fn):
    say(f"\n▶ {name}…")
    t = time.time()
    try:
        r = fn()
        say(f"  ✔ {name}: {time.time() - t:.1f} с · {ram()}")
        return r
    except Exception as e:
        say(f"  ✘ {name}: {type(e).__name__}: {e}")
        return None


say(f"Python {sys.version.split()[0]} · {sys.executable}")
torch = step("import torch", lambda: __import__("torch"))
if torch:
    say(f"  torch {torch.__version__}, CUDA доступна: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        say(f"  GPU: {p.name}, {p.total_memory / 2**30:.1f} ГБ")

import wine_engine as we  # noqa: E402
say(f"\nДанные: {we.BASE_DIR}")
for name in ["wine_index_v2.npy", "wine_index_v2_meta.json", "dataset.csv", "strapi_catalog.csv",
             "engine_config.json", "images_dataset", os.path.basename(we.LOCAL_MODEL_DIR)]:
    say(f"  {'✔' if os.path.exists(os.path.join(we.BASE_DIR, name)) else '✘ НЕТ'} {name}")
say(f"  веса CLIP: {we._find_local_weights() or 'не найдены локально — будут скачаны'}")

step("CLIP ViT-H", lambda: we.load_clip(say))
if torch and torch.cuda.is_available():
    say(f"  видеопамять занята: {torch.cuda.memory_allocated() / 2**30:.1f} ГБ")


def yolo():
    from ultralytics import YOLO, __version__
    say(f"  ultralytics {__version__} (для YOLO26 нужна ≥ 8.4)")
    return YOLO(os.path.join(we.BASE_DIR, we.SEG_WEIGHTS[0]))


step("YOLO26-seg", yolo)


def ocr():
    import easyocr
    return easyocr.Reader(we.OCR_LANGS, gpu=bool(torch and torch.cuda.is_available()), verbose=False)


step("EasyOCR ru+en", ocr)
say("\nГотово. Если все шаги ✔ — запускайте: uvicorn app.server:app --host 0.0.0.0 --port 8080")
