# Своё Вино — сканер этикеток

Сервис по фото этикетки находит карточку вина в каталоге «Своё Вино» (2110 позиций) и показывает
её в мобильном интерфейсе, плюс аналоги из других виноделен и «Цифровой сомелье».

## Быстрый старт

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
# GPU: поставьте torch с CUDA по инструкции pytorch.org, если ещё не стоит

# данные (не в git): эталонные фото и веса CLIP
#   data/images_dataset/                          — фото каталога (нужны для картинок карточек и build)
#   data/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin  — локальные веса (иначе скачаются)
# индекс уже лежит в data/: wine_index_v2.npy + wine_index_v2_meta.json + dataset.csv

uvicorn app.server:app --host 0.0.0.0 --port 8080
# открыть http://localhost:8080 
```

## API

| Метод | Путь | Ответ |
|---|---|---|
| POST | `/v1/eval/predict` | `{"slug":"wine-slug"}` — для скрипта кейсодержателя (поле `image`) |
| POST | `/predict` | то же самое (любое имя поля или сырые байты) |
| POST | `/api/scan` | карточка, `top5`, `confidence_top1`, `confidence_top5`, `margin_top1_top2`, `found`, аналоги |
| GET | `/api/wine/{slug}` | карточка |
| GET | `/api/analogs/{slug}` | аналоги из других виноделен |
| POST | `/api/sommelier` | `{"dish":"рыба","sweetness":"сухое","like_slug":"..."}` → подборка |

```bash
curl -F "image=@photo.jpg" http://127.0.0.1:8080/v1/eval/predict

# официальный скрипт кейсодержателя (нужны bash, curl, jq, awk)
cd eval
./participant_test.sh --images-dir ./queries --manifest ./queries.tsv \
  --endpoint 'http://127.0.0.1:8080/v1/eval/predict' --output ./predictions.jsonl

# свой прогон по папке фото + время ответа
bash scripts/eval_api.sh data/images_test http://127.0.0.1:8080/v1/eval/predict
```

## Переменные окружения

| Переменная | По умолчанию | Смысл |
|---|---|---|
| `WINE_DATA_DIR` | `./data` | где индекс, dataset.csv, images_dataset, веса |
| `USE_OCR` | `1` | `0` — только CV (быстрее, но хуже на near-duplicates) |
| `STRAPI_CSV` | `strapi_catalog.csv` | выгрузка Strapi в `WINE_DATA_DIR` (описания карточек) |
| `MIN_CONFIDENCE` | `0.35` | ниже — «точного совпадения нет», показываем похожие |
| `MIN_CLIP` | `0.45` | минимальное CLIP-сходство для «найдено» |

## Переиндексация и разметка

```bash
python wine_engine.py build        # пересобрать индекс после пополнения каталога
python wine_engine.py segtest      # контрольный лист вырезки бутылок YOLO
python label_tool.py               # окно разметки фото из data/images_test
python wine_engine.py eval         # метрики top-1/top-5 и подбор весов → engine_config.json
```

## Проверьте при старте

В логе запуска должно быть `[YOLO] загрузка yolo26m-seg.pt…` и `OCR=on`. Если YOLO26 не грузится
(старый ultralytics) — сервис молча берёт YOLO11, он заметно хуже; если OCR не встал — работает
только CLIP. Лечится: `pip install -U ultralytics easyocr`.

Качество (эмбеддинги ViT-H + OCR, 35 полевых фото из каталога): top-1 32/35, top-5 35/35;
без OCR — 30/35.

## Ограничения

- Карточки берутся из `data/strapi_catalog.csv` по slug: найдено для 2030 из 2110 вин.
  Ссылки на фото в выгрузке не совпадают с файлами — фото берутся из `dataset.csv`.
  Для остальных 80 вин атрибуты (цвет, сладость, сорт) выводятся из slug/подписи.
- Near-duplicates (одна этикетка, разный год/категория) различаются в основном через OCR.
- Первый запуск скачивает веса YOLO26-seg (~52 МБ) и EasyOCR.
