# -*- coding: utf-8 -*-
"""
wine_engine.py — распознавание вина по фото этикетки (CLIP + OCR + детектор бутылки).

Что добавлено по сравнению с clip_predict.py:
  • Многоракурсный индекс. Эталонная бутылка (горизонтально, белый фон) автоматически
    обрезается по содержимому, ставится вертикально горлышком вверх, и дополнительно
    кадрируются «тело» бутылки и зона этикетки. Каждый вариант — отдельный вектор с той
    же меткой. Так эталоны гораздо ближе к фото с полки / из руки.
  • Учёт EXIF-поворота телефонных фото (без него часть фото приходит в CLIP боком).
  • Сегментация бутылки (YOLO26-seg, класс COCO «bottle»): центральная бутылка вырезается
    по контуру и кладётся на белый фон — как в эталонном датасете. Фон, рука, соседние
    бутылки и ценники уходят. Наклонённая бутылка выпрямляется, делается и горизонтальный
    вариант (горлышком в ту же сторону, что в датасете).
  • TTA: запрос кодируется несколькими кадрированиями, эмбеддинги усредняются.
    Какие кадрирования использовать и вес OCR подбирает `eval` и сохраняет в engine_config.json.
  • OCR (EasyOCR, ru+en) + нечёткое сравнение с названиями из базы: транслитерация
    (FANAGORIA ↔ Фанагория), починка смешанной кириллицы/латиницы из OCR, IDF-веса слов.
    Итоговый скор = CLIP + OCR_WEIGHT * OCR.
  • Кэш анализа фото и оценка точности по ручной разметке (label_tool.py).

Команды:
  python wine_engine.py build              построить индекс (wine_index_v2.npy + _meta.json)
  python wine_engine.py predict фото.jpg   распознать одно фото
  python wine_engine.py eval               точность по test_annotations.csv (+ подбор настроек)
  python wine_engine.py segtest            контрольный лист: как YOLO вырезает бутылки из images_test
"""
import os
import sys
import csv
import json
import re
import math
import time
import pickle
import argparse
import unicodedata
import contextlib
from collections import Counter

import numpy as np
from PIL import Image, ImageOps

try:  # фото с iPhone в .heic — необязательно
    from pillow_heif import register_heif_opener
    register_heif_opener()
    _HEIC = True
except Exception:
    _HEIC = False

# ─── Конфигурация ───────────────────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.environ.get("WINE_DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")))
IMAGES_DIR = os.path.join(BASE_DIR, "images_dataset")
TEST_DIR = os.path.join(BASE_DIR, "images_test")
CSV_FILE = os.path.join(BASE_DIR, "dataset.csv")
LOCAL_MODEL_DIR = os.path.join(BASE_DIR, "CLIP-ViT-H-14-laion2B-s32B-b79K")

EMB_PATH = os.path.join(BASE_DIR, "wine_index_v2.npy")
META_PATH = os.path.join(BASE_DIR, "wine_index_v2_meta.json")
ANNOT_PATH = os.path.join(BASE_DIR, "test_annotations.csv")
CACHE_PATH = os.path.join(BASE_DIR, "analysis_cache.pkl")
CONFIG_PATH = os.path.join(BASE_DIR, "engine_config.json")

MODEL_NAME = "ViT-H-14"
PRETRAINED = "laion2b_s32b_b79k"      # используется, если нет локальных весов
# Модели сегментации по порядку предпочтения (скачиваются автоматически).
# На тестовых фото YOLO26 заметно лучше YOLO11: находит бутылку в руке, крупные планы.
SEG_WEIGHTS = ["yolo26m-seg.pt", "yolo11m-seg.pt"]
BOTTLE_CLASS = 39                     # «bottle» в COCO
SEG_CONF = 0.15
OCR_LANGS = ["ru", "en"]
OCR_WEIGHT = 0.3                      # вес OCR в итоговом скоре (подберите через eval)
OCR_MATCH_MIN = 0.75                  # порог похожести слова (0..1)
OCR_SMOOTH = 12.0                     # сглаживание: короткая метка («Мерло · merlo-2») не получает
                                      # 100% за одно общее слово. Подобрано на 71 фото из images_test:
                                      # top-1 только по тексту 12.7% → 43.7%
WORK_SIDE = 1600                      # до какого размера уменьшать фото перед анализом
NOT_IN_DB = "__NOT_IN_DB__"

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
if _HEIC:
    IMG_EXT |= {".heic", ".heif"}

WHITE = (255, 255, 255)
GRAY = (124, 124, 124)
_T = getattr(Image, "Transpose", Image)  # совместимость со старым Pillow

# ─── Устройство и CLIP ──────────────────────────────────────────────────────
_DEV = None


def get_device():
    global _DEV
    if _DEV is None:
        import torch
        _DEV = "cuda" if torch.cuda.is_available() else "cpu"
    return _DEV


def _find_local_weights():
    if not os.path.isdir(LOCAL_MODEL_DIR):
        return None
    for name in ("open_clip_pytorch_model.bin", "open_clip_model.safetensors"):
        p = os.path.join(LOCAL_MODEL_DIR, name)
        if os.path.isfile(p):
            return p
    return None


def load_clip(log=print):
    import open_clip
    import torch
    weights = _find_local_weights() or PRETRAINED
    dev = get_device()
    # на GPU — fp16: ViT-H в fp32 занимает ~4 ГБ видеопамяти и ~8 ГБ ОЗУ на пике загрузки
    precision = os.environ.get("CLIP_PRECISION", "fp16" if dev == "cuda" else "fp32")
    log(f"[CLIP] {MODEL_NAME} ← {weights} ({dev}, {precision}) — загрузка 30–90 с…")
    t = time.time()
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=weights, device=dev, precision=precision)
    model.eval()
    if dev == "cuda":
        log(f"[CLIP] готово за {time.time() - t:.0f} с, видеопамять {torch.cuda.memory_allocated() / 2**30:.1f} ГБ")
    else:
        log(f"[CLIP] готово за {time.time() - t:.0f} с")
    return model, preprocess


def encode_images(model, preprocess, pils, batch=None):
    """Список PIL → нормализованные эмбеддинги float32 [N, D]."""
    import torch
    dev = get_device()
    batch = batch or (32 if dev == "cuda" else 8)
    out = []
    with torch.no_grad():
        for i in range(0, len(pils), batch):
            x = torch.stack([preprocess(p) for p in pils[i:i + batch]]).to(dev)
            x = x.to(next(model.parameters()).dtype)
            ctx = (torch.autocast(device_type="cuda", dtype=torch.float16)
                   if dev == "cuda" else contextlib.nullcontext())
            with ctx:
                f = model.encode_image(x)
            f = f.float()
            f = f / f.norm(dim=-1, keepdim=True)
            out.append(f.cpu().numpy())
    return np.concatenate(out).astype(np.float32)


# ─── Работа с картинками ────────────────────────────────────────────────────
def load_image(path):
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)  # телефонные фото: учесть ориентацию
    if img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info):
        # эталоны сайта — webp с прозрачным фоном; простой convert("RGB") делает фон
        # ЧЁРНЫМ. Кладём на белый, как на обычной товарной фотографии.
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    return img.convert("RGB")


def pad_to_square(img, color=WHITE):
    w, h = img.size
    s = max(w, h)
    canvas = Image.new("RGB", (s, s), color)
    canvas.paste(img, ((s - w) // 2, (s - h) // 2))
    return canvas


def center_crop_frac(img, frac):
    w, h = img.size
    cw, ch = max(1, int(w * frac)), max(1, int(h * frac))
    x0, y0 = (w - cw) // 2, (h - ch) // 2
    return img.crop((x0, y0, x0 + cw, y0 + ch))


def crop_to_content(img, thr=235):
    """Обрезать белые поля вокруг бутылки. Возвращает (картинка, маска бутылки).
    Строки/столбцы с единичными «шумовыми» пикселями (JPEG) не учитываются."""
    g = np.asarray(img.convert("L"))
    mask = g < thr
    if mask.sum() < 100:
        return img, mask
    h, w = mask.shape
    cols = np.nonzero(mask.sum(axis=0) > max(2, 0.004 * h))[0]
    rows = np.nonzero(mask.sum(axis=1) > max(2, 0.004 * w))[0]
    if len(cols) == 0 or len(rows) == 0:
        return img, mask
    x0, x1, y0, y1 = cols[0], cols[-1] + 1, rows[0], rows[-1] + 1
    return img.crop((int(x0), int(y0), int(x1), int(y1))), mask[y0:y1, x0:x1]


def _spans(mask, axis):
    """Толщина силуэта вдоль оси: от первого до последнего «не фона» (белая этикетка
    внутри силуэта не делает его тоньше)."""
    m = mask if axis == 0 else mask.T          # axis=0: по столбцам
    has = m.any(axis=0)
    first = np.argmax(m, axis=0)
    last = m.shape[0] - 1 - np.argmax(m[::-1], axis=0)
    return np.where(has, last - first + 1, 0).astype(np.float32)


def _ends(prof):
    """Средняя толщина у двух концов (5–25% длины, без самых кончиков)."""
    n = len(prof)
    a, b = max(0, int(n * 0.05)), max(1, int(n * 0.25))
    return float(np.median(prof[a:b])), float(np.median(prof[n - b:n - a]))


def make_upright(img, return_side=False):
    """Эталон на белом фоне → вертикальная бутылка горлышком вверх.
    Горлышко определяется как более узкий конец силуэта.
    return_side=True → ещё и где было горлышко: 'left' / 'right' / 'up' / 'down'."""
    img, mask = crop_to_content(img, thr=248)
    h, w = mask.shape
    side = "up"
    if w > h * 1.2:                                   # лежит горизонтально
        left, right = _ends(_spans(mask, 0))
        if left < right:                              # горлышко слева
            img = img.transpose(_T.ROTATE_270)        # по часовой: левый край → верх
            side = "left"
        else:                                         # горлышко справа
            img = img.transpose(_T.ROTATE_90)         # против часовой: правый край → верх
            side = "right"
    # Вертикальную бутылку не переворачиваем: на 2106 эталонах сайта все стоят горлышком
    # вверх, а «вверх дном» по силуэту путалось на светлом стекле с белым колпачком.
    return (img, side) if return_side else img


def upright_to_side(img, side):
    """Вертикальную бутылку (горлышко вверх) положить горлышком в сторону side."""
    if side == "left":
        return img.transpose(_T.ROTATE_90)
    if side == "right":
        return img.transpose(_T.ROTATE_270)
    if side == "down":
        return img.transpose(_T.ROTATE_180)
    return img


def estimate_neck_side(n=300):
    """В какую сторону чаще смотрит горлышко на эталонах (для старых индексов без этого поля)."""
    names = list_images(IMAGES_DIR)
    step = max(1, len(names) // n)
    c = Counter()
    for fn in names[::step][:n]:
        try:
            c[make_upright(load_image(os.path.join(IMAGES_DIR, fn)), True)[1]] += 1
        except Exception:
            pass
    return c.most_common(1)[0][0] if c else "left"


def deskew(img, mask):
    """Выпрямить наклонённую бутылку по главной оси маски. img — вырезка на белом,
    mask — bool той же формы. Поворачиваем только явно вытянутые силуэты с наклоном 7–40°."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 500:
        return img
    ev, evec = np.linalg.eigh(np.cov(np.stack([xs, ys]).astype(np.float64)))
    if ev[1] < 3.0 * max(ev[0], 1e-6):
        return img
    vx, vy = evec[:, 1]
    if vy < 0:
        vx, vy = -vx, -vy
    ang = math.degrees(math.atan2(vx, vy))            # 0 — строго вертикально
    if not (7 <= abs(ang) <= 40):
        return img
    rot = img.rotate(-ang, resample=Image.BICUBIC, expand=True, fillcolor=WHITE)
    return crop_to_content(rot, thr=250)[0]


REF_SIDE = 1024                       # эталоны уменьшаем: CLIP всё равно смотрит 224×224, а в базе
                                      # есть картинки до 6337×9506 (94 шт. >10 Мп) — без этого не хватит памяти


def reference_views(img):
    """Варианты эталонного фото, каждый пойдёт в индекс отдельным вектором."""
    img = img.copy()
    img.thumbnail((REF_SIDE, REF_SIDE))
    up = make_upright(img)
    W, H = up.size
    return {
        "orig": pad_to_square(img, WHITE),
        "up_full": pad_to_square(up, WHITE),
        "up_body": up.crop((0, int(H * 0.35), W, H)),            # CLIP сам вырежет центр
        "up_label": pad_to_square(up.crop((0, int(H * 0.45), W, int(H * 0.95))), WHITE),
    }


def list_images(folder):
    if not os.path.isdir(folder):
        return []
    return sorted(f for f in os.listdir(folder)
                  if os.path.splitext(f)[1].lower() in IMG_EXT
                  and os.path.isfile(os.path.join(folder, f)))


# ─── Текст: нормализация и транслитерация ──────────────────────────────────
_TRANSLIT = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e", "ж": "zh",
    "з": "z", "и": "i", "й": "i", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o",
    "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f", "х": "h", "ц": "ts",
    "ч": "ch", "ш": "sh", "щ": "sch", "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu",
    "я": "ya",
}
_LAT2CYR_UP = str.maketrans("ABCEHKMOPTXY", "АВСЕНКМОРТХУ")
_LAT2CYR_LO = str.maketrans("aceopxy", "асеорху")
_HAS_CYR = re.compile(r"[а-яёА-ЯЁ]")
_HAS_LAT = re.compile(r"[A-Za-z]")

STOP = {
    "vino", "vina", "wine", "krasnoe", "beloe", "rozovoe", "roze", "suhoe", "polusuhoe",
    "polusladkoe", "sladkoe", "desertnoe", "igristoe", "shampanskoe", "bryut", "brut",
    "stolovoe", "sortovoe", "marochnoe", "zaschischennogo", "geograficheskogo",
    "ukazaniya", "naimenovaniya", "mesta", "proishozhdeniya", "rossiya", "rossii",
    "god", "goda", "urozhaya", "red", "white", "dry", "rose", "the", "and", "litr",
}


def _fix_mixed(word):
    """OCR часто пишет «KAБEPHE» латиницей вперемешку — приводим к кириллице."""
    if _HAS_CYR.search(word) and _HAS_LAT.search(word):
        word = word.translate(_LAT2CYR_UP).translate(_LAT2CYR_LO)
    return word


def to_latin_tokens(text):
    """Любой текст → список латинских токенов в нижнем регистре (Château → chateau,
    Фанагория → fanagoriya)."""
    s = " ".join(_fix_mixed(w) for w in str(text).split())
    s = s.lower().replace("ё", "е")
    s = "".join(_TRANSLIT.get(c, c) for c in s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.findall(r"[a-z0-9]+", s)


def ocr_tokens(text):
    """Токены OCR-текста для сравнения с базой. Лечит типичные ошибки OCR:
    • логотипы с разрядкой читаются по буквам: «FA N A G 0 R | A», «Т А Б И Я», «INKE R MAN» —
      склеиваем соседние обрывки (в каждой склейке есть кусок ≤2 символов);
    • 0/|/1 внутри слов вместо букв o/i/l."""
    text = re.sub(r"(?<=\s)[|!](?=\s)", " i ", f" {text} ")
    toks = to_latin_tokens(text)
    fixed = []
    for t in toks:
        if re.search(r"[a-z]", t):                         # в слове есть буквы — цифры это буквы
            t = t.translate(str.maketrans("0158", "oisv"))
        fixed.append(t)
    for i, t in enumerate(fixed):                          # одиночные 0/1 среди букв разрядки
        if t in ("0", "1"):
            nb = [fixed[j] for j in (i - 1, i + 1) if 0 <= j < len(fixed)]
            if any(len(x) <= 2 and x.isalpha() for x in nb):
                fixed[i] = "o" if t == "0" else "i"
    out = set(t for t in fixed if len(t) >= 3 and not t.isdigit())
    n = len(fixed)
    i = 0
    while i < n:                                           # целые серии коротких обрывков
        j = i
        while j < n and len(fixed[j]) <= 2 and not fixed[j].isdigit():
            j += 1
        if j - i >= 2:
            out.add("".join(fixed[i:j]))
        i = max(j, i + 1)
    for i in range(n):                                     # 2–3 соседних токена, если есть короткий
        for k in (2, 3):
            part = fixed[i:i + k]
            if len(part) == k and any(len(t) <= 2 for t in part) and not any(t.isdigit() for t in part):
                out.add("".join(part))
    return sorted(t for t in out if len(t) >= 4 or t in fixed)


def is_significant(tok):
    return len(tok) >= 3 and not tok.isdigit() and tok not in STOP


# ─── Датасет, метаданные, разметка ─────────────────────────────────────────
def make_label(row):
    """Уникальное имя вина для индекса. Подписи в dataset.csv неуникальны
    («Фанагория» — у 89 разных вин, «Рислинг» — у 32), а часть — имена файлов
    («Ajla_Vermentino-kopia»). Поэтому: подпись + slug из page_url
    («Фанагория · fanagoriya-formula-q-krasnoe-suhoe»). Slug уникален и содержит
    производителя и название — он же помогает сопоставлять текст с этикетки."""
    cap = re.sub(r"[_]+", " ", (row.get("caption") or "")).strip()
    cap = re.sub(r"\s*-?kopi[ya]a?$", "", cap, flags=re.I).strip()
    slug = (row.get("page_url") or "").rstrip("/").rsplit("/", 1)[-1]
    if not slug:
        return cap
    return f"{cap} · {slug}" if cap else slug


def load_dataset():
    samples = []
    with open(CSV_FILE, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            fn = row.get("image_filename", "")
            label = make_label(row)
            if label and os.path.exists(os.path.join(IMAGES_DIR, fn)):
                samples.append((fn, label))
    return samples


def load_meta():
    if not (os.path.exists(META_PATH) and os.path.exists(EMB_PATH)):
        raise FileNotFoundError(
            "Нет индекса v2. Сначала выполните:\n    python wine_engine.py build")
    with open(META_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


ANNOT_FIELDS = ["image_filename", "true_label", "pred_top1", "rank", "note",
                "ocr_text", "annotated_at"]


def read_annotations(path=ANNOT_PATH):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return {r["image_filename"]: r for r in csv.DictReader(f)}


def write_annotations(ann, path=ANNOT_PATH):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ANNOT_FIELDS, extrasaction="ignore")
        w.writeheader()
        for fn in sorted(ann):
            w.writerow(ann[fn])
    os.replace(tmp, path)


# ─── Построение индекса ─────────────────────────────────────────────────────
def build_index(chunk=16):
    from tqdm import tqdm
    model, preprocess = load_clip()
    samples = load_dataset()
    print(f"Эталонов: {len(samples)}, по 4 ракурса → ~{len(samples) * 4} векторов")
    if get_device() == "cpu":
        print("ВНИМАНИЕ: считаем на CPU, ViT-H будет идти долго (часы). GPU сильно ускорит.")

    labels, label_images, lab2id = [], [], {}
    vecs, vec_label, vec_view = [], [], []
    sides = Counter()
    for s in tqdm(range(0, len(samples), chunk), desc="Индексация"):
        pils, owners, names = [], [], []
        for fn, cap in samples[s:s + chunk]:
            try:
                im = load_image(os.path.join(IMAGES_DIR, fn))
                views = reference_views(im)
                sides[make_upright(im, True)[1]] += 1
            except Exception as e:
                tqdm.write(f"Ошибка {fn}: {e}")
                continue
            if cap not in lab2id:
                lab2id[cap] = len(labels)
                labels.append(cap)
                label_images.append(fn)
            for name, im in views.items():
                pils.append(im)
                owners.append(lab2id[cap])
                names.append(name)
        if pils:
            vecs.append(encode_images(model, preprocess, pils))
            vec_label += owners
            vec_view += names

    emb = np.concatenate(vecs).astype(np.float16)
    np.save(EMB_PATH, emb)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump({"model": MODEL_NAME, "labels": labels, "label_images": label_images,
                   "vec_label": vec_label, "vec_view": vec_view,
                   "neck_side": sides.most_common(1)[0][0] if sides else "left",
                   "built_at": time.strftime("%Y-%m-%d %H:%M:%S")}, f, ensure_ascii=False)
    print(f"Готово: {len(labels)} вин, {emb.shape[0]} векторов × {emb.shape[1]} → {EMB_PATH}")


# ─── Кэш анализа фото ───────────────────────────────────────────────────────
class AnalysisCache:
    def __init__(self, tag, path=CACHE_PATH):
        self.tag, self.path, self.dirty = tag, path, False
        try:
            with open(path, "rb") as f:
                self.data = pickle.load(f)
        except Exception:
            self.data = {}

    def get(self, p):
        e = self.data.get(os.path.basename(p))
        if e and e["tag"] == self.tag and e["mtime"] == os.path.getmtime(p):
            return e["an"]
        return None

    def put(self, p, an):
        self.data[os.path.basename(p)] = {"tag": self.tag, "mtime": os.path.getmtime(p), "an": an}
        self.dirty = True

    def save(self):
        if not self.dirty:
            return
        tmp = self.path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(self.data, f)
        os.replace(tmp, self.path)
        self.dirty = False


# ─── Распознаватель ─────────────────────────────────────────────────────────
# Наборы кадрирований запроса (что подаётся в CLIP). eval сравнивает их все.
VIEW_SETS = {
    "1 кадр (как раньше)": ["full_center"],
    "TTA по всему кадру": ["full_center", "full_pad", "zoom"],
    "YOLO-вырезка на белом": ["seg_white", "seg_center"],
    "YOLO-вырезка + горизонт.": ["seg_white", "seg_center", "seg_horiz"],
    "всё вместе": ["full_center", "full_pad", "zoom", "seg_white", "seg_center", "seg_horiz"],
}
FALLBACK_VIEWS = VIEW_SETS["TTA по всему кадру"]  # если бутылку не вырезали
# Подобрано на полевых фото (ViT-H, 35 фото из каталога): 32/35 top-1, 35/35 top-5
DEFAULT_CFG = {"views": ["seg_white", "seg_center", "seg_horiz", "full_center"],
               "agg": "max", "ocr_weight": 0.3}


def load_config():
    cfg = dict(DEFAULT_CFG)
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg.update(json.load(f))
    except Exception:
        pass
    return cfg


class WineRecognizer:
    def __init__(self, use_ocr=True, use_detector=True, log=print):
        self.log = log
        meta = load_meta()
        self.labels = meta["labels"]
        self.label_images = meta["label_images"]
        self.vec_label = np.asarray(meta["vec_label"], dtype=np.int64)
        self.emb = np.load(EMB_PATH).astype(np.float32)
        self.neck_side = meta.get("neck_side") or estimate_neck_side()
        self.cfg = load_config()
        self.model, self.preprocess = load_clip(log)
        self._prep_text()
        self._ocr_cache = {}
        self.ocr = self._init_ocr() if use_ocr else None
        self.det = self._init_det() if use_detector else None
        tag = (f"{MODEL_NAME}|ocr={self.ocr is not None}|seg={getattr(self, 'det_name', None)}"
               f"|side={self.neck_side}|v3")
        self.cache = AnalysisCache(tag)

    # --- инициализация OCR / детектора (если пакета нет — просто выключаем) ---
    def _init_ocr(self):
        try:
            import easyocr
            self.log("[OCR] загрузка EasyOCR (ru+en; при первом запуске скачивает модели)…")
            t = time.time()
            r = easyocr.Reader(OCR_LANGS, gpu=(get_device() == "cuda"), verbose=False)
            self.log(f"[OCR] готово за {time.time() - t:.0f} с")
            return r
        except Exception as e:
            self.log(f"[OCR] отключён: {e}")
            return None

    def _init_det(self):
        try:
            from ultralytics import YOLO
        except Exception as e:
            self.log(f"[YOLO] отключён (pip install -U ultralytics): {e}")
            return None
        for w in SEG_WEIGHTS:
            try:
                path = os.path.join(BASE_DIR, w)          # веса лежат рядом со скриптом
                self.log(f"[YOLO] загрузка {w}…")
                m = YOLO(path)
                self.det_name = w
                self.log(f"[YOLO] готово: {w}")
                return m
            except Exception as e:
                self.log(f"[YOLO] {w} не загрузился: {e}")
        return None

    # --- текстовая часть ---
    def _prep_text(self):
        n = len(self.labels)
        per_label = [sorted({t for t in to_latin_tokens(l) if is_significant(t)})
                     for l in self.labels]
        df = Counter(t for ts in per_label for t in ts)
        self.vocab = list(df)
        vid = {t: i for i, t in enumerate(self.vocab)}
        self.vocab_len = np.array([len(t) for t in self.vocab], dtype=np.int32)
        idf = np.array([math.log(1 + n / df[t]) for t in self.vocab], dtype=np.float64)
        tok_idx, tok_lab = [], []
        for li, ts in enumerate(per_label):
            for t in ts:
                tok_idx.append(vid[t])
                tok_lab.append(li)
        self._tok_idx = np.array(tok_idx, dtype=np.int64)
        self._tok_lab = np.array(tok_lab, dtype=np.int64)
        self._tok_w = idf[self._tok_idx] if len(tok_idx) else np.zeros(0)
        self._lab_wsum = np.bincount(self._tok_lab, weights=self._tok_w, minlength=n)

    def ocr_scores(self, text):
        """Насколько OCR-текст подтверждает каждую метку, [n_labels] в 0..1:
        сумма IDF-весов найденных слов метки / (сумма весов всех её слов + OCR_SMOOTH).
        Редкие слова (бренды: «табия», «inkerman») весят много, общие («мерло») — мало."""
        n = len(self.labels)
        if not text:
            return np.zeros(n, dtype=np.float32)
        if text in self._ocr_cache:
            return self._ocr_cache[text]
        from rapidfuzz import fuzz
        from rapidfuzz.process import cdist
        q = ocr_tokens(text)
        out = np.zeros(n, dtype=np.float32)
        if q and self.vocab:
            R = np.asarray(cdist(self.vocab, q, scorer=fuzz.ratio, workers=-1), dtype=np.float32)
            P = np.asarray(cdist(self.vocab, q, scorer=fuzz.partial_ratio, workers=-1), dtype=np.float32)
            ql = np.array([len(t) for t in q])[None, :]
            vl = self.vocab_len[:, None]
            # partial — только если OCR склеил слова (токен OCR длиннее слова из базы)
            P = np.where((vl >= 5) & (ql > vl), 0.9 * P, 0)
            best = np.maximum(R, P).max(axis=1) / 100.0
            best[best < OCR_MATCH_MIN] = 0
            num = np.bincount(self._tok_lab, weights=self._tok_w * best[self._tok_idx], minlength=n)
            out = (num / (self._lab_wsum + OCR_SMOOTH)).astype(np.float32)
        self._ocr_cache[text] = out
        return out

    def read_text(self, img, box=None):
        if self.ocr is None:
            return ""
        im = img.crop(box) if box else img
        if max(im.size) < 900:  # мелкий кроп — увеличим для OCR
            k = 900 / max(im.size)
            im = im.resize((int(im.width * k), int(im.height * k)), Image.LANCZOS)
        try:
            res = self.ocr.readtext(np.asarray(im))
        except Exception as e:
            self.log(f"[OCR] ошибка: {e}")
            return ""
        return " ".join(t for _, t, conf in res if conf >= 0.2)

    # --- сегментация бутылки ---
    def segment_bottle(self, img):
        """Центральная бутылка: {'box', 'poly', 'conf'} или None.
        Бутылка, которая далеко от центра кадра (соседняя на полке), не берётся —
        тогда работаем по всему кадру (крупный план этикетки YOLO часто не видит)."""
        dev = 0 if get_device() == "cuda" else "cpu"
        r = self.det.predict(img, classes=[BOTTLE_CLASS], conf=SEG_CONF, verbose=False,
                             device=dev, retina_masks=True)[0]
        if r.boxes is None or len(r.boxes) == 0 or r.masks is None:
            return None
        W, H = img.size
        cx, cy, half_diag = W / 2, H / 2, math.hypot(W, H) / 2
        best, best_s = None, -1.0
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        for (x0, y0, x1, y1), c, poly in zip(boxes, confs, r.masks.xy):
            if len(poly) < 3:
                continue
            if not (x0 - 0.1 * W <= cx <= x1 + 0.1 * W):   # бутылка сбоку — не наша
                continue
            area = (x1 - x0) * (y1 - y0) / (W * H)
            if area < 0.04:
                continue
            d = math.hypot((x0 + x1) / 2 - cx, (y0 + y1) / 2 - cy) / half_diag
            s = area * (1 - 0.8 * d) * (0.2 + float(c))
            if s > best_s:
                best_s = s
                best = {"box": tuple(int(round(v)) for v in (max(0, x0), max(0, y0), min(W, x1), min(H, y1))),
                        "poly": [(float(x), float(y)) for x, y in poly], "conf": float(c)}
        return best

    @staticmethod
    def solid_mask(size, poly, box):
        """Маска бутылки без дыр и прилипших кусков соседей: растеризуем контур,
        морфологическим открытием срезаем тонкие отростки, берём самый большой кусок
        и его выпуклую оболочку (бутылка выпуклая — так закрываются «откушенные» края)."""
        from PIL import ImageDraw
        m = Image.new("L", size, 0)
        ImageDraw.Draw(m).polygon(poly, fill=255)
        try:
            import cv2
        except Exception:
            return m, poly
        a = np.asarray(m).copy()
        k = max(3, int(0.05 * (box[2] - box[0])) | 1)
        opened = cv2.morphologyEx(a, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        cnts, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return m, poly
        c = max(cnts, key=cv2.contourArea)
        hull = cv2.convexHull(c)
        if cv2.contourArea(hull) > 1.3 * max(cv2.contourArea(c), 1):
            return m, poly                             # форма странная — оставим как есть
        out = np.zeros_like(a)
        cv2.fillPoly(out, [hull], 255)
        return Image.fromarray(out), [(float(x), float(y)) for x, y in hull.reshape(-1, 2)]

    def cutout(self, img, seg):
        """Бутылка по маске на белом фоне, обрезанная и выпрямленная."""
        m, seg["shape"] = self.solid_mask(img.size, seg["poly"], seg["box"])
        cut = Image.new("RGB", img.size, WHITE)
        cut.paste(img, (0, 0), m)
        box = m.getbbox() or seg["box"]
        cut, mask = cut.crop(box), np.asarray(m.crop(box)) > 0
        return deskew(cut, mask)

    # --- анализ фото (тяжёлая часть, кэшируется) ---
    def analyze(self, path, use_cache=True):
        if use_cache:
            an = self.cache.get(path)
            if an is not None:
                return an
        img = load_image(path)
        img.thumbnail((WORK_SIDE, WORK_SIDE))
        seg = self.segment_bottle(img) if self.det is not None else None
        views = {
            "full_center": img,
            "full_pad": pad_to_square(img, GRAY),
            "zoom": center_crop_frac(img, 0.6),
        }
        if seg:
            cut = self.cutout(img, seg)
            views["seg_white"] = pad_to_square(cut, WHITE)
            views["seg_center"] = cut                      # CLIP сам возьмёт центр — зону этикетки
            if self.neck_side in ("left", "right"):          # эталоны лежат — дадим и лежачий вариант
                views["seg_horiz"] = pad_to_square(upright_to_side(cut, self.neck_side), WHITE)
        embs = encode_images(self.model, self.preprocess, list(views.values()))
        poly = (seg.get("shape") or seg["poly"]) if seg else None   # контур, который реально вырезан
        if poly and len(poly) > 200:                       # для отрисовки хватит 200 точек
            poly = poly[::math.ceil(len(poly) / 200)]
        an = {
            "views": {k: embs[i] for i, k in enumerate(views)},
            "ocr_text": self.read_text(img, seg["box"] if seg else None),
            "box": seg["box"] if seg else None,
            "poly": poly,
            "seg_conf": seg["conf"] if seg else None,
            "size": img.size,
        }
        self.cache.put(path, an)
        return an

    # --- скоринг (лёгкая часть) ---
    def clip_scores(self, q):
        sims = self.emb @ q
        out = np.full(len(self.labels), -1.0, dtype=np.float32)
        np.maximum.at(out, self.vec_label, sims)  # по каждой вине — лучший ракурс
        return out

    def scores(self, an, views=None, agg=None, ocr_weight=None):
        """views/agg/ocr_weight по умолчанию — из engine_config.json (его пишет eval)."""
        views = views or self.cfg["views"]
        agg = agg or self.cfg["agg"]
        ocr_weight = self.cfg["ocr_weight"] if ocr_weight is None else ocr_weight
        v = an["views"]
        keys = [k for k in views if k in v]
        if not any(k.startswith("seg_") for k in keys) and any(k.startswith("seg_") for k in views):
            keys = [k for k in dict.fromkeys(keys + FALLBACK_VIEWS) if k in v]  # не вырезали — весь кадр
        if agg == "max":
            clip = np.max([self.clip_scores(v[k]) for k in keys], axis=0)
        else:
            q = np.mean([v[k] for k in keys], axis=0)
            q = q / (np.linalg.norm(q) + 1e-8)
            clip = self.clip_scores(q)
        ocr = self.ocr_scores(an.get("ocr_text", "")) if ocr_weight > 0 else np.zeros_like(clip)
        return clip + ocr_weight * ocr, clip, ocr

    def top(self, an, k=10, **kw):
        final, clip, ocr = self.scores(an, **kw)
        idx = np.argsort(-final)[:k]
        return [{"label_id": int(i), "label": self.labels[i], "image": self.label_images[i],
                 "score": float(final[i]), "clip": float(clip[i]), "ocr": float(ocr[i])}
                for i in idx]

    def predict(self, path, k=5):
        an = self.analyze(path)
        self.cache.save()
        return self.top(an, k=k), an


# ─── Оценка точности по разметке ────────────────────────────────────────────
def evaluate(no_ocr=False, no_det=False, show_errors=15, save=True):
    from tqdm import tqdm
    ann = read_annotations()
    if not ann:
        print(f"Нет разметки ({ANNOT_PATH}). Сначала разметьте фото в label_tool.py")
        return
    rec = WineRecognizer(use_ocr=not no_ocr, use_detector=not no_det)
    lab2id = {l: i for i, l in enumerate(rec.labels)}
    items, n_out, n_unknown = [], 0, 0
    for fn, r in ann.items():
        t = r.get("true_label", "")
        if t == NOT_IN_DB:
            n_out += 1
        elif t not in lab2id:
            n_unknown += 1
        elif os.path.exists(os.path.join(TEST_DIR, fn)):
            items.append((fn, lab2id[t]))
    print(f"Фото с вином из базы: {len(items)} · «нет в базе»: {n_out} · метка не найдена в индексе: {n_unknown}")
    if not items:
        return

    analyses = [(fn, rec.analyze(os.path.join(TEST_DIR, fn)), t)
                for fn, t in tqdm(items, desc="Анализ фото")]
    rec.cache.save()
    n_seg = sum(1 for _, an, _ in analyses if an.get("box"))
    print(f"YOLO вырезал бутылку на {n_seg} из {len(analyses)} фото")

    def run(views, agg, w):
        ranks = []
        for _, an, t in analyses:
            final = rec.scores(an, views=views, agg=agg, ocr_weight=w)[0]
            ranks.append(1 + int((final > final[t]).sum()))
        r = np.array(ranks)
        return r, float(np.mean(1.0 / r))

    def row(name, r, mrr):
        print(f"{name:<40}{np.mean(r == 1):>8.1%}{np.mean(r <= 5):>8.1%}{np.mean(r <= 10):>8.1%}{mrr:>8.3f}")

    print(f"\n{'Только CLIP':<40}{'top-1':>8}{'top-5':>8}{'top-10':>8}{'MRR':>8}")
    best = None
    for name, views in VIEW_SETS.items():
        for agg in ("mean", "max"):
            if agg == "max" and len(views) == 1:
                continue
            r, mrr = run(views, agg, 0.0)
            row(f"{name}, {agg}", r, mrr)
            if best is None or mrr > best[3]:
                best = (name, views, agg, mrr, r, 0.0)
    if rec.ocr is not None:
        ranks = []
        for _, an, t in analyses:
            o = rec.ocr_scores(an.get("ocr_text", ""))
            ranks.append(1 + int((o > o[t]).sum()) if o[t] > 0 else len(o))
        r_ocr = np.array(ranks)
        row("Только текст (OCR), без CLIP", r_ocr, float(np.mean(1.0 / r_ocr)))
        name, views, agg = best[:3]
        print(f"\n+ OCR к лучшему варианту «{name}, {agg}»:")
        for w in (0.1, 0.2, 0.3, 0.5, 0.8, 1.2):
            r, mrr = run(views, agg, w)
            row(f"  вес OCR {w}", r, mrr)
            if mrr > best[3]:
                best = (name, views, agg, mrr, r, w)

    name, views, agg, mrr, r, w = best
    print(f"\nЛучшее: «{name}», {agg}, вес OCR {w} → top-1 {np.mean(r == 1):.1%}, MRR {mrr:.3f}")
    if save:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump({"views": views, "agg": agg, "ocr_weight": w, "n_eval": len(analyses),
                       "mrr": mrr}, f, ensure_ascii=False, indent=2)
        print(f"Сохранено в {CONFIG_PATH} — predict и label_tool теперь используют эти настройки.")
    print(f"\nПромахи top-1 (до {show_errors}):")
    shown = 0
    for (fn, an, t), rank in zip(analyses, r):
        if rank > 1 and shown < show_errors:
            pred = rec.top(an, k=1, views=views, agg=agg, ocr_weight=w)[0]["label"]
            print(f"  {fn}: верно «{rec.labels[t]}» (место {rank}), выдано «{pred}»")
            shown += 1


def segtest(out="segtest.jpg", limit=None, per_sheet=30):
    """Контрольный лист: исходное фото с контуром + что пойдёт в CLIP. Без CLIP и индекса."""
    from PIL import ImageDraw
    rec = WineRecognizer.__new__(WineRecognizer)
    rec.log = print
    rec.det = rec._init_det()
    if rec.det is None:
        return
    rec.neck_side = "left"
    from PIL import ImageFont
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    tiles = []
    for fn in list_images(TEST_DIR)[:limit]:
        img = load_image(os.path.join(TEST_DIR, fn))
        img.thumbnail((WORK_SIDE, WORK_SIDE))
        seg = rec.segment_bottle(img)
        vis = img.copy()
        cut = Image.new("RGB", (10, 10), (230, 230, 230))
        if seg:
            cut = rec.cutout(img, seg)
            shape = seg.get("shape") or seg["poly"]
            ImageDraw.Draw(vis).line(shape + [shape[0]], fill=(0, 255, 0), width=5)
        vis.thumbnail((300, 400))
        cut.thumbnail((300, 400))
        t = Image.new("RGB", (610, 425), "white")
        t.paste(vis, (0, 0))
        t.paste(cut, (310, 0))
        ImageDraw.Draw(t).text((5, 405), f"{fn[:40]}  {'conf %.2f' % seg['conf'] if seg else 'NOT FOUND -> full frame'}",
                               fill=(200, 0, 0) if not seg else (0, 0, 0), font=font)
        tiles.append(t)
        print(fn, "→", f"conf {seg['conf']:.2f}" if seg else "весь кадр")
    if not tiles:
        print(f"В {TEST_DIR} нет картинок")
        return
    cols = 3
    base, ext = os.path.splitext(out)
    for n, s0 in enumerate(range(0, len(tiles), per_sheet), 1):
        chunk = tiles[s0:s0 + per_sheet]
        rows = math.ceil(len(chunk) / cols)
        sheet = Image.new("RGB", (cols * 615, rows * 430), (128, 128, 128))
        for i, t in enumerate(chunk):
            sheet.paste(t, ((i % cols) * 615, (i // cols) * 430))
        name = out if len(tiles) <= per_sheet else f"{base}_{n}{ext}"
        sheet.save(os.path.join(BASE_DIR, name), quality=85)
        print(f"Сохранено: {os.path.join(BASE_DIR, name)}")


# ─── CLI ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Распознавание вин по фото")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build", help="построить индекс v2")
    p = sub.add_parser("predict", help="распознать фото")
    p.add_argument("image")
    p.add_argument("--top", type=int, default=5)
    e = sub.add_parser("eval", help="точность по разметке")
    e.add_argument("--no-ocr", action="store_true")
    e.add_argument("--no-det", action="store_true")
    e.add_argument("--no-save", action="store_true", help="не 