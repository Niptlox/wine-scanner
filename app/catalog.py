# -*- coding: utf-8 -*-
"""Каталог «Своё Вино» на уровне slug + распознавание, уверенность, аналоги, сомелье.

Индекс (wine_index_v2.npy) строится wine_engine.py build: 4 вектора на строку dataset.csv
(orig, up_full, up_body, up_label), строки идут в порядке CSV. Метки в старых индексах —
подписи (у 89 вин подпись «Фанагория»), поэтому здесь векторы переназначаются на slug:
вектор 4k..4k+3 → k-я строка CSV → её slug.
"""
import csv
import math
import os
import re
import time

import numpy as np

import wine_engine as we

# ─── атрибуты из slug/подписи ───────────────────────────────────────────────
COLORS = [("rozov", "розовое"), ("roze", "розовое"), ("rose", "розовое"), ("krasn", "красное"),
          ("red", "красное"), ("belo", "белое"), ("belyy", "белое"), ("belyj", "белое"),
          ("blanc", "белое"), ("white", "белое"), ("oranzh", "оранжевое"), ("orange", "оранжевое")]
SWEET = [("polusladk", "полусладкое"), ("polusuh", "полусухое"), ("ekstra-bryut", "экстра брют"),
         ("extra-brut", "экстра брют"), ("bryut", "брют"), ("brut", "брют"), ("sladk", "сладкое"),
         ("desert", "десертное"), ("suhoe", "сухое"), ("suh", "сухое"), ("semi-sweet", "полусладкое")]
GRAPES = {
    "kaberne-sovinon": "Каберне Совиньон", "cabernet-sauvignon": "Каберне Совиньон",
    "kaberne-fran": "Каберне Фран", "cabernet-franc": "Каберне Фран", "merlo": "Мерло", "merlot": "Мерло",
    "saperavi": "Саперави", "pino-nuar": "Пино Нуар", "pinot-noir": "Пино Нуар", "sira": "Сира",
    "syrah": "Сира", "krasnostop": "Красностоп", "shardone": "Шардоне", "chardonnay": "Шардоне",
    "risling": "Рислинг", "riesling": "Рислинг", "sovinon-blan": "Совиньон Блан",
    "sauvignon-blanc": "Совиньон Блан", "aligote": "Алиготе", "rkatsiteli": "Ркацители",
    "rkacziteli": "Ркацители", "muskat": "Мускат", "muscat": "Мускат", "tsitron": "Цитрон Магарача",
    "czitron": "Цитрон Магарача", "pino-gri": "Пино Гри", "vione": "Вионье", "kokur": "Кокур",
    "bastardo": "Бастардо", "rubin-golodrigi": "Рубин Голодриги", "tsimlyanskiy": "Цимлянский чёрный",
    "marselan": "Марселан", "gryuner": "Грюнер", "shenen": "Шенен Блан", "traminer": "Траминер",
    "pti-verdo": "Пти Вердо", "petit-verdot": "Пти Вердо", "kefesiya": "Кефесия",
}
SPARKLING = ("igrist", "bryut", "brut", "petnat", "pet-nat", "shampan", "spumante", "frizzante")

PAIRING = {
    "красное": "стейк, баранина, выдержанные сыры, блюда на гриле",
    "белое": "рыба, морепродукты, птица, мягкие сыры, салаты",
    "розовое": "лёгкие закуски, паста, пицца, азиатская кухня, летний стол",
    "оранжевое": "пряные блюда, ферментированные закуски, твёрдые сыры",
    "игристое": "аперитив, устрицы, сыры, праздничный стол",
    "сладкое": "десерты, фрукты, голубые сыры, фуа-гра",
}


STRAPI_CSV = os.path.join(we.BASE_DIR, os.environ.get("STRAPI_CSV", "strapi_catalog.csv"))
CATEGORY_COLOR = {"белое": "белое", "красное": "красное", "розовое": "розовое", "оранжевое": "оранжевое"}


def load_strapi(path=STRAPI_CSV):
    """Выгрузка Strapi: название, категория, цвет, регион, сорт, описание, винодельня, slug.
    Ссылки на фото там не совпадают с реальными файлами — фото берём из dataset.csv по slug."""
    if not os.path.exists(path):
        return {}
    csv.field_size_limit(10 ** 9)
    out = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            slug = (r.get("Slug") or "").strip()
            if slug and slug not in out:                      # в выгрузке каждая строка дублируется
                out[slug] = {k: (v or "").strip() for k, v in r.items()}
    return out


def slug_of(row):
    return (row.get("page_url") or "").rstrip("/").rsplit("/", 1)[-1]


def attributes(row, info=None):
    info = info or {}
    text = (slug_of(row) + " " + " ".join(we.to_latin_tokens(row.get("caption", "") + " "
                                                              + info.get("Название вина", "")))).lower()
    color = CATEGORY_COLOR.get(info.get("Категория", "").lower()) or next((v for k, v in COLORS if k in text), None)
    sweet = next((v for k, v in SWEET if k in text), None)
    grapes = [g.strip() for g in info.get("Сорт винограда", "").split(",") if g.strip()]
    if not grapes:
        for k, v in GRAPES.items():
            if k in text and v not in grapes:
                grapes.append(v)
    sparkling = any(k in text for k in SPARKLING)
    kind = "игристое" if sparkling else color
    if sweet in ("полусладкое", "сладкое", "десертное") and not sparkling:
        pair = PAIRING["сладкое"] if sweet != "полусладкое" else PAIRING.get(color or "белое")
    else:
        pair = PAIRING.get(kind or "", PAIRING["красное"] if color == "красное" else PAIRING["белое"])
    return {"color": color, "sweetness": sweet, "sparkling": sparkling, "grapes": grapes,
            "pairing": pair}


GENERIC = {"vinodelnya", "usadba", "dom", "pomeste", "shato", "chateau", "imenie", "kuban", "valeriy",
           "valerij", "vino", "vina", "semeynaya", "zolotaya", "novyy", "novyj", "derbent", "abrau", "alma"}


def producer_key(row):
    """Ключ винодельни для «аналогов из других виноделен»: первые слова slug
    (для «vinodelnya-…», «usadba-…» и т.п. — два слова); у коротких slug — первое слово подписи."""
    toks = [t for t in re.split(r"[-_]", slug_of(row)) if t]
    if len(toks) >= 3:
        return "-".join(toks[:2]) if toks[0] in GENERIC else toks[0]
    cap = we.to_latin_tokens(row.get("caption", ""))
    return cap[0] if cap else (toks[0] if toks else "")


# ─── распознаватель на уровне slug ───────────────────────────────────────────
class SlugRecognizer(we.WineRecognizer):
    def __init__(self, use_ocr=True, use_detector=True, log=print):
        meta = we.load_meta()
        emb = np.load(we.EMB_PATH).astype(np.float32)
        with open(we.CSV_FILE, "r", encoding="utf-8-sig", newline="") as f:
            all_rows = list(csv.DictReader(f))
        # те же строки и в том же порядке, что использовал build_index
        rows = [r for r in all_rows if (r.get("caption") or "").strip()
                and os.path.exists(os.path.join(we.IMAGES_DIR, r["image_filename"]))]
        n_views = 4
        if emb.shape[0] != n_views * len(rows):
            # картинки лежат не рядом — берём все строки с подписью (порядок build_index)
            rows = [r for r in all_rows if (r.get("caption") or "").strip()]
        if emb.shape[0] != n_views * len(rows):
            raise RuntimeError(f"Индекс ({emb.shape[0]} векторов) не соответствует dataset.csv "
                               f"({len(rows)} строк × 4). Пересоберите: python wine_engine.py build")
        self.log = log
        self.rows = rows
        self.slugs = [slug_of(r) for r in rows]
        self.strapi = load_strapi()
        self.info = [self.strapi.get(slug_of(r), {}) for r in rows]
        # текст для OCR-сопоставления: подпись · slug + винодельня + полное название из Strapi
        self.labels = [we.make_label(r) + (" " + i.get("Винодельня", "") + " " + i.get("Название вина", "")
                                           if i else "") for r, i in zip(rows, self.info)]
        log(f"[Каталог] {len(rows)} вин, описания Strapi найдены для {sum(1 for i in self.info if i)}")
        self.label_images = [r["image_filename"] for r in rows]
        self.vec_label = np.repeat(np.arange(len(rows)), n_views)
        self.emb = emb
        self.neck_side = meta.get("neck_side") or "up"
        self.cfg = we.load_config()
        self.model, self.preprocess = we.load_clip(log)
        self._prep_text()
        self._ocr_cache = {}
        self.ocr = self._init_ocr() if use_ocr else None
        self.det = self._init_det() if use_detector else None
        self.cache = we.AnalysisCache("api-" + str(time.time()))
        self.attrs = [attributes(r, i) for r, i in zip(rows, self.info)]
        self.producers = [(i.get("Винодельня") or producer_key(r)).lower() for r, i in zip(rows, self.info)]
        # центроиды эталонов для «похожих вин»
        e = emb.reshape(len(rows), n_views, -1).mean(1)
        self.centroids = e / np.linalg.norm(e, axis=1, keepdims=True)
        self.slug2i = {s: i for i, s in enumerate(self.slugs)}

    # --- распознавание + уверенность ---
    def recognize(self, path, k=5, temperature=0.02):
        t0 = time.time()
        an = self.analyze(path, use_cache=False)
        final, clip, ocr = self.scores(an)
        idx = np.argsort(-final)[:max(k, 10)]
        z = (final[idx] - final[idx[0]]) / temperature
        p = np.exp(z) / np.exp(z).sum()                   # softmax по топ-10 → «уверенность»
        top = [{"slug": self.slugs[i], "name": self.display_name(i), "score": round(float(final[i]), 4),
                "clip": round(float(clip[i]), 4), "ocr": round(float(ocr[i]), 3),
                "confidence": round(float(p[j]), 4)} for j, i in enumerate(idx[:k])]
        margin = float(final[idx[0]] - final[idx[1]])
        conf1 = float(p[0])
        found = conf1 >= float(os.environ.get("MIN_CONFIDENCE", 0.35)) and float(clip[idx[0]]) >= float(
            os.environ.get("MIN_CLIP", 0.45))
        return {
            "slug": self.slugs[idx[0]],
            "found": bool(found),
            "confidence_top1": round(conf1, 4),
            "confidence_top5": round(float(p[:5].sum()), 4),
            "margin_top1_top2": round(margin, 4),
            "top5": top,
            "ocr_text": an.get("ocr_text", ""),
            "bottle_box": an.get("box"),
            "time_ms": int((time.time() - t0) * 1000),
        }

    def display_name(self, i):
        return (self.info[i].get("Название вина") or (self.rows[i].get("caption") or "").strip()).replace("_", " ")

    # --- карточка ---
    def card(self, slug):
        i = self.slug2i[slug]
        r = self.rows[i]
        a = self.attrs[i]
        info = self.info[i]
        return {"slug": slug, "name": self.display_name(i),
                "winery": info.get("Винодельня") or None, "region": info.get("Регион") or None,
                "category": info.get("Категория") or (a["color"] or "").capitalize() or None,
                "appearance": info.get("Цвет") or None, "description": info.get("Описание") or None,
                "url": r.get("page_url"), "image": f"/img/{r['image_filename']}", **a}

    def analogs(self, slug, n=6, other_producers=True):
        i = self.slug2i[slug]
        sims = self.centroids @ self.centroids[i]
        a = self.attrs[i]
        bonus = np.array([0.05 * (x["color"] == a["color"]) + 0.03 * (x["sweetness"] == a["sweetness"])
                          + 0.03 * len(set(x["grapes"]) & set(a["grapes"])) for x in self.attrs])
        order = np.argsort(-(sims + bonus))
        out = []
        for j in order:
            if j == i or (other_producers and self.producers[j] == self.producers[i]):
                continue
            out.append(self.card(self.slugs[j]))
            if len(out) >= n:
                break
        return out

    # --- «Цифровой сомелье»: подбор по ответам ---
    def sommelier(self, dish=None, color=None, sweetness=None, sparkling=None, like_slug=None, region=None, n=5):
        want = {}
        dish_map = {"мясо": "красное", "стейк": "красное", "рыба": "белое", "морепродукты": "белое",
                    "птица": "белое", "сыр": None, "десерт": None, "паста": "розовое", "праздник": None}
        if dish in dish_map and dish_map[dish]:
            want["color"] = dish_map[dish]
        if dish == "десерт":
            want["sweet_set"] = {"сладкое", "полусладкое", "десертное"}
        if dish == "праздник":
            sparkling = True
        if color:
            want["color"] = color
        scores = np.zeros(len(self.rows))
        for j, x in enumerate(self.attrs):
            s = 0.0
            if "color" in want:
                s += 2.0 if x["color"] == want["color"] else -2.0
            if sweetness:
                s += 1.5 if x["sweetness"] == sweetness else -0.5
            if "sweet_set" in want:
                s += 1.5 if x["sweetness"] in want["sweet_set"] else -1.0
            if sparkling is not None:
                s += 1.5 if x["sparkling"] == bool(sparkling) else -1.5
            if region:
                s += 1.5 if self.info[j].get("Регион") == region else -1.0
            s += 0.3 * bool(self.info[j])              # вина с полным описанием — чуть выше
            scores[j] = s
        if like_slug and like_slug in self.slug2i:
            scores = scores + 3.0 * (self.centroids @ self.centroids[self.slug2i[like_slug]])
            scores[self.slug2i[like_slug]] = -1e9
        order = np.argsort(-scores)[:n]
        tips = PAIRING.get(want.get("color") or ("игристое" if sparkling else ""), "")
        return {"tip": tips, "wines": [self.card(self.slugs[j]) for j in order]}
