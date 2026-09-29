# -*- coding: utf-8 -*-
"""
label_tool.py — окно ручной разметки фото из images_test.

Слева фото (зелёный контур — бутылка, которую YOLO вырезал и положил на белый фон;
если контура нет — распознавание шло по всему кадру), справа 8 лучших кандидатов
с миниатюрами эталонов из базы. Отмечаете правильный — разметка сразу пишется
в test_annotations.csv и открывается следующее фото. Распознавание идёт в фоне,
следующие 2 фото считаются заранее; результаты кэшируются в analysis_cache.pkl.

Клавиши:
  1–8     выбрать кандидата          Enter   = кандидат №1
  0       «нет в базе» (если в поиске что-то набрано — сохранится как примечание)
  ← / →   листать                    N       следующая неразмеченная / непроверенная
  Пробел  черновик верный (если файл разметки заполнен заранее — примечание «черновик…»)
  /  или Ctrl+F   поиск по всей базе (Enter — выбрать выделенное, ↓ — к списку, Esc — очистить)
Если поиск пуст, в списке под кандидатами — кандидаты 9–50.
"""
import os
import time
import queue
import threading
import itertools
import tkinter as tk
from tkinter import ttk, messagebox

from PIL import Image, ImageTk, ImageDraw

import wine_engine as we

N_CAND = 8
TOP_K = 50
CANVAS_W, CANVAS_H = 620, 840
THUMB = (160, 190)
MAX_LIST = 300
OK_BG = "#b7ebc0"
DRAFT_BG = "#ffe7a3"
DRAFT = "черновик"   # примечание у предразметки — такие фото считаются непроверенными


class LabelTool:
    def __init__(self, root):
        self.root = root
        root.title("Разметка фото вин — images_test")
        try:
            root.state("zoomed")
        except tk.TclError:
            root.geometry("1500x950")

        meta = we.load_meta()  # FileNotFoundError, если индекс не построен
        self.labels = meta["labels"]
        self.label_images = meta["label_images"]
        self.search_index = [(l.lower().replace("ё", "е"), " ".join(we.to_latin_tokens(l)))
                             for l in self.labels]

        self.files = we.list_images(we.TEST_DIR)
        self.ann = we.read_annotations()
        self.idx = next((i for i, f in enumerate(self.files) if not self.is_done(f)), 0)

        self.results, self.tops = {}, {}
        self.jobs, self.msgs = queue.PriorityQueue(), queue.Queue()
        self.counter = itertools.count()
        self.thumbs = {}
        self.list_items = []
        self.cur_img = self.disp_base = self.tk_img = None

        self._build_ui()
        threading.Thread(target=self._worker, daemon=True).start()
        root.after(100, self._poll)
        self.show()

    def is_done(self, fn):
        """Размечено человеком (черновик, сделанный заранее, считается непроверенным)."""
        r = self.ann.get(fn)
        return bool(r) and not str(r.get("note", "")).startswith(DRAFT)

    # ─── фоновое распознавание ─────────────────────────────────────────────
    def _worker(self):
        def log(s):
            print(s)
            self.msgs.put(("status", s))

        log("Загружаю CLIP, OCR и детектор… первый запуск может занять несколько минут")
        try:
            rec = we.WineRecognizer(log=log)
        except Exception as e:
            self.msgs.put(("fatal", f"{type(e).__name__}: {e}"))
            return
        log(f"Готово: {len(rec.labels)} вин в базе · OCR {'вкл' if rec.ocr else 'выкл'} · "
            f"YOLO {getattr(rec, 'det_name', 'выкл') if rec.det else 'выкл'} · {we.get_device()} · "
            f"кадры: {', '.join(rec.cfg['views'])} · вес OCR {rec.cfg['ocr_weight']}")
        done = set()
        while True:
            _, _, fn = self.jobs.get()
            if fn in done:
                continue
            done.add(fn)
            try:
                an = rec.analyze(os.path.join(we.TEST_DIR, fn))
                top = rec.top(an, k=TOP_K)
                rec.cache.save()
            except Exception as e:
                an, top = {"error": f"{type(e).__name__}: {e}"}, []
            self.msgs.put(("result", fn, an, top))

    def _request(self, i, prio):
        if 0 <= i < len(self.files) and self.files[i] not in self.results:
            self.jobs.put((prio, next(self.counter), self.files[i]))

    def _poll(self):
        try:
            while True:
                m = self.msgs.get_nowait()
                if m[0] == "status":
                    self.status.set(m[1])
                elif m[0] == "fatal":
                    self.status.set("Ошибка загрузки модели: " + m[1])
                    messagebox.showerror("Ошибка загрузки модели", m[1])
                elif m[0] == "result":
                    _, fn, an, top = m
                    self.results[fn], self.tops[fn] = an, top
                    if fn == self.cur_fn():
                        self._render_image()
                        self._render_candidates()
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    # ─── интерфейс ─────────────────────────────────────────────────────────
    def _build_ui(self):
        top = ttk.Frame(self.root, padding=(8, 6))
        top.pack(fill="x")
        self.info, self.stats, self.cur_ann = tk.StringVar(), tk.StringVar(), tk.StringVar()
        ttk.Label(top, textvariable=self.info, font=("Segoe UI", 11, "bold")).pack(side="left")
        ttk.Label(top, textvariable=self.stats).pack(side="right")

        self.status = tk.StringVar()
        ttk.Label(self.root, textvariable=self.status, relief="sunken", anchor="w",
                  padding=(6, 2)).pack(fill="x", side="bottom")

        body = ttk.Frame(self.root)
        body.pack(fill="both", expand=True, padx=8)
        self.canvas = tk.Canvas(body, width=CANVAS_W, height=CANVAS_H, bg="#1e1e1e",
                                highlightthickness=0)
        self.canvas.pack(side="left", anchor="n")

        right = ttk.Frame(body, padding=(10, 0))
        right.pack(side="left", fill="both", expand=True)
        ttk.Label(right, textvariable=self.cur_ann, font=("Segoe UI", 10, "bold"),
                  foreground="#1a7f37", wraplength=760).pack(anchor="w")
        ttk.Label(right, text="Кандидаты: 1–8 — выбрать, Enter — №1, 0 — нет в базе").pack(
            anchor="w", pady=(4, 2))

        grid = ttk.Frame(right)
        grid.pack(anchor="w")
        self.blank = ImageTk.PhotoImage(Image.new("RGB", THUMB, (235, 235, 235)))
        self.btns = []
        for i in range(N_CAND):
            b = tk.Button(grid, image=self.blank, compound="top", width=180, height=270,
                          wraplength=175, justify="center", font=("Segoe UI", 9),
                          relief="groove", command=lambda i=i: self.pick_candidate(i))
            b.grid(row=i // 4, column=i % 4, padx=3, pady=3)
            self.btns.append(b)
        self.btn_bg = self.btns[0].cget("bg")

        self.ocr_var = tk.StringVar()
        ttk.Label(right, textvariable=self.ocr_var, wraplength=760, foreground="#555").pack(
            anchor="w", pady=(6, 4))

        sf = ttk.LabelFrame(right, text="Поиск по базе (/ или Ctrl+F). Пусто — кандидаты 9–50",
                            padding=6)
        sf.pack(fill="both", expand=True)
        self.search_var = tk.StringVar()
        self.search = ttk.Entry(sf, textvariable=self.search_var, font=("Segoe UI", 11))
        self.search.pack(fill="x")
        row = ttk.Frame(sf)
        row.pack(fill="both", expand=True, pady=(4, 0))
        self.lb = tk.Listbox(row, height=10, font=("Segoe UI", 10), exportselection=False)
        sb = ttk.Scrollbar(row, command=self.lb.yview)
        self.lb.config(yscrollcommand=sb.set)
        self.lb.pack(side="left", fill="both", expand=True)
        sb.pack(side="left", fill="y")
        self.preview = ttk.Label(row, image=self.blank)
        self.preview.pack(side="left", padx=8)
        ttk.Button(sf, text="Выбрать из списка (Enter)", command=self.pick_from_list).pack(
            anchor="e", pady=(4, 0))

        nav = ttk.Frame(right)
        nav.pack(fill="x", pady=6)
        for text, cmd in [("← Назад", lambda: self.go(-1)),
                          ("Вперёд →", lambda: self.go(1)),
                          ("К неразмеченной (N)", self.next_unlabeled),
                          ("Черновик верный (Пробел)", self.confirm_draft),
                          ("Нет в базе (0)", self.mark_not_in_db),
                          ("Снять разметку", self.clear_annotation)]:
            ttk.Button(nav, text=text, command=cmd).pack(side="left", padx=2)

        self.search_var.trace_add("write", lambda *a: self.refresh_list())
        self.search.bind("<Return>", lambda e: self._pick_list_break())
        self.search.bind("<Down>", self._focus_list)
        self.search.bind("<Escape>", self._clear_search)
        self.lb.bind("<<ListboxSelect>>", lambda e: self._update_preview())
        self.lb.bind("<Double-Button-1>", lambda e: self.pick_from_list())
        self.lb.bind("<Return>", lambda e: self._pick_list_break())
        self.lb.bind("<Escape>", self._clear_search)
        self.root.bind("<Key>", self.on_key)

    def on_key(self, e):
        if isinstance(e.widget, tk.Entry):
            return None
        ctrl = bool(e.state & 0x4)
        # keycode — чтобы работало и в русской раскладке (F=70, N=78, /=191 на Windows)
        if ctrl and (e.keycode == 70 or e.keysym.lower() == "f"):
            return self._focus_search()
        if ctrl:
            return None
        if e.char and e.char in "12345678":
            self.pick_candidate(int(e.char) - 1)
        elif e.char == "0":
            self.mark_not_in_db()
        elif e.keysym in ("Return", "KP_Enter"):
            self.pick_candidate(0)
        elif e.keysym == "space":
            self.confirm_draft()
        elif e.keysym in ("Right", "Next"):
            self.go(1)
        elif e.keysym in ("Left", "Prior"):
            self.go(-1)
        elif e.keycode == 78 or e.char in ("n", "N", "т", "Т"):
            self.next_unlabeled()
        elif e.char == "/" or e.keycode == 191:
            return self._focus_search()
        return None

    def _focus_search(self, e=None):
        self.search.focus_set()
        self.search.select_range(0, "end")
        return "break"

    def _focus_list(self, e=None):
        if self.lb.size():
            self.lb.focus_set()
            self.lb.selection_clear(0, "end")
            self.lb.selection_set(0)
            self.lb.activate(0)
            self._update_preview()
        return "break"

    def _clear_search(self, e=None):
        self.search_var.set("")
        self.canvas.focus_set()
        return "break"

    def _pick_list_break(self):
        self.pick_from_list()
        return "break"

    # ─── отображение ───────────────────────────────────────────────────────
    def cur_fn(self):
        return self.files[self.idx] if self.files else None

    def show(self):
        if not self.files:
            self.info.set(f"В папке {we.TEST_DIR} нет картинок")
            return
        fn = self.cur_fn()
        try:
            self.cur_img = we.load_image(os.path.join(we.TEST_DIR, fn))
            self.disp_base = self.cur_img.copy()
            self.disp_base.thumbnail((CANVAS_W, CANVAS_H))
        except Exception as e:
            self.cur_img = self.disp_base = None
            self.status.set(f"Не открылось {fn}: {e}")
        self.info.set(f"{self.idx + 1} / {len(self.files)}   ·   {fn}")
        self._request(self.idx, 0)
        self._request(self.idx + 1, 1)
        self._request(self.idx + 2, 2)
        self.search_var.set("")
        self._render_image()
        self._render_candidates()
        self._update_stats()
        self.canvas.focus_set()

    def _render_image(self):
        self.canvas.delete("all")
        if self.disp_base is None:
            return
        disp = self.disp_base.copy()
        an = self.results.get(self.cur_fn()) or {}
        if an.get("size") and (an.get("poly") or an.get("box")):
            s = disp.width / an["size"][0]
            d = ImageDraw.Draw(disp)
            if an.get("poly"):
                pts = [(x * s, y * s) for x, y in an["poly"]]
                d.line(pts + [pts[0]], fill=(40, 255, 40), width=3)
            else:
                d.rectangle([v * s for v in an["box"]], outline=(255, 50, 50), width=3)
        self.tk_img = ImageTk.PhotoImage(disp)
        self.canvas.create_image(CANVAS_W // 2, CANVAS_H // 2, image=self.tk_img)

    def thumb(self, lid):
        if lid in self.thumbs:
            return self.thumbs[lid]
        ph = self.blank
        try:
            img = we.load_image(os.path.join(we.IMAGES_DIR, self.label_images[lid]))
            img.thumbnail((600, 600))              # в базе есть картинки до 6337×9506
            up = we.make_upright(img)
            W, H = up.size
            up = up.crop((0, int(H * 0.3), W, H))
            up.thumbnail(THUMB)
            canvas = Image.new("RGB", THUMB, (255, 255, 255))
            canvas.paste(up, ((THUMB[0] - up.width) // 2, (THUMB[1] - up.height) // 2))
            ph = ImageTk.PhotoImage(canvas)
        except Exception:
            pass
        self.thumbs[lid] = ph
        return ph

    def _render_candidates(self):
        fn = self.cur_fn()
        an = self.results.get(fn)
        top = self.tops.get(fn, [])
        rec = self.ann.get(fn)
        true = rec["true_label"] if rec else None
        draft = bool(rec) and str(rec.get("note", "")).startswith(DRAFT)
        if rec:
            head = "✎ ЧЕРНОВИК — подтвердите или исправьте: " if draft else "✔ Разметка: "
            if true == we.NOT_IN_DB:
                note = f" ({rec['note']})" if rec.get("note") and not draft else ""
                self.cur_ann.set(head + "НЕТ В БАЗЕ" + note)
            else:
                self.cur_ann.set(head + true)
        else:
            self.cur_ann.set("")

        if an is None:
            self.ocr_var.set("⏳ Распознаю…")
        elif "error" in an:
            self.ocr_var.set("Ошибка распознавания: " + an["error"])
        else:
            yolo = (f"YOLO: вырезано (conf {an['seg_conf']:.2f})" if an.get("box")
                    else "YOLO: бутылка не найдена — весь кадр")
            self.ocr_var.set(yolo + "   ·   OCR: " + (an.get("ocr_text") or "—"))

        for i, b in enumerate(self.btns):
            if i < len(top):
                c = top[i]
                name = c["label"] if len(c["label"]) <= 90 else c["label"][:87] + "…"
                b.config(image=self.thumb(c["label_id"]), state="normal",
                         text=f"{i + 1}. {name}\n{c['score']:.3f}  (clip {c['clip']:.3f} · ocr {c['ocr']:.2f})",
                         bg=(DRAFT_BG if draft else OK_BG) if c["label"] == true else self.btn_bg)
            else:
                b.config(image=self.blank, text=f"{i + 1}. …", state="disabled", bg=self.btn_bg)
        if not self.search_var.get().strip():
            self.refresh_list()

    def refresh_list(self):
        q = self.search_var.get().strip()
        self.lb.delete(0, "end")
        self.list_items = []
        if not q:
            top = self.tops.get(self.cur_fn(), [])
            for n, c in enumerate(top[N_CAND:], start=N_CAND + 1):
                self.list_items.append(c["label_id"])
                self.lb.insert("end", f"{n}. {c['label']}   ({c['score']:.3f})")
        else:
            words = q.lower().replace("ё", "е").split()
            lat = we.to_latin_tokens(q)
            for i, (low, lt) in enumerate(self.search_index):
                if all(w in low for w in words) or (lat and all(t in lt for t in lat)):
                    self.list_items.append(i)
                    self.lb.insert("end", self.labels[i])
                    if len(self.list_items) >= MAX_LIST:
                        break
        if self.list_items:
            self.lb.selection_set(0)
        self._update_preview()

    def _update_preview(self):
        sel = self.lb.curselection()
        img = self.thumb(self.list_items[sel[0]]) if sel and self.list_items else self.blank
        self.preview.config(image=img)

    def _update_stats(self):
        n_draft = sum(1 for f in self.files if f in self.ann and not self.is_done(f))
        vals = [r for f, r in self.ann.items() if self.is_done(f)]
        in_db = [r for r in vals if r.get("true_label") != we.NOT_IN_DB]
        ranked = []
        for r in in_db:
            v = str(r.get("rank", "")).strip()
            if v.isdigit():
                ranked.append(int(v))
        s = f"Проверено {len(vals)}/{len(self.files)} · черновиков {n_draft} · нет в базе {len(vals) - len(in_db)}"
        if ranked:
            t1 = sum(1 for r in ranked if r == 1)
            t5 = sum(1 for r in ranked if 1 <= r <= 5)
            s += f" · top-1 {t1 / len(ranked):.0%} · top-5 {t5 / len(ranked):.0%} (n={len(ranked)})"
        self.stats.set(s)

    # ─── действия ──────────────────────────────────────────────────────────
    def annotate(self, label, note=""):
        if not self.files:
            return
        fn = self.cur_fn()
        top = self.tops.get(fn, [])
        an = self.results.get(fn) or {}
        if label == we.NOT_IN_DB or not top:
            rank = ""
        else:  # место в выдаче (0 — не вошла в топ-50)
            rank = next((i + 1 for i, c in enumerate(top) if c["label"] == label), 0)
        self.ann[fn] = {
            "image_filename": fn, "true_label": label,
            "pred_top1": top[0]["label"] if top else "", "rank": rank, "note": note,
            "ocr_text": an.get("ocr_text", ""), "annotated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        try:
            we.write_annotations(self.ann)
        except PermissionError:
            messagebox.showerror("Не удалось сохранить",
                                 f"{we.ANNOT_PATH}\nоткрыт в другой программе (Excel?). "
                                 "Закройте его и выберите вариант ещё раз.")
            return
        self.status.set(f"Сохранено: {fn} → {'нет в базе' if label == we.NOT_IN_DB else label}")
        if self.idx < len(self.files) - 1:
            self.idx += 1
            self.show()
        else:
            self.show()
            messagebox.showinfo("Готово", "Это было последнее фото в папке.")

    def confirm_draft(self):
        """Пробел: черновик верный — сохранить его как проверенный."""
        r = self.ann.get(self.cur_fn())
        if r and not self.is_done(self.cur_fn()):
            self.annotate(r["true_label"])

    def pick_candidate(self, i):
        top = self.tops.get(self.cur_fn(), [])
        if i < len(top):
            self.annotate(top[i]["label"])

    def pick_from_list(self):
        if not self.list_items:
            return
        sel = self.lb.curselection()
        self.annotate(self.labels[self.list_items[sel[0] if sel else 0]])

    def mark_not_in_db(self):
        self.annotate(we.NOT_IN_DB, note=self.search_var.get().strip())

    def clear_annotation(self):
        fn = self.cur_fn()
        if fn in self.ann:
            del self.ann[fn]
            we.write_annotations(self.ann)
            self.show()

    def go(self, d):
        if self.files:
            self.idx = max(0, min(len(self.files) - 1, self.idx + d))
            self.show()

    def next_unlabeled(self):
        n = len(self.files)
        for k in range(1, n + 1):
            j = (self.idx + k) % n
            if not self.is_done(self.files[j]):
                self.idx = j
                self.show()
                return
        messagebox.showinfo("Всё размечено", "Неразмеченных фото не осталось.")


def main():
    root = tk.Tk()
    try:
        LabelTool(root)
    except FileNotFoundError as e:
        root.withdraw()
        messagebox.showerror("Нет индекса", str(e))
        return
    root.mainloop()


if __name__ == "__main__":
    main()
