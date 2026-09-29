#!/usr/bin/env bash
# Прогон как у кейсодержателя: фото по одному → POST /predict → {"slug":...} + время ответа.
# Использование: bash scripts/eval_api.sh [папка_с_фото] [url]
DIR="${1:-data/images_test}"
URL="${2:-http://127.0.0.1:8000/predict}"
OUT="eval_results.csv"
echo "file,slug,time_s" > "$OUT"
for f in "$DIR"/*; do
  [ -f "$f" ] || continue
  res=$(curl -s -w ' %{time_total}' -F "file=@${f}" "$URL")
  t=${res##* }; body=${res% *}
  slug=$(echo "$body" | python -c "import sys,json;print(json.load(sys.stdin).get('slug',''))" 2>/dev/null)
  echo "$(basename "$f"),$slug,$t" | tee -a "$OUT"
done
python - "$OUT" <<'PY'
import csv,sys,statistics as st
r=list(csv.DictReader(open(sys.argv[1])));t=[float(x["time_s"]) for x in r if x["time_s"]]
print(f"\nфото: {len(r)}  среднее время: {st.mean(t):.2f}s  медиана: {st.median(t):.2f}s  макс: {max(t):.2f}s" if t else "нет ответов")
PY
