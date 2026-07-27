#!/bin/sh
set -e

python - <<'PY'
import os
import sys
import time

import ollama

model = os.environ.get("MODEL_ADI", "gemma3:4b-it-qat")

print("[Docker] Ollama sunucusuna baglaniliyor...", flush=True)
for _ in range(60):
    try:
        mevcut = ollama.list()
        break
    except Exception:
        time.sleep(2)
else:
    sys.exit("[Docker] Ollama sunucusuna ulasilamadi. 'docker compose logs ollama' ile kontrol edin.")

adlar = [m.get("model") or m.get("name") for m in mevcut.get("models", [])]

if model not in adlar:
    print(f"[Docker] '{model}' indiriliyor (~4 GB). Bu SADECE ilk calistirmada olur.", flush=True)
    for parca in ollama.pull(model, stream=True):
        toplam = parca.get("total")
        tamam = parca.get("completed")
        if toplam and tamam:
            print(f"\r  {parca.get('status', '')}: %{tamam * 100 // toplam}   ", end="", flush=True)
    print("\n[Docker] Model hazir.", flush=True)
else:
    print(f"[Docker] Model zaten mevcut: {model}", flush=True)
PY

exec "$@"
