#!/usr/bin/env python3
import os
import sys
import psycopg2
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
from tqdm import tqdm

# ============================================================
# CONFIG
# ============================================================

# ---- Model ----
MODEL_NAME = "yiyanghkust/finbert-tone"

# ---- Chunking / batching ----
CHUNK_SIZE = 400          # characters per chunk (same logic as your original)
MAX_TOKENS = 512          # tokenizer max_length
BATCH_SIZE = 64           # H100 can often handle 64-256 depending on sequence lengths

# ---- Output files ----
DAILY_OUT_CSV = "SP500_sentiment_daily.csv"
#ARTICLE_DEBUG_CSV = "SP500_sentiment_articles_debug.csv"  # set to None to disable

# ---- DB credentials (use env vars if possible) ----
DB_HOST = os.getenv("MRN_HOST", "localhost")
DB_PORT = os.getenv("MRN_PORT", "port")
DB_NAME = os.getenv("MRN_DB", "dbname")
DB_USER = os.getenv("MRN_USER", "username")
DB_PASS = os.getenv("MRN_PASS", "password")  

# ============================================================
# SQL QUERY
# ============================================================
QUERY = """
WITH base AS (
  SELECT
      d.item_id,
      d.first_created,
      regexp_replace(trim(d.headline), '\s+', ' ', 'g') AS norm_headline,
      d.headline,
      d.body
  FROM item_data d
  WHERE d.first_created >= date '2010-01-01'
    AND d.first_created <  date '2026-01-01'
    AND d.item_language = 'en'
),
filtered AS (
  SELECT *
  FROM base
  WHERE (
        /* ---- S&P 500 variants ---- */
        headline ~* '\ms\s*&\s*p\s*500\M' OR body ~* '\ms\s*&\s*p\s*500\M'
     OR headline ~* '\msp\s*500\M'        OR body ~* '\msp\s*500\M'
     OR headline ~* '\mstandard\s+(and|&)\s+poor''s\s+500\M'
        OR body  ~* '\mstandard\s+(and|&)\s+poor''s\s+500\M'
     OR body LIKE '%<.SPX>%'

        /* ---- SPY ETF ---- */
     OR headline ~* '\mspy\M' OR body ~* '\mspy\M'
     OR headline ~* '\mspdr\s+s\s*&\s*p\s+500\s+etf\M'
        OR body  ~* '\mspdr\s+s\s*&\s*p\s+500\s+etf\M'
     OR body LIKE '%<SPY>%'
  )
),
dedup AS (
  -- choose the most recent update per headline
  SELECT DISTINCT ON (norm_headline)
      item_id,
      first_created,
      norm_headline,
      body
  FROM filtered
  ORDER BY norm_headline, first_created DESC
)
SELECT
    item_id,
    first_created,
    norm_headline AS headline,
    body
FROM dedup
ORDER BY first_created;
"""

# ============================================================
# GPU / MODEL SETUP
# ============================================================

def setup_torch():
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

def load_finbert(device):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()
    return tokenizer, model

# ============================================================
# FINBERT INFERENCE (BATCHED)
# ============================================================

LABELS = ["positive", "negative", "neutral"]

def chunk_text(text: str, chunk_size: int):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def finbert_on_body_batched(
    text,
    tokenizer,
    model,
    device,
    chunk_size=CHUNK_SIZE,
    batch_size=BATCH_SIZE,
    max_tokens=MAX_TOKENS
):
    """
    Returns:
      label (str),
      avg_probs ([pos, neg, neu]),
      chunks_total (int),
      chunks_valid (int),
      incomplete_chunks (int),
      failed_chunks (int)
    """
    # --- sanitize ---
    if text is None or not isinstance(text, str):
        return "neutral", [0.0, 0.0, 1.0], 0, 0, 0, 0

    text = text.strip()
    if not text:
        return "neutral", [0.0, 0.0, 1.0], 0, 0, 0, 0

    chunks = chunk_text(text, chunk_size)
    chunks_total = len(chunks)
    incomplete_chunks = sum(1 for c in chunks if 0 < len(c) < chunk_size)

    all_probs = []
    failed_chunks = 0

    # Run batches of chunks
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        try:
            enc = tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=max_tokens,
                padding=True
            )
            enc = {k: v.to(device) for k, v in enc.items()}

            with torch.no_grad():
                # H100-friendly mixed precision
                with torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=torch.bfloat16):
                    logits = model(**enc).logits
                    probs = softmax(logits, dim=1).detach().cpu().tolist()

            # probs is list of [pos, neg, neu]
            # guard just in case
            for p in probs:
                if isinstance(p, list) and len(p) == 3:
                    all_probs.append(p)
                else:
                    failed_chunks += 1

        except Exception:
            # If a batch fails, count all its chunks as failed
            failed_chunks += len(batch)

    if not all_probs:
        return "neutral", [0.0, 0.0, 1.0], chunks_total, 0, incomplete_chunks, failed_chunks

    avg = torch.tensor(all_probs).mean(dim=0).tolist()
    label = LABELS[int(torch.tensor(avg).argmax().item())]
    chunks_valid = len(all_probs)
    return label, avg, chunks_total, chunks_valid, incomplete_chunks, failed_chunks

# ============================================================
# MAIN
# ============================================================

def main():
    setup_torch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer, model = load_finbert(device)

    # --- DB connect ---
    conn = None
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASS
        )
        print("Connected to database!")

        df = pd.read_sql(QUERY, conn)
        print("Total articles (raw):", len(df))

        # Keep needed fields
        df = df[["item_id", "first_created", "body"]].dropna()
        df["first_created"] = pd.to_datetime(df["first_created"], utc=True, errors="coerce")
        df = df.dropna(subset=["first_created"])
        df["date"] = df["first_created"].dt.date

        print("Total articles (after cleaning):", len(df))
        if len(df) == 0:
            print("No data returned after cleaning. Exiting.")
            return 0

        # --- per-article inference ---
        sentiments = []
        prob_scores = []
        chunks_total_list = []
        chunks_valid_list = []
        incomplete_chunks_list = []
        failed_chunks_list = []

        for body in tqdm(df["body"], desc="FinBERT (GPU batched chunks)"):
            label, probs, n_total, n_valid, n_incomplete, n_failed = finbert_on_body_batched(
                body,
                tokenizer=tokenizer,
                model=model,
                device=device,
                chunk_size=CHUNK_SIZE,
                batch_size=BATCH_SIZE,
                max_tokens=MAX_TOKENS
            )
            sentiments.append(label)
            prob_scores.append(probs)
            chunks_total_list.append(n_total)
            chunks_valid_list.append(n_valid)
            incomplete_chunks_list.append(n_incomplete)
            failed_chunks_list.append(n_failed)

        df["sentiment"] = sentiments
        df["probs"] = prob_scores
        df["sent_score"] = df["probs"].apply(lambda p: float(p[0]) - float(p[1]))

        df["chunks_total"] = chunks_total_list
        df["chunks_valid"] = chunks_valid_list
        df["incomplete_chunks"] = incomplete_chunks_list
        df["failed_chunks"] = failed_chunks_list

        # --- daily aggregation ---
        daily = (
            df.groupby("date")
              .agg(
                  daily_sentiment=("sent_score", "mean"),
                  articles_per_day=("item_id", "count"),
                  chunks_per_day=("chunks_valid", "sum"),
                  incomplete_chunks_per_day=("incomplete_chunks", "sum"),
                  failed_chunks_per_day=("failed_chunks", "sum"),
                  chunks_total_per_day=("chunks_total", "sum"),
              )
              .reset_index()
              .sort_values("date")
        )

        print(daily.head(10))
        print("Days:", len(daily))

        daily.to_csv(DAILY_OUT_CSV, index=False)
        print(f"Saved daily CSV -> {DAILY_OUT_CSV}")

        if ARTICLE_DEBUG_CSV:
            df.to_csv(ARTICLE_DEBUG_CSV, index=False)
            print(f"Saved per-article debug CSV -> {ARTICLE_DEBUG_CSV}")

        return 0

    except Exception as e:
        print("Error:", repr(e))
        return 1

    finally:
        if conn is not None:
            conn.close()
            print("Database connection closed.")

if __name__ == "__main__":
    raise SystemExit(main())
