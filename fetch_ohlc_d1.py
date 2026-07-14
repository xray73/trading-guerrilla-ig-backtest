"""
fetch_ohlc_d1.py — Scarica OHLC DAX+FTSE100 da Cloudflare D1 via REST API
diretta. Nessun wrangler/Node necessario: GitHub Actions ha accesso di rete
pieno (a differenza del sandbox Claude Code on the web, che lo blocca).

Richiede due secrets del repository GitHub (Settings > Secrets and
variables > Actions):
  CLOUDFLARE_API_TOKEN   (permesso "D1 read" o superiore)
  CLOUDFLARE_ACCOUNT_ID

Output: DAX_full.csv, FTSE100_full.csv nella working directory del job.

Paginazione: la API D1 potrebbe limitare la dimensione di una singola
risposta, quindi si scarica a blocchi (LIMIT/OFFSET) invece che in
un'unica query, con verifica finale del conteggio righe contro un
SELECT COUNT(*) preliminare.
"""

from __future__ import annotations

import os
import sys
import time

import pandas as pd
import requests

DATABASE_ID = "b9fbd4d6-7837-4d86-9c0f-ca60c0cf69e3"
CHUNK_SIZE = 5000
SYMBOLS = ["DAX", "FTSE100"]
API_BASE = "https://api.cloudflare.com/client/v4/accounts"


def d1_query(sql: str, account_id: str, token: str) -> list[dict]:
    url = f"{API_BASE}/{account_id}/d1/database/{DATABASE_ID}/query"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    resp = requests.post(url, headers=headers, json={"sql": sql}, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if not data.get("success"):
        raise RuntimeError(f"D1 query fallita: {data.get('errors')}")
    return data["result"][0]["results"]


def count_rows(symbol: str, account_id: str, token: str) -> int:
    sql = f"SELECT COUNT(*) as n FROM ohlc_prices WHERE symbol='{symbol}'"
    result = d1_query(sql, account_id, token)
    return int(result[0]["n"])


def fetch_symbol(symbol: str, account_id: str, token: str) -> pd.DataFrame:
    expected = count_rows(symbol, account_id, token)
    print(f"  {symbol}: {expected} righe attese (da COUNT(*))")

    rows = []
    offset = 0
    while True:
        sql = (
            f"SELECT timestamp, open, high, low, close, volume FROM ohlc_prices "
            f"WHERE symbol='{symbol}' ORDER BY timestamp LIMIT {CHUNK_SIZE} OFFSET {offset}"
        )
        batch = d1_query(sql, account_id, token)
        if not batch:
            break
        rows.extend(batch)
        offset += CHUNK_SIZE
        print(f"  {symbol}: {len(rows)}/{expected} righe scaricate...")
        if len(batch) < CHUNK_SIZE:
            break
        time.sleep(0.2)  # rispetto rate limit API Cloudflare

    df = pd.DataFrame(rows)
    if len(df) != expected:
        print(f"  ATTENZIONE {symbol}: attese {expected} righe, scaricate {len(df)} "
              f"— controllare manualmente prima di fidarsi del risultato della grid search.",
              file=sys.stderr)
    return df


def main():
    token = os.environ.get("CLOUDFLARE_API_TOKEN")
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    if not token or not account_id:
        print("ERRORE: CLOUDFLARE_API_TOKEN o CLOUDFLARE_ACCOUNT_ID mancanti "
              "(devono essere configurati come secrets del repository GitHub).",
              file=sys.stderr)
        sys.exit(1)

    for symbol in SYMBOLS:
        print(f"Scaricando {symbol}...")
        df = fetch_symbol(symbol, account_id, token)
        if df.empty:
            print(f"ERRORE: nessuna riga trovata per {symbol}. Verifica il "
                  f"database_id o il nome del simbolo.", file=sys.stderr)
            sys.exit(1)
        out_path = f"{symbol}_full.csv"
        df.to_csv(out_path, index=False)
        print(f"  {symbol}: {len(df)} righe -> {out_path}")

    print("Download completato.")


if __name__ == "__main__":
    main()
