"""
persistence_check_generic.py — Versione generica di persistence_check.py
/ persistence_check_smi.py: accetta simboli target e simbolo di
riferimento da riga di comando, stessa metodologia esatta per tutti
(compute_persistence invariata — nessuna discrepanza tra confronti).

Uso:
  python persistence_check_generic.py SMI DAX
  python persistence_check_generic.py IBEX35,ITALY40 DAX

Primo argomento: simboli target (uno o più, separati da virgola)
Secondo argomento: simbolo di riferimento (default DAX se omesso)

Nessuna whitelist necessaria (a differenza del caricamento dati): legge
da D1 via SELECT, se il simbolo non esiste stampa solo "nessun dato,
salto" — non c'è rischio di caricare/confondere dati di uno strumento
sbagliato come nel caso Dukascopy.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
import requests

import engine as eng

DATABASE_ID = "b9fbd4d6-7837-4d86-9c0f-ca60c0cf69e3"
API_BASE = "https://api.cloudflare.com/client/v4/accounts"
CHUNK_SIZE = 5000

FORWARD_BARS = 20

PERIODS = {
    "2015-2016": ("2015-01-01", "2016-12-31"),
    "2020-covid": ("2020-01-01", "2020-12-31"),
    "2023": ("2023-01-01", "2023-12-31"),
    "2024-2025": ("2024-01-01", "2025-12-31"),
    "2026-ytd": ("2026-01-01", "2026-07-14"),
}
WARMUP_DAYS = 30


def d1_query(sql: str, account_id: str, token: str) -> list[dict]:
    url = f"{API_BASE}/{account_id}/d1/database/{DATABASE_ID}/query"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    resp = requests.post(url, headers=headers, json={"sql": sql}, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if not data.get("success"):
        raise RuntimeError(f"D1 query fallita: {data.get('errors')}")
    return data["result"][0]["results"]


def fetch_symbol_range(symbol: str, start: str, end: str, account_id: str, token: str) -> pd.DataFrame:
    rows = []
    offset = 0
    while True:
        sql = (
            f"SELECT timestamp, open, high, low, close FROM ohlc_prices "
            f"WHERE symbol='{symbol}' AND timestamp >= '{start}' AND timestamp < '{end}' "
            f"ORDER BY timestamp LIMIT {CHUNK_SIZE} OFFSET {offset}"
        )
        batch = d1_query(sql, account_id, token)
        if not batch:
            break
        rows.extend(batch)
        offset += CHUNK_SIZE
        if len(batch) < CHUNK_SIZE:
            break
        time.sleep(0.15)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def compute_persistence(df: pd.DataFrame, forward_bars: int = FORWARD_BARS) -> dict:
    out = df.copy()
    out["ema20"] = eng.ema(out["close"], 20)
    out["ema50"] = eng.ema(out["close"], 50)
    out["adx"] = eng.adx_wilder(out, 14)

    direction_long = out["ema20"] > out["ema50"]
    direction_short = out["ema20"] < out["ema50"]
    trend_context = out["adx"] > 20

    out["forward_close"] = out["close"].shift(-forward_bars)
    out["forward_return"] = (out["forward_close"] - out["close"]) / out["close"]

    long_ctx = out[trend_context & direction_long & out["forward_return"].notna()]
    short_ctx = out[trend_context & direction_short & out["forward_return"].notna()]

    long_persist = (long_ctx["forward_return"] > 0).mean() if len(long_ctx) else np.nan
    short_persist = (short_ctx["forward_return"] < 0).mean() if len(short_ctx) else np.nan

    n_total_ctx = len(long_ctx) + len(short_ctx)
    if n_total_ctx > 0:
        combined_persist = (
            (long_ctx["forward_return"] > 0).sum() + (short_ctx["forward_return"] < 0).sum()
        ) / n_total_ctx
    else:
        combined_persist = np.nan

    return {
        "n_bars_totali": len(out),
        "n_contesto_trend": n_total_ctx,
        "pct_barre_in_trend": n_total_ctx / len(out) if len(out) else np.nan,
        "persistenza_long": long_persist,
        "persistenza_short": short_persist,
        "persistenza_combinata": combined_persist,
    }


def main():
    if len(sys.argv) < 2:
        print("Uso: python persistence_check_generic.py SIMBOLO1,SIMBOLO2 [RIFERIMENTO]")
        sys.exit(1)

    targets = [s.strip().upper() for s in sys.argv[1].split(",") if s.strip()]
    reference = sys.argv[2].strip().upper() if len(sys.argv) > 2 else "DAX"

    token = os.environ.get("CLOUDFLARE_API_TOKEN")
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    if not token or not account_id:
        print("ERRORE: secrets mancanti.", file=sys.stderr)
        sys.exit(1)

    os.makedirs("results", exist_ok=True)
    rows = []
    symbols_to_check = targets + [reference]

    for symbol in symbols_to_check:
        print(f"\n=== {symbol} ===")
        for period_label, (start_str, end_str) in PERIODS.items():
            start = pd.Timestamp(start_str, tz="UTC") - pd.Timedelta(days=WARMUP_DAYS)
            end = pd.Timestamp(end_str, tz="UTC") + pd.Timedelta(days=1)
            df = fetch_symbol_range(symbol, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"),
                                     account_id, token)
            if df.empty:
                print(f"  {period_label}: nessun dato, salto.")
                continue

            result = compute_persistence(df)
            result["symbol"] = symbol
            result["period"] = period_label
            rows.append(result)

            print(f"  {period_label}: {result['n_contesto_trend']} barre in trend "
                  f"({result['pct_barre_in_trend']*100:.1f}% del totale) — "
                  f"persistenza combinata: {result['persistenza_combinata']*100:.1f}%")

    if not rows:
        print("\nNessun dato trovato per nessuno dei simboli richiesti. Interrompo.")
        sys.exit(1)

    df_results = pd.DataFrame(rows)
    out_name = f"results/persistence_check_{'_'.join(t.lower() for t in targets)}.csv"
    df_results.to_csv(out_name, index=False)

    print("\n" + "=" * 70)
    print("RIEPILOGO — persistenza combinata per strumento (media sui periodi)")
    print("=" * 70)
    summary = df_results.groupby("symbol")["persistenza_combinata"].agg(["mean", "std", "count"])
    print(summary.to_string())

    print(f"\nCompletato. File: {out_name}")


if __name__ == "__main__":
    main()
