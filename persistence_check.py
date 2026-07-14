"""
persistence_check.py — Passo 3 dell'imbuto di selezione asset (RCA
Addendum sez.37): "persistenza direzionale grezza", economico rispetto
a un walk-forward completo — non apre trade, non simula il motore,
misura solo se il prezzo continua nella direzione indicata dal
contesto di trend (ADX>20 + EMA20/50), stessa definizione già usata
dal segnale Variante 6, su ITALY40 confrontato con DAX come riferimento.

Metrica: per ogni barra con ADX(14)>20 e EMA20 vs EMA50 che indicano
una direzione, guarda il ritorno del prezzo dopo K barre (default 20,
stessa scala del lookback breakout). "Persistenza" = frazione di quei
casi in cui il prezzo si è mosso a favore della direzione indicata
(non necessariamente vinto un trade — solo se il prezzo è andato nel
verso giusto).

Nessun filtro costruito, nessuna decisione presa qui — solo dato per
decidere se vale la pena una validazione completa (Variante 6 +
walk-forward) su ITALY40.
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

FORWARD_BARS = 20  # stessa scala del lookback breakout DAX

PERIODS = {
    "2020-covid": ("2020-01-01", "2020-12-31"),
    "2023": ("2023-01-01", "2023-12-31"),
    "2024-2025": ("2024-01-01", "2025-12-31"),
    "2026-ytd": ("2026-01-01", "2026-07-14"),
}
WARMUP_DAYS = 30  # basta per ADX/EMA20/50, non serve il warmup EMA100/200 qui


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
    """df deve avere colonne timestamp/open/high/low/close, gia' nel
    periodo desiderato (con warmup incluso)."""
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
    token = os.environ.get("CLOUDFLARE_API_TOKEN")
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    if not token or not account_id:
        print("ERRORE: secrets mancanti.", file=sys.stderr)
        sys.exit(1)

    os.makedirs("results", exist_ok=True)
    rows = []

    for symbol in ["ITALY40", "DAX"]:
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

    df_results = pd.DataFrame(rows)
    df_results.to_csv("results/persistence_check.csv", index=False)

    print("\n" + "=" * 70)
    print("RIEPILOGO — persistenza combinata per strumento (media sui periodi)")
    print("=" * 70)
    summary = df_results.groupby("symbol")["persistenza_combinata"].agg(["mean", "std", "count"])
    print(summary.to_string())

    print("\nCompletato. File in results/.")


if __name__ == "__main__":
    main()
