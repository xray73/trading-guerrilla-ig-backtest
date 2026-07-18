"""
extract_adx_context_raw.py — Estrazione dati grezzi, v4 (18/07/2026).

AGGIUNTA v4: RSI(14) e Bollinger Bands (20 periodi, 2 dev std) — gli
indicatori REALI usati dalla mean-reversion (mean_reversion_signals.py),
necessari per verificare se la scoperta "FTSE100 reverte meglio di DAX
a bassissimo ADX" (trovata usando EMA50 come proxy grezzo della media)
regge anche con l'indicatore vero (RSI) che il sistema userebbe
davvero in produzione.

Funzioni RSI/Bollinger IMPORTATE da mean_reversion_signals.py, non
reimplementate — stessa metodologia "nessun parametro nuovo inventato"
già seguita per ADX/EMA/ATR/breakout.

Storico continuo 2015-2026 via Dukascopy diretto. Scrive SOLO nella
tabella diagnostica adx_diagnostic_raw (ricreata da zero, scratch
table, non tocca ohlc_prices).

Schema completo v4 (base + v2 + v3 + v4):
  symbol, bar_index, timestamp, close, high, low,
  adx, ema20, ema50, ema100, ema200, atr,
  rolling_high_20, rolling_low_20, rolling_high_40, rolling_low_40,
  trend_duration_adx20,
  rsi14, bb_upper, bb_mid, bb_lower   [NUOVI v4]
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

import dukascopy_python
from dukascopy_python.instruments import INSTRUMENT_IDX_EUROPE_E_DAAX, INSTRUMENT_IDX_EUROPE_E_FUTSEE_100
from datetime import datetime, timezone

import engine as eng
from mean_reversion_signals import _rsi_wilder, _bollinger_bands, RSI_PERIOD, BB_PERIOD, BB_STD

DATABASE_ID = "b9fbd4d6-7837-4d86-9c0f-ca60c0cf69e3"
API_BASE = "https://api.cloudflare.com/client/v4/accounts"
INSERT_BATCH = 250  # ridotto ulteriormente per via delle 4 colonne aggiuntive

SYMBOLS = {"DAX": INSTRUMENT_IDX_EUROPE_E_DAAX, "FTSE100": INSTRUMENT_IDX_EUROPE_E_FUTSEE_100}
FETCH_START = datetime(2015, 1, 1, tzinfo=timezone.utc)
FETCH_END = datetime(2026, 7, 19, tzinfo=timezone.utc)

NEVER_TESTED_YEARS = {2017, 2018, 2019, 2021, 2022}


def d1_query(sql: str, account_id: str, token: str) -> list[dict]:
    import requests
    url = f"{API_BASE}/{account_id}/d1/database/{DATABASE_ID}/query"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    resp = requests.post(url, headers=headers, json={"sql": sql}, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if not data.get("success"):
        raise RuntimeError(f"D1 query fallita: {data.get('errors')}")
    return data["result"][0]["results"]


def fetch_full(symbol_const) -> pd.DataFrame:
    df = dukascopy_python.fetch(
        symbol_const, dukascopy_python.INTERVAL_MIN_30, dukascopy_python.OFFER_SIDE_BID,
        FETCH_START, FETCH_END,
    ).reset_index()
    ts_col = df.columns[0]
    df = df.rename(columns={ts_col: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def compute_trend_duration(adx: pd.Series, threshold: float = 20.0) -> pd.Series:
    above = (adx > threshold)
    block_id = (above != above.shift()).cumsum()
    streak = above.groupby(block_id).cumcount() + 1
    return np.where(above, streak, 0)


def recreate_table(account_id: str, token: str):
    d1_query("DROP TABLE IF EXISTS adx_diagnostic_raw", account_id, token)
    d1_query(
        "CREATE TABLE adx_diagnostic_raw ("
        "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "  symbol TEXT NOT NULL,"
        "  bar_index INTEGER NOT NULL,"
        "  timestamp TEXT NOT NULL,"
        "  close REAL NOT NULL,"
        "  high REAL NOT NULL,"
        "  low REAL NOT NULL,"
        "  adx REAL,"
        "  ema20 REAL,"
        "  ema50 REAL,"
        "  ema100 REAL,"
        "  ema200 REAL,"
        "  atr REAL,"
        "  rolling_high_20 REAL,"
        "  rolling_low_20 REAL,"
        "  rolling_high_40 REAL,"
        "  rolling_low_40 REAL,"
        "  trend_duration_adx20 INTEGER,"
        "  rsi14 REAL,"
        "  bb_upper REAL,"
        "  bb_mid REAL,"
        "  bb_lower REAL,"
        "  UNIQUE(symbol, bar_index)"
        ")",
        account_id, token,
    )


def main():
    token = os.environ.get("CLOUDFLARE_API_TOKEN")
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    if not token or not account_id:
        print("ERRORE: CLOUDFLARE_API_TOKEN o CLOUDFLARE_ACCOUNT_ID mancanti.")
        return

    print("=== Estrazione dati grezzi v4 (+ RSI/Bollinger) — storico continuo 2015-2026 ===\n")
    print(f"RSI period={RSI_PERIOD}, Bollinger period={BB_PERIOD} std={BB_STD} "
          f"(importati da mean_reversion_signals.py, nessun parametro nuovo)\n")

    recreate_table(account_id, token)
    print("Tabella adx_diagnostic_raw ricreata con schema v4.\n")

    total_rows = 0
    for symbol, const in SYMBOLS.items():
        print(f"Scarico {symbol} da Dukascopy (2015-2026 continuo)...")
        df = fetch_full(const)
        if df.empty:
            print(f"  [{symbol}] ERRORE: nessun dato ritornato.")
            continue

        years_present = set(df["timestamp"].dt.year.unique())
        missing = NEVER_TESTED_YEARS - years_present
        if missing:
            print(f"  [{symbol}] ATTENZIONE: mancano {sorted(missing)}.")

        df["adx"] = eng.adx_wilder(df, 14)
        df["ema20"] = eng.ema(df["close"], 20)
        df["ema50"] = eng.ema(df["close"], 50)
        df["ema100"] = eng.ema(df["close"], 100)
        df["ema200"] = eng.ema(df["close"], 200)
        df["atr"] = eng.atr_wilder(df, 14)
        df["rolling_high_20"] = df["high"].shift(1).rolling(20).max()
        df["rolling_low_20"] = df["low"].shift(1).rolling(20).min()
        df["rolling_high_40"] = df["high"].shift(1).rolling(40).max()
        df["rolling_low_40"] = df["low"].shift(1).rolling(40).min()
        df["trend_duration_adx20"] = compute_trend_duration(df["adx"], 20.0)

        # --- nuovi v4, funzioni riusate da mean_reversion_signals.py ---
        df["rsi14"] = _rsi_wilder(df, RSI_PERIOD)
        bb_upper, bb_mid, bb_lower = _bollinger_bands(df, BB_PERIOD, BB_STD)
        df["bb_upper"], df["bb_mid"], df["bb_lower"] = bb_upper, bb_mid, bb_lower

        required_cols = ["adx", "ema20", "ema50", "ema100", "ema200", "atr",
                          "rolling_high_20", "rolling_low_20", "rolling_high_40", "rolling_low_40",
                          "rsi14", "bb_upper", "bb_mid", "bb_lower"]
        df = df.dropna(subset=required_cols).reset_index(drop=True)
        df["bar_index"] = range(len(df))

        n = len(df)
        print(f"  [{symbol}] {n} barre valide, {df['timestamp'].min()} -> {df['timestamp'].max()}")

        cols = ["bar_index", "timestamp", "close", "high", "low", "adx", "ema20", "ema50",
                "ema100", "ema200", "atr", "rolling_high_20", "rolling_low_20",
                "rolling_high_40", "rolling_low_40", "trend_duration_adx20",
                "rsi14", "bb_upper", "bb_mid", "bb_lower"]
        records = df[cols].to_dict("records")

        for i in range(0, len(records), INSERT_BATCH):
            batch = records[i:i + INSERT_BATCH]
            values_sql = ",".join(
                f"('{symbol}',{r['bar_index']},'{r['timestamp'].isoformat()}',"
                f"{r['close']},{r['high']},{r['low']},{r['adx']},{r['ema20']},{r['ema50']},"
                f"{r['ema100']},{r['ema200']},{r['atr']},"
                f"{r['rolling_high_20']},{r['rolling_low_20']},"
                f"{r['rolling_high_40']},{r['rolling_low_40']},"
                f"{int(r['trend_duration_adx20'])},"
                f"{r['rsi14']},{r['bb_upper']},{r['bb_mid']},{r['bb_lower']})"
                for r in batch
            )
            d1_query(
                "INSERT INTO adx_diagnostic_raw "
                "(symbol, bar_index, timestamp, close, high, low, adx, ema20, ema50, "
                "ema100, ema200, atr, rolling_high_20, rolling_low_20, "
                "rolling_high_40, rolling_low_40, trend_duration_adx20, "
                "rsi14, bb_upper, bb_mid, bb_lower) "
                f"VALUES {values_sql}",
                account_id, token,
            )
            if (i // INSERT_BATCH) % 40 == 0:
                print(f"    ...{i + len(batch)}/{n} righe inserite")

        total_rows += n
        print(f"  [{symbol}] completato: {n} righe inserite.\n")

    print(f"=== Completato. {total_rows} righe totali in adx_diagnostic_raw (schema v4). ===")


if __name__ == "__main__":
    main()
