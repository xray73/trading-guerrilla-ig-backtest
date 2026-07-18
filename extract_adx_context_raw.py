"""
extract_adx_context_raw.py — Estrazione dati grezzi, v3 ESTESA
(18/07/2026, Fase 0 del percorso di analisi concordato).

Obiettivo dichiarato: fare l'estensione del dataset UNA VOLTA SOLA,
comprensiva di tutto quello che serve per l'intera Fase 1 di analisi
(vedi RCA), così le query successive non richiedono mai di tornare
qui a ri-estrarre.

Nessuna soglia di "successo/fallimento" pre-calcolata, nessuna finestra
di persistenza scelta — solo indicatori/livelli grezzi, alcuni già
usati dal motore (ATR, EMA100/200, massimo/minimo breakout 20/40
barre — literalmente gli stessi parametri di V6, nessun valore
inventato), uno nuovo ma esplicitamente derivato dalla soglia già
esistente nel motore (durata del contesto ADX>20, non un nuovo
parametro arbitrario).

Storico CONTINUO 2015-2026 via Dukascopy diretto (D1/ohlc_prices ha
buchi 2017-19/2021-22, mai caricati — vedi v2). Scrive SOLO in questa
tabella diagnostica, mai su ohlc_prices.

Tabella D1 (ricreata da zero ad ogni run — tabella di scratch, non
tocca dati ufficiali): adx_diagnostic_raw

  Base (invariati da v1/v2):
    symbol, bar_index (continuo per symbol), timestamp, close, high, low

  Contesto trend (V6, già validati):
    adx (ADX 14), ema20, ema50, ema100, ema200

  Volatilità (V6, già validato):
    atr (ATR 14, metodo Wilder)

  Livelli di breakout (V6, stessi lookback usati per DAX/FTSE100):
    rolling_high_20, rolling_low_20 (massimo/minimo 20 barre precedenti,
      ESCLUSA la barra corrente — stessa convenzione di compute_indicators
      in engine.py, shift(1) prima del rolling)
    rolling_high_40, rolling_low_40 (idem, lookback 40 — usato da FTSE100)

  Derivato (durata contesto, soglia ADX>20 esistente, non nuova):
    trend_duration_adx20 (quante barre CONSECUTIVE, fino a questa
      inclusa, hanno ADX>20 — 0 se questa barra ha ADX<=20)

  NON incluso qui (calcolabile a query-time via self-join su bar_index,
  nessun bisogno di pre-calcolarlo): pendenza ADX su N barre qualunque —
  basta confrontare adx di due righe con bar_index distanziato di N,
  stessa tecnica già usata per la persistenza.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

import dukascopy_python
from dukascopy_python.instruments import INSTRUMENT_IDX_EUROPE_E_DAAX, INSTRUMENT_IDX_EUROPE_E_FUTSEE_100
from datetime import datetime, timezone

import engine as eng

DATABASE_ID = "b9fbd4d6-7837-4d86-9c0f-ca60c0cf69e3"
API_BASE = "https://api.cloudflare.com/client/v4/accounts"
INSERT_BATCH = 300  # ridotto rispetto a v2 per via delle colonne aggiuntive (dimensione riga più grande)

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
    """Quante barre consecutive (fino a questa inclusa) hanno adx>threshold.
    0 se la barra corrente non è in contesto di trend."""
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

    print("=== Estrazione dati grezzi ESTESA (Fase 0) — storico continuo 2015-2026 ===\n")

    recreate_table(account_id, token)
    print("Tabella adx_diagnostic_raw ricreata con schema esteso.\n")

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
            print(f"  [{symbol}] ATTENZIONE: mancano {sorted(missing)} anche stavolta.")

        # --- indicatori base (già in v1/v2) ---
        df["adx"] = eng.adx_wilder(df, 14)
        df["ema20"] = eng.ema(df["close"], 20)
        df["ema50"] = eng.ema(df["close"], 50)

        # --- nuovi in v3 ---
        df["ema100"] = eng.ema(df["close"], 100)
        df["ema200"] = eng.ema(df["close"], 200)
        df["atr"] = eng.atr_wilder(df, 14)
        df["rolling_high_20"] = df["high"].shift(1).rolling(20).max()
        df["rolling_low_20"] = df["low"].shift(1).rolling(20).min()
        df["rolling_high_40"] = df["high"].shift(1).rolling(40).max()
        df["rolling_low_40"] = df["low"].shift(1).rolling(40).min()
        df["trend_duration_adx20"] = compute_trend_duration(df["adx"], 20.0)

        required_cols = ["adx", "ema20", "ema50", "ema100", "ema200", "atr",
                          "rolling_high_20", "rolling_low_20", "rolling_high_40", "rolling_low_40"]
        df = df.dropna(subset=required_cols).reset_index(drop=True)
        df["bar_index"] = range(len(df))

        n = len(df)
        print(f"  [{symbol}] {n} barre valide (dopo warmup di tutti gli indicatori, "
              f"il più esigente è EMA200/rolling_40), {df['timestamp'].min()} -> {df['timestamp'].max()}")

        cols = ["bar_index", "timestamp", "close", "high", "low", "adx", "ema20", "ema50",
                "ema100", "ema200", "atr", "rolling_high_20", "rolling_low_20",
                "rolling_high_40", "rolling_low_40", "trend_duration_adx20"]
        records = df[cols].to_dict("records")

        for i in range(0, len(records), INSERT_BATCH):
            batch = records[i:i + INSERT_BATCH]
            values_sql = ",".join(
                f"('{symbol}',{r['bar_index']},'{r['timestamp'].isoformat()}',"
                f"{r['close']},{r['high']},{r['low']},{r['adx']},{r['ema20']},{r['ema50']},"
                f"{r['ema100']},{r['ema200']},{r['atr']},"
                f"{r['rolling_high_20']},{r['rolling_low_20']},"
                f"{r['rolling_high_40']},{r['rolling_low_40']},"
                f"{int(r['trend_duration_adx20'])})"
                for r in batch
            )
            d1_query(
                "INSERT INTO adx_diagnostic_raw "
                "(symbol, bar_index, timestamp, close, high, low, adx, ema20, ema50, "
                "ema100, ema200, atr, rolling_high_20, rolling_low_20, "
                "rolling_high_40, rolling_low_40, trend_duration_adx20) "
                f"VALUES {values_sql}",
                account_id, token,
            )
            if (i // INSERT_BATCH) % 30 == 0:
                print(f"    ...{i + len(batch)}/{n} righe inserite")

        total_rows += n
        print(f"  [{symbol}] completato: {n} righe inserite.\n")

    print(f"=== Completato. {total_rows} righe totali in adx_diagnostic_raw (schema esteso). ===")
    print("Fase 0 conclusa. Tutte le analisi di Fase 1 sono ora query dirette, nessuna nuova estrazione necessaria.")


if __name__ == "__main__":
    main()
