"""
extract_adx_context_raw.py — Estrazione dati grezzi (18/07/2026,
v2 continua), NON un'analisi. Nessuna soglia ADX pre-applicata,
nessuna finestra di persistenza scelta, nessun successo/fallimento
pre-calcolato — solo gli ingredienti già validati (ADX(14), EMA20/50,
stessi parametri del motore, nessun parametro nuovo) per DAX e
FTSE100, con un indice di barra CONTINUO che permette di guardare
"N barre più avanti" per QUALUNQUE N in un secondo momento, via query
dirette — senza dover decidere ora quale finestra o soglia usare.

AGGIORNAMENTO v2 (18/07/2026): storico CONTINUO 2015-2026, non più
solo le finestre dei 5 periodi ufficiali. Scoperto che D1
(ohlc_prices) NON ha 2017/2018/2019/2021/2022 — mai caricati, il
progetto ha sempre usato solo le finestre dei 5 periodi ufficiali.
Questo script quindi scarica da DUKASCOPY DIRETTO (non da D1) per
avere continuità reale, e salva SOLO in questa nuova tabella
diagnostica — NON scrive su ohlc_prices (quella resta il dataset
ufficiale dietro tutti i backtest validati, ampliarla è una decisione
separata, non necessaria per questa diagnostica).

Se Dukascopy non ha dati per uno degli anni mai testati prima
(2017-2019, 2021-2022), lo script lo segnala esplicitamente nel log
invece di fallire silenziosamente (stesso standard già applicato al
buco scoperto per ITALY40).

Tabella D1 creata (se non esiste): adx_diagnostic_raw
  symbol, bar_index (sequenziale CONTINUO per symbol, 0-based,
  ordinato per timestamp su tutto lo storico 2015-2026), timestamp,
  close, high, low, adx, ema20, ema50.

bar_index permette self-join in SQL per "N barre dopo" con un JOIN su
bar_index = bar_index + N — qualunque N, deciso a query-time, non qui.
Periodo/anno si ricava da timestamp a query-time, nessuna colonna
period_label pre-assegnata (evita di reintrodurre i confini dei 5
periodi che stiamo proprio cercando di superare).
"""

from __future__ import annotations

import os
import pandas as pd

import dukascopy_python
from dukascopy_python.instruments import INSTRUMENT_IDX_EUROPE_E_DAAX, INSTRUMENT_IDX_EUROPE_E_FUTSEE_100
from datetime import datetime, timezone

import engine as eng

DATABASE_ID = "b9fbd4d6-7837-4d86-9c0f-ca60c0cf69e3"
API_BASE = "https://api.cloudflare.com/client/v4/accounts"
INSERT_BATCH = 400

SYMBOLS = {"DAX": INSTRUMENT_IDX_EUROPE_E_DAAX, "FTSE100": INSTRUMENT_IDX_EUROPE_E_FUTSEE_100}
FETCH_START = datetime(2015, 1, 1, tzinfo=timezone.utc)
FETCH_END = datetime(2026, 7, 19, tzinfo=timezone.utc)

KNOWN_LOADED_YEARS = {2015, 2016, 2020, 2023, 2024, 2025, 2026}  # già in D1, per confronto nel log
NEVER_TESTED_YEARS = {2017, 2018, 2019, 2021, 2022}  # mai caricati prima nel progetto


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


def ensure_table(account_id: str, token: str):
    d1_query(
        "CREATE TABLE IF NOT EXISTS adx_diagnostic_raw ("
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

    print("=== Estrazione dati grezzi ADX/EMA — storico continuo 2015-2026 (Dukascopy diretto) ===\n")

    ensure_table(account_id, token)
    d1_query("DELETE FROM adx_diagnostic_raw", account_id, token)
    print("Tabella adx_diagnostic_raw pronta e svuotata (idempotenza).\n")

    total_rows = 0
    for symbol, const in SYMBOLS.items():
        print(f"Scarico {symbol} da Dukascopy (2015-2026 continuo)...")
        df = fetch_full(const)
        if df.empty:
            print(f"  [{symbol}] ERRORE: nessun dato ritornato.")
            continue

        years_present = set(df["timestamp"].dt.year.unique())
        missing_never_tested = NEVER_TESTED_YEARS - years_present
        if missing_never_tested:
            print(f"  [{symbol}] ATTENZIONE: Dukascopy non ha dati per {sorted(missing_never_tested)} "
                  f"— buco alla fonte, non solo mancanza nel caricamento D1 precedente. "
                  f"Verificare se è un limite noto (come già visto per ITALY40).")
        recovered = NEVER_TESTED_YEARS & years_present
        if recovered:
            print(f"  [{symbol}] Recuperati con successo anni mai testati prima: {sorted(recovered)}")

        df["adx"] = eng.adx_wilder(df, 14)
        df["ema20"] = eng.ema(df["close"], 20)
        df["ema50"] = eng.ema(df["close"], 50)
        df = df.dropna(subset=["adx", "ema20", "ema50"]).reset_index(drop=True)
        df["bar_index"] = range(len(df))

        n = len(df)
        print(f"  [{symbol}] {n} barre valide (dopo warmup indicatori), "
              f"{df['timestamp'].min()} -> {df['timestamp'].max()}")

        records = df[["bar_index", "timestamp", "close", "high", "low", "adx", "ema20", "ema50"]].to_dict("records")
        for i in range(0, len(records), INSERT_BATCH):
            batch = records[i:i + INSERT_BATCH]
            values_sql = ",".join(
                f"('{symbol}',{r['bar_index']},'{r['timestamp'].isoformat()}',"
                f"{r['close']},{r['high']},{r['low']},{r['adx']},{r['ema20']},{r['ema50']})"
                for r in batch
            )
            d1_query(
                "INSERT INTO adx_diagnostic_raw "
                "(symbol, bar_index, timestamp, close, high, low, adx, ema20, ema50) "
                f"VALUES {values_sql}",
                account_id, token,
            )
            if (i // INSERT_BATCH) % 20 == 0:
                print(f"    ...{i + len(batch)}/{n} righe inserite")

        total_rows += n
        print(f"  [{symbol}] completato: {n} righe inserite.\n")

    print(f"=== Completato. {total_rows} righe totali in adx_diagnostic_raw (storico continuo). ===")
    print("Nessuna analisi eseguita qui — solo dati grezzi pronti per query esplorative dirette.")


if __name__ == "__main__":
    main()
