"""
ohlc_data_source.py — Fonte dati OHLC condivisa per gli script di test.

Invece di scaricare l'intero storico da Dukascopy ad ogni run (lento,
ridondante — le stesse serie DAX/FTSE100/GOLD 2015-2026 già in D1),
o di leggere ciecamente da D1 rischiando dati vecchi, questa funzione
fa entrambe le cose nell'ordine giusto:
  1. Legge il MAX(timestamp) già presente in D1 (ohlc_prices) per il
     simbolo richiesto.
  2. Se mancano barre tra quell'ultimo timestamp e adesso, le scarica
     da Dukascopy (SOLO il pezzo mancante, non tutto lo storico) e le
     inserisce in D1.
  3. Legge da D1 l'intera serie aggiornata e la ritorna.

Ogni script che chiama get_ohlc(symbol, ...) è quindi sempre aggiornato
automaticamente, senza un passo di manutenzione separato da ricordare.

Simboli supportati: DAX, FTSE100, GOLD (estendibile aggiungendo a
DUKASCOPY_CONST). Nessuna modifica a engine.py. Scrive SOLO righe nuove
in ohlc_prices (mai UPDATE/DELETE su righe esistenti).
"""

from __future__ import annotations

import time
import pandas as pd
import requests

import dukascopy_python
from dukascopy_python.instruments import (
    INSTRUMENT_IDX_EUROPE_E_DAAX, INSTRUMENT_IDX_EUROPE_E_FUTSEE_100,
    INSTRUMENT_FX_METALS_XAU_USD,
)

D1_DATABASE_ID = "b9fbd4d6-7837-4d86-9c0f-ca60c0cf69e3"
D1_API_BASE = "https://api.cloudflare.com/client/v4/accounts"
D1_READ_CHUNK = 5000
D1_INSERT_CHUNK = 500

DUKASCOPY_CONST = {
    "DAX": INSTRUMENT_IDX_EUROPE_E_DAAX,
    "FTSE100": INSTRUMENT_IDX_EUROPE_E_FUTSEE_100,
    "GOLD": INSTRUMENT_FX_METALS_XAU_USD,
}


def _d1_query(sql: str, account_id: str, token: str) -> list[dict]:
    url = f"{D1_API_BASE}/{account_id}/d1/database/{D1_DATABASE_ID}/query"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    resp = requests.post(url, headers=headers, json={"sql": sql}, timeout=60)
    if resp.status_code != 200:
        # mostra il corpo della risposta PRIMA di sollevare, altrimenti
        # raise_for_status() nasconde il motivo reale (es. errore SQL D1)
        print(f"[ohlc_data_source] D1 ha risposto {resp.status_code}: {resp.text[:1000]}")
        print(f"[ohlc_data_source] SQL che ha causato l'errore (primi 500 char): {sql[:500]}")
    resp.raise_for_status()
    data = resp.json()
    if not data.get("success"):
        raise RuntimeError(f"D1 query fallita: {data.get('errors')}")
    return data["result"][0]["results"]


def _get_last_timestamp(symbol: str, account_id: str, token: str) -> pd.Timestamp | None:
    result = _d1_query(f"SELECT MAX(timestamp) as last_ts FROM ohlc_prices WHERE symbol='{symbol}'",
                        account_id, token)
    last_ts = result[0]["last_ts"] if result else None
    if last_ts is None:
        return None
    ts = pd.Timestamp(last_ts)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _fetch_incremental_dukascopy(symbol_const, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    df = dukascopy_python.fetch(
        symbol_const, dukascopy_python.INTERVAL_MIN_30, dukascopy_python.OFFER_SIDE_BID,
        start.to_pydatetime(), end.to_pydatetime(),
    ).reset_index()
    ts_col = df.columns[0]
    df = df.rename(columns={ts_col: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def _insert_rows(symbol: str, df: pd.DataFrame, account_id: str, token: str) -> int:
    before = len(df)
    df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"  [{symbol}] {dropped} righe scartate (valori NaN, tipico di buchi di liquidita' "
              f"weekend/festivi in Dukascopy) — non inserite in D1.")
    if df.empty:
        return 0

    inserted = 0
    for i in range(0, len(df), D1_INSERT_CHUNK):
        chunk = df.iloc[i:i + D1_INSERT_CHUNK]
        values = ", ".join(
            f"('{symbol}', '{row.timestamp.isoformat()}', {row.open}, {row.high}, {row.low}, {row.close}, 0)"
            for row in chunk.itertuples()
        )
        sql = ("INSERT OR IGNORE INTO ohlc_prices (symbol, timestamp, open, high, low, close, volume) "
               f"VALUES {values}")
        _d1_query(sql, account_id, token)
        inserted += len(chunk)
        time.sleep(0.1)
    return inserted


def _read_full_from_d1(symbol: str, account_id: str, token: str) -> pd.DataFrame:
    rows = []
    offset = 0
    while True:
        sql = (
            "SELECT timestamp, open, high, low, close FROM ohlc_prices "
            f"WHERE symbol='{symbol}' ORDER BY timestamp LIMIT {D1_READ_CHUNK} OFFSET {offset}"
        )
        batch = _d1_query(sql, account_id, token)
        if not batch:
            break
        rows.extend(batch)
        offset += D1_READ_CHUNK
        if len(batch) < D1_READ_CHUNK:
            break
        time.sleep(0.1)
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    for col in ("open", "high", "low", "close"):
        df[col] = df[col].astype(float)
    return df.sort_values("timestamp").reset_index(drop=True)


def get_ohlc(symbol: str, account_id: str, token: str, log=print) -> pd.DataFrame:
    """Punto d'ingresso unico: aggiorna D1 se serve, poi ritorna la
    serie OHLC completa e aggiornata per `symbol` (DAX/FTSE100/GOLD)."""
    if symbol not in DUKASCOPY_CONST:
        raise ValueError(f"Simbolo '{symbol}' non supportato da ohlc_data_source.py "
                          f"(disponibili: {list(DUKASCOPY_CONST)})")

    now = pd.Timestamp.now(tz="UTC")
    last_ts = _get_last_timestamp(symbol, account_id, token)

    if last_ts is None:
        raise RuntimeError(f"Nessun dato per '{symbol}' in D1 — serve un primo caricamento completo "
                            f"(load_ohlc_generic.py), questo modulo fa solo aggiornamenti incrementali.")

    start = last_ts + pd.Timedelta(minutes=30)
    if start < now - pd.Timedelta(hours=1):  # margine, evita di rincorrere l'ultima barra parziale
        log(f"  [{symbol}] D1 fermo a {last_ts.isoformat()} — scarico barre mancanti da Dukascopy...")
        new_data = _fetch_incremental_dukascopy(DUKASCOPY_CONST[symbol], start, now)
        if not new_data.empty:
            n = _insert_rows(symbol, new_data, account_id, token)
            log(f"  [{symbol}] {n} righe nuove inserite in D1 (fino a {new_data['timestamp'].max().isoformat()})")
        else:
            log(f"  [{symbol}] Nessuna barra nuova disponibile da Dukascopy.")
    else:
        log(f"  [{symbol}] D1 già aggiornato (ultimo dato: {last_ts.isoformat()})")

    return _read_full_from_d1(symbol, account_id, token)
