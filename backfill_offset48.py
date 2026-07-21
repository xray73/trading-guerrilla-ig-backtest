"""
backfill_offset48.py — Fix mirato: aggiunge la barra a bar_offset=48
(mancante per un errore di conteggio off-by-one, scoperto 21/07/2026)
ai candidati V6 e MR gia' estratti in research_v6_candidate_path /
research_mr_candidate_path.

CAUSA: nel motore reale, bars_held = bar_index - entry_bar_index, e il
controllo max_holding (bars_held >= 48) scatta alla barra di offset 48
rispetto all'entrata — non 47. Il path estratto (PATH_BARS=48, offset
0-47) non includeva mai quella barra, quindi nessun candidato poteva
mai essere mascherato come "uscito per max_holding" con dati corretti.

FIX PERMANENTE: PATH_BARS portato a 49 in extract_v6_candidates.py e
extract_mr_candidates.py per le estrazioni future (candidati non
ancora estratti). Questo script backfilla SOLO la barra 48 mancante
per gli 8.884 candidati (8.070 V6 + 814 MR) gia' scritti prima del fix
— non tocca le barre 0-47 gia' corrette, non duplica nulla (INSERT OR
REPLACE su bar_offset=48, chiave naturale candidate_key+bar_offset).

Nessuna scrittura su trades/backtest_runs/live_*. Nessuna modifica a
engine.py, engine_three_asset_gold.py, mean_reversion_signals.py.
"""

from __future__ import annotations

import os
import time
import numpy as np
import pandas as pd
import requests

import engine as eng
from engine_three_asset_gold import GOLD_CONFIG
from mean_reversion_signals import generate_mean_reversion_signals

D1_DATABASE_ID = "b9fbd4d6-7837-4d86-9c0f-ca60c0cf69e3"
D1_API_BASE = "https://api.cloudflare.com/client/v4/accounts"

INSTRUMENTS_ALL = dict(eng.INSTRUMENTS)
INSTRUMENTS_ALL["GOLD"] = GOLD_CONFIG


def d1_query(sql: str, account_id: str, token: str) -> tuple[list[dict], dict]:
    url = f"{D1_API_BASE}/{account_id}/d1/database/{D1_DATABASE_ID}/query"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    resp = requests.post(url, headers=headers, json={"sql": sql}, timeout=60)
    try:
        resp.raise_for_status()
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"D1 HTTP {resp.status_code}: {resp.text[:500]}") from e
    data = resp.json()
    if not data.get("success"):
        raise RuntimeError(f"Query D1 fallita: {data.get('errors')}")
    block = data["result"][0]
    return block["results"], block.get("meta", {})


def fmt_val(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "NULL"
    if isinstance(v, str):
        return "'" + v.replace("'", "''") + "'"
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v)


def insert_chunked(rows: list[dict], table: str, account_id: str, token: str, chunk_size: int = 300) -> int:
    if not rows:
        return 0
    df = pd.DataFrame(rows)
    cols = list(df.columns)
    total = 0
    i = 0
    while i < len(df):
        chunk = df.iloc[i:i + chunk_size]
        values = ", ".join(f"({', '.join(fmt_val(r[c]) for c in cols)})" for _, r in chunk.iterrows())
        sql = f"INSERT OR REPLACE INTO {table} ({', '.join(cols)}) VALUES {values}"
        _r, meta = d1_query(sql, account_id, token)
        total += meta.get("changes", 0)
        i += len(chunk)
        time.sleep(0.1)
    return total


def backfill_v6(account_id: str, token: str, signals_full: dict):
    print("\n=== Backfill V6 (offset 48) ===")
    rows, _ = d1_query(
        "SELECT candidate_key, instrument, entry_time, atr_at_entry FROM research_v6_candidates",
        account_id, token)
    print(f"  {len(rows)} candidati V6 da controllare.")

    new_rows = []
    for r in rows:
        inst = r["instrument"]
        sdf = signals_full[inst]
        entry_time = pd.Timestamp(r["entry_time"])
        entry_idx_arr = sdf.index[sdf["timestamp"] == entry_time]
        if len(entry_idx_arr) == 0:
            continue
        entry_idx = entry_idx_arr[0]
        target_idx = entry_idx + 48
        if target_idx >= len(sdf):
            continue  # candidato troppo vicino alla fine dello storico, nessuna barra 48 disponibile
        bar = sdf.iloc[target_idx]

        atr_entry = r.get("atr_at_entry")
        stop_distance = atr_entry * INSTRUMENTS_ALL[inst].atr_multiplier if atr_entry else None
        # direzione non salvata nella select sopra ma serve solo il segno del price_r,
        # ricava dal segno gia' presente implicitamente: ricalcoliamo separatamente
        new_rows.append({
            "candidate_key": r["candidate_key"], "bar_offset": 48,
            "timestamp": bar["timestamp"].isoformat(),
            "open": float(bar["open"]) if pd.notna(bar["open"]) else None,
            "high": float(bar["high"]) if pd.notna(bar["high"]) else None,
            "low": float(bar["low"]) if pd.notna(bar["low"]) else None,
            "close": float(bar["close"]) if pd.notna(bar["close"]) else None,
            "price_r": None,  # placeholder, ricalcolato sotto con la direzione corretta
            "adx": float(bar["adx"]) if pd.notna(bar["adx"]) else None,
            "ema_fast": float(bar["ema_fast"]) if pd.notna(bar["ema_fast"]) else None,
            "ema_slow": float(bar["ema_slow"]) if pd.notna(bar["ema_slow"]) else None,
            "_entry_idx": entry_idx, "_inst": inst, "_stop_distance": stop_distance,
        })

    # seconda passata: serve la direzione e il close dell'entrata per price_r —
    # rilette dalla tabella candidati stessa (colonna direction/close_entry gia' presenti)
    dir_rows, _ = d1_query(
        "SELECT candidate_key, direction, close_entry FROM research_v6_candidates", account_id, token)
    dir_map = {d["candidate_key"]: (d["direction"], d["close_entry"]) for d in dir_rows}

    final_rows = []
    for nr in new_rows:
        ck = nr["candidate_key"]
        direction, close_entry = dir_map.get(ck, (None, None))
        stop_distance = nr.pop("_stop_distance")
        nr.pop("_entry_idx"); nr.pop("_inst")
        if direction and stop_distance and stop_distance > 0 and close_entry is not None and nr["close"] is not None:
            nr["price_r"] = ((nr["close"] - close_entry) / stop_distance if direction == "long"
                              else (close_entry - nr["close"]) / stop_distance)
        final_rows.append(nr)

    print(f"  {len(final_rows)} righe offset=48 da scrivere (V6).")
    written = insert_chunked(final_rows, "research_v6_candidate_path", account_id, token)
    print(f"  {written} righe scritte in research_v6_candidate_path.")


def backfill_mr(account_id: str, token: str, signals_full: dict):
    print("\n=== Backfill MR (offset 48) ===")
    rows, _ = d1_query(
        "SELECT candidate_key, instrument, entry_time, atr_at_entry, direction, close_entry "
        "FROM research_mr_candidates", account_id, token)
    print(f"  {len(rows)} candidati MR da controllare.")

    final_rows = []
    for r in rows:
        inst = r["instrument"]
        sdf = signals_full[inst]
        entry_time = pd.Timestamp(r["entry_time"])
        entry_idx_arr = sdf.index[sdf["timestamp"] == entry_time]
        if len(entry_idx_arr) == 0:
            continue
        entry_idx = entry_idx_arr[0]
        target_idx = entry_idx + 48
        if target_idx >= len(sdf):
            continue
        bar = sdf.iloc[target_idx]

        atr_entry = r.get("atr_at_entry")
        stop_distance = atr_entry * eng.INSTRUMENTS[inst].atr_multiplier if atr_entry else None
        close_val = float(bar["close"]) if pd.notna(bar["close"]) else None
        price_r = None
        if r["direction"] and stop_distance and stop_distance > 0 and close_val is not None:
            price_r = ((close_val - r["close_entry"]) / stop_distance if r["direction"] == "long"
                       else (r["close_entry"] - close_val) / stop_distance)

        final_rows.append({
            "candidate_key": r["candidate_key"], "bar_offset": 48,
            "timestamp": bar["timestamp"].isoformat(),
            "open": float(bar["open"]) if pd.notna(bar["open"]) else None,
            "high": float(bar["high"]) if pd.notna(bar["high"]) else None,
            "low": float(bar["low"]) if pd.notna(bar["low"]) else None,
            "close": close_val,
            "price_r": price_r,
            "adx": float(bar["adx"]) if pd.notna(bar["adx"]) else None,
            "rsi": float(bar["rsi"]) if "rsi" in bar.index and pd.notna(bar["rsi"]) else None,
            "ema_fast": float(bar["ema_fast"]) if pd.notna(bar["ema_fast"]) else None,
            "ema_slow": float(bar["ema_slow"]) if pd.notna(bar["ema_slow"]) else None,
        })

    print(f"  {len(final_rows)} righe offset=48 da scrivere (MR).")
    written = insert_chunked(final_rows, "research_mr_candidate_path", account_id, token)
    print(f"  {written} righe scritte in research_mr_candidate_path.")


def main():
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    token = os.environ.get("CLOUDFLARE_API_TOKEN")
    if not account_id or not token:
        print("ERRORE: CLOUDFLARE_ACCOUNT_ID o CLOUDFLARE_API_TOKEN mancanti.")
        return

    from ohlc_data_source import get_ohlc

    print("Scarico/aggiorno storico DAX/FTSE100/GOLD...")
    raw = {name: get_ohlc(name, account_id, token, log=print) for name in ("DAX", "FTSE100", "GOLD")}
    signals_v6 = {name: eng.generate_signals(raw[name], INSTRUMENTS_ALL[name]) for name in raw}
    signals_mr = {name: generate_mean_reversion_signals(raw[name], eng.INSTRUMENTS[name], mode="rsi")
                  for name in ("DAX", "FTSE100")}
    # aggiungo ema anche alle serie MR (coerente con extract_mr_candidates.py)
    for name in signals_mr:
        signals_mr[name]["ema_fast"] = eng.ema(signals_mr[name]["close"], eng.PARAMS.ema_fast)
        signals_mr[name]["ema_slow"] = eng.ema(signals_mr[name]["close"], eng.PARAMS.ema_slow)
    print("Fatto.\n")

    backfill_v6(account_id, token, signals_v6)
    backfill_mr(account_id, token, signals_mr)

    print("\n=== Backfill completato. ===")


if __name__ == "__main__":
    main()
