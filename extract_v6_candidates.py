"""
extract_v6_candidates.py — Infrastruttura "analisi via query" fase 2.

Estende research_v6_trade_features/path (che contengono SOLO i trade
davvero eseguiti) a TUTTI i segnali V6 validi generati sui prezzi,
eseguiti o no. Il segnale d'ingresso dipende SOLO dai prezzi (EMA/ADX/
breakout/trend ampio) — mai dallo stato di slot/capitale — quindi può
essere ricalcolato per intero riusando eng.generate_signals() cosi'
com'e' (nessuna reimplementazione, stesso identico codice del motore
di produzione).

Universo: DAX, FTSE100, GOLD (parametri GOLD da
engine_three_asset_gold.GOLD_CONFIG, RCA Addendum 13/07 sez.22 —
"ipotesi di lavoro solida, non parametro definitivo").

Due tabelle create (IF NOT EXISTS, mai DROP — dataset append-only per
decisione esplicita 21/07/2026):

research_v6_candidates (1 riga per candidato):
  candidate_key (PK: instrument + entry_time ISO)
  instrument, direction, entry_time, signal_bar_time,
  adx_at_entry, atr_at_entry, ema_fast_entry, ema_slow_entry,
  ema_broad_fast_entry, ema_broad_slow_entry, close_entry,
  breakout_level_entry, breakout_distance_pts,
  persistence_bars, adx_regime_age_bars,
  vix_entry, vix3m_entry,
  was_executed (0/1 — 1 solo se DAX/FTSE100 E il trade esiste davvero
    in research_v6_trade_features; GOLD sempre 0, mai stato in produzione),
  matched_trade_key (nullable, per join diretto quando was_executed=1),
  extraction_run_at

research_v6_candidate_path (48 barre FISSE per candidato, NEUTRO — MAI
cappato a stop/target/max_holding, quello lo applica chi analizza dopo
con la regola che vuole testare):
  candidate_key, bar_offset (0-47), timestamp,
  open, high, low, close,          -- OHLC grezzo, high/low per tocco intrabar esatto
  price_r,                          -- close-based, stop_distance = atr_at_entry * atr_multiplier strumento (riferimento V6 baseline, ricalcolabile diversamente da chi analizza)
  adx, ema_fast, ema_slow

Nessuna scrittura su trades/backtest_runs/live_*/research_v6_trade_*
esistenti. Nessuna modifica a engine.py o alle sottoclassi.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import requests
import yfinance as yf

import engine as eng
from engine_three_asset_gold import GOLD_CONFIG

D1_DATABASE_ID = "b9fbd4d6-7837-4d86-9c0f-ca60c0cf69e3"
D1_API_BASE = "https://api.cloudflare.com/client/v4/accounts"

SYMBOLS = ["DAX", "FTSE100", "GOLD"]
PATH_BARS = 48  # fisso, come max_holding attuale — vedi decisione 21/07/2026

# Copia locale degli strumenti + GOLD (stessa convenzione di
# engine_three_asset_gold.py — mai modificare eng.INSTRUMENTS)
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


def ensure_tables(account_id: str, token: str):
    # IF NOT EXISTS, mai DROP — dataset append-only (decisione 21/07/2026,
    # niente cap temporale, spazio verificato: 108MB/5GB, 2.1% del free tier)
    d1_query("""
        CREATE TABLE IF NOT EXISTS research_v6_candidates (
          candidate_key TEXT PRIMARY KEY,
          instrument TEXT NOT NULL,
          direction TEXT NOT NULL,
          entry_time TEXT NOT NULL,
          signal_bar_time TEXT NOT NULL,
          adx_at_entry REAL,
          atr_at_entry REAL,
          ema_fast_entry REAL,
          ema_slow_entry REAL,
          ema_broad_fast_entry REAL,
          ema_broad_slow_entry REAL,
          close_entry REAL,
          breakout_level_entry REAL,
          breakout_distance_pts REAL,
          persistence_bars INTEGER,
          adx_regime_age_bars INTEGER,
          vix_entry REAL,
          vix3m_entry REAL,
          was_executed INTEGER,
          matched_trade_key TEXT,
          extraction_run_at TEXT
        )
    """, account_id, token)

    d1_query("""
        CREATE TABLE IF NOT EXISTS research_v6_candidate_path (
          candidate_key TEXT NOT NULL,
          bar_offset INTEGER NOT NULL,
          timestamp TEXT NOT NULL,
          open REAL,
          high REAL,
          low REAL,
          close REAL,
          price_r REAL,
          adx REAL,
          ema_fast REAL,
          ema_slow REAL
        )
    """, account_id, token)
    print("Tabelle research_v6_candidates / research_v6_candidate_path pronte (IF NOT EXISTS, append-only).")


def get_already_extracted_keys(account_id: str, token: str) -> set[str]:
    """Legge le candidate_key gia' presenti — permette run incrementali
    future (aggiungere solo i candidati nuovi) senza duplicare lavoro,
    coerente con lo spirito append-only."""
    rows, _ = d1_query("SELECT candidate_key FROM research_v6_candidates", account_id, token)
    return {r["candidate_key"] for r in rows}


def get_executed_trade_keys(account_id: str, token: str) -> set[str]:
    """trade_key esistenti in research_v6_trade_features (formato identico:
    instrument_entrytimeISO) — usato per il flag was_executed. Solo
    DAX/FTSE100 possono comparire (GOLD non e' mai stato in produzione)."""
    rows, _ = d1_query("SELECT trade_key FROM research_v6_trade_features", account_id, token)
    return {r["trade_key"] for r in rows}


def fmt_val(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "NULL"
    if isinstance(v, str):
        return "'" + v.replace("'", "''") + "'"
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v)


def insert_chunked_adaptive(df: pd.DataFrame, table: str, account_id: str, token: str,
                             start_chunk: int = 300, min_chunk: int = 5) -> int:
    if df.empty:
        return 0
    cols = list(df.columns)
    total_written = 0
    i = 0
    chunk_size = start_chunk
    while i < len(df):
        chunk = df.iloc[i:i + chunk_size]
        values = ", ".join(
            f"({', '.join(fmt_val(r[c]) for c in cols)})" for _, r in chunk.iterrows()
        )
        sql = f"INSERT OR REPLACE INTO {table} ({', '.join(cols)}) VALUES {values}"
        try:
            _results, meta = d1_query(sql, account_id, token)
        except RuntimeError as e:
            msg = str(e).lower()
            too_big = ("sqlite_toobig" in msg or "statement too long" in msg
                       or "d1 http 400" in msg or "d1 http 413" in msg
                       or "too large" in msg or "payload too large" in msg)
            if too_big:
                if chunk_size <= min_chunk:
                    raise RuntimeError(f"Chunk gia' al minimo ({min_chunk}) ma ancora troppo lungo.")
                chunk_size = max(chunk_size // 2, min_chunk)
                print(f"  [{table}] statement troppo lungo, riduco chunk a {chunk_size} e riprovo...")
                continue
            raise
        changes = meta.get("changes", 0)
        if changes == 0 and len(chunk) > 0:
            print(f"  ATTENZIONE [{table}]: chunk di {len(chunk)} righe inviato ma changes=0.")
        total_written += changes
        i += len(chunk)
        time.sleep(0.1)
    return total_written


def count_consecutive_backward(bool_series: pd.Series, end_idx: int) -> int:
    i = end_idx
    count = 0
    while i >= 0 and bool(bool_series.iloc[i]):
        count += 1
        i -= 1
    return count


def load_vix_lookup() -> tuple[dict, dict]:
    """VIX + VIX3M storico giornaliero (Yahoo Finance), stessa logica di
    lookback fino a 5 giorni indietro gia' usata in analyze_vix_per_trade.py
    (weekend/festivi: usa l'ultimo valore disponibile)."""
    print("Scarico storico VIX + VIX3M (Yahoo Finance)...")
    vix = yf.download("^VIX", start="2014-10-01", progress=False)
    vix3m = yf.download("^VIX3M", start="2014-10-01", progress=False)
    for df in (vix, vix3m):
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
    vix = vix.reset_index()
    vix3m = vix3m.reset_index()
    vix["Date"] = pd.to_datetime(vix["Date"]).dt.tz_localize(None).dt.normalize()
    vix3m["Date"] = pd.to_datetime(vix3m["Date"]).dt.tz_localize(None).dt.normalize()
    vix_by_date = vix.set_index("Date")["Close"]
    vix3m_by_date = vix3m.set_index("Date")["Close"]
    print(f"  VIX: {len(vix)} barre, VIX3M: {len(vix3m)} barre.\n")
    return vix_by_date, vix3m_by_date


def lookup_with_backfill(series: pd.Series, date: pd.Timestamp, max_days_back: int = 5):
    for delta in range(0, max_days_back + 1):
        check_date = date - pd.Timedelta(days=delta)
        if check_date in series.index:
            val = series.loc[check_date]
            return float(val) if pd.notna(val) else None
    return None


def main():
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    token = os.environ.get("CLOUDFLARE_API_TOKEN")
    if not account_id or not token:
        print("ERRORE: CLOUDFLARE_ACCOUNT_ID o CLOUDFLARE_API_TOKEN mancanti.")
        return

    from ohlc_data_source import get_ohlc

    ensure_tables(account_id, token)
    already_extracted = get_already_extracted_keys(account_id, token)
    executed_keys = get_executed_trade_keys(account_id, token)
    print(f"Candidati gia' estratti in precedenza: {len(already_extracted)}")
    print(f"Trade realmente eseguiti (DAX/FTSE100, per il flag was_executed): {len(executed_keys)}\n")

    vix_by_date, vix3m_by_date = load_vix_lookup()

    print("Scarico/aggiorno storico DAX/FTSE100/GOLD...")
    raw = {name: get_ohlc(name, account_id, token, log=print) for name in SYMBOLS}
    signals = {name: eng.generate_signals(raw[name], INSTRUMENTS_ALL[name]) for name in SYMBOLS}
    print("Fatto.\n")

    extraction_ts = datetime.now(timezone.utc).isoformat()
    cand_rows = []
    path_rows = []
    n_new = 0
    n_skipped_existing = 0

    for inst in SYMBOLS:
        sdf = signals[inst]
        n = len(sdf)
        print(f"--- {inst}: {n} barre totali ---")

        for i in range(1, n):
            prev_bar = sdf.iloc[i - 1]
            if prev_bar["signal"] not in ("long", "short"):
                continue

            entry_bar = sdf.iloc[i]
            direction = prev_bar["signal"]
            entry_time = entry_bar["timestamp"]
            candidate_key = f"{inst}_{entry_time.isoformat()}"

            if candidate_key in already_extracted:
                n_skipped_existing += 1
                continue

            breakout_level = prev_bar["rolling_high"] if direction == "long" else prev_bar["rolling_low"]
            breakout_dist = ((prev_bar["close"] - breakout_level) if direction == "long"
                              else (breakout_level - prev_bar["close"]))

            persistence = count_consecutive_backward(sdf["signal"] == direction, i - 1)
            adx_regime_age = count_consecutive_backward(sdf["adx"] > eng.PARAMS.adx_min_context, i - 1)

            entry_date_naive = pd.Timestamp(entry_time).tz_localize(None).normalize()
            vix_val = lookup_with_backfill(vix_by_date, entry_date_naive)
            vix3m_val = lookup_with_backfill(vix3m_by_date, entry_date_naive)

            was_executed = 1 if candidate_key in executed_keys else 0

            cand_rows.append({
                "candidate_key": candidate_key,
                "instrument": inst,
                "direction": direction,
                "entry_time": entry_time.isoformat(),
                "signal_bar_time": prev_bar["timestamp"].isoformat(),
                "adx_at_entry": float(prev_bar["adx"]) if pd.notna(prev_bar["adx"]) else None,
                "atr_at_entry": float(prev_bar["atr"]) if pd.notna(prev_bar["atr"]) else None,
                "ema_fast_entry": float(prev_bar["ema_fast"]) if pd.notna(prev_bar["ema_fast"]) else None,
                "ema_slow_entry": float(prev_bar["ema_slow"]) if pd.notna(prev_bar["ema_slow"]) else None,
                "ema_broad_fast_entry": float(prev_bar["ema_broad_fast"]) if pd.notna(prev_bar["ema_broad_fast"]) else None,
                "ema_broad_slow_entry": float(prev_bar["ema_broad_slow"]) if pd.notna(prev_bar["ema_broad_slow"]) else None,
                "close_entry": float(prev_bar["close"]),
                "breakout_level_entry": float(breakout_level) if pd.notna(breakout_level) else None,
                "breakout_distance_pts": float(breakout_dist) if pd.notna(breakout_dist) else None,
                "persistence_bars": persistence,
                "adx_regime_age_bars": adx_regime_age,
                "vix_entry": vix_val,
                "vix3m_entry": vix3m_val,
                "was_executed": was_executed,
                "matched_trade_key": candidate_key if was_executed else None,
                "extraction_run_at": extraction_ts,
            })

            # --- path 48 barre NEUTRO, mai cappato a stop/target/max_holding ---
            atr_entry = prev_bar["atr"]
            stop_distance = (atr_entry * INSTRUMENTS_ALL[inst].atr_multiplier
                              if pd.notna(atr_entry) else None)
            end_idx = min(i + PATH_BARS, n)
            for offset, j in enumerate(range(i, end_idx)):
                bar = sdf.iloc[j]
                if stop_distance and stop_distance > 0:
                    price_r = ((bar["close"] - entry_bar["close"]) / stop_distance if direction == "long"
                               else (entry_bar["close"] - bar["close"]) / stop_distance)
                else:
                    price_r = None
                path_rows.append({
                    "candidate_key": candidate_key,
                    "bar_offset": offset,
                    "timestamp": bar["timestamp"].isoformat(),
                    "open": float(bar["open"]) if pd.notna(bar["open"]) else None,
                    "high": float(bar["high"]) if pd.notna(bar["high"]) else None,
                    "low": float(bar["low"]) if pd.notna(bar["low"]) else None,
                    "close": float(bar["close"]) if pd.notna(bar["close"]) else None,
                    "price_r": float(price_r) if price_r is not None and pd.notna(price_r) else None,
                    "adx": float(bar["adx"]) if pd.notna(bar["adx"]) else None,
                    "ema_fast": float(bar["ema_fast"]) if pd.notna(bar["ema_fast"]) else None,
                    "ema_slow": float(bar["ema_slow"]) if pd.notna(bar["ema_slow"]) else None,
                })
            n_new += 1

        print(f"  {inst}: candidati nuovi trovati finora nello scan = {n_new}")

    print(f"\nCandidati NUOVI da scrivere: {n_new} ({n_skipped_existing} gia' presenti, saltati)")
    print(f"Righe di path da scrivere: {len(path_rows)}")
    if not cand_rows:
        print("Nulla di nuovo da scrivere. Completato.")
        return

    print("\nScrivo research_v6_candidates su D1...")
    df_cand = pd.DataFrame(cand_rows)
    written_c = insert_chunked_adaptive(df_cand, "research_v6_candidates", account_id, token, start_chunk=200)
    print(f"  {written_c} righe scritte in research_v6_candidates.")

    print("\nScrivo research_v6_candidate_path su D1 (puo' richiedere tempo, molte righe)...")
    df_path = pd.DataFrame(path_rows)
    written_p = insert_chunked_adaptive(df_path, "research_v6_candidate_path", account_id, token, start_chunk=400)
    print(f"  {written_p} righe scritte in research_v6_candidate_path.")

    n_executed_flagged = int(df_cand["was_executed"].sum())
    print(f"\n=== Completato. {len(df_cand)} candidati nuovi, di cui {n_executed_flagged} "
          f"marcati was_executed=1. ===")
    print("SANITY CHECK DA FARE MANUALMENTE DOPO: confrontare n_executed_flagged "
          "(solo DAX+FTSE100) con il totale di righe in research_v6_trade_features "
          "(atteso: 2.156) — devono combaciare esattamente, altrimenti la logica "
          "del segnale qui non e' identica a quella di produzione.")


if __name__ == "__main__":
    main()
