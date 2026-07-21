"""
extract_mr_candidates.py — Infrastruttura "analisi via query", ramo
mean-reversion. Stessa filosofia di extract_v6_candidates.py: estrae
TUTTI i segnali MR validi (eseguiti o no), riusando
generate_mean_reversion_signals(mode="rsi") cosi' com'e' — nessuna
reimplementazione (Bollinger scartata in produzione, PF 0.78 negativo,
non estratta qui).

DIFFERENZA STRUTTURALE rispetto a V6: non esiste una tabella D1 con i
trade MR "ufficiali" gia' calcolati su tutto lo storico (research_v6_
trade_features esisteva solo per V6). Il flag was_executed viene quindi
calcolato QUI, facendo girare il motore reale di produzione
(BacktestEngineMeanReversion, capitale 600 EUR — sotto-pool 30% reale,
Decision Log 17-18/07/2026) sui 5 periodi ufficiali, esattamente come
mean_reversion_full_pipeline.py FASE 2 gia' fa per la validazione
capitale. Stessa struttura a due passate di extract_v6_trade_features.py:
1) segnali calcolati sulla serie COMPLETA (per contesto/persistenza
   corretti, mai frammentata), 2) esecuzione REALE periodo per periodo
   (capitale resettato a inizio periodo, come da convenzione ufficiale).

Universo: DAX, FTSE100 (MR non e' mai stato esteso a GOLD in
produzione — GOLD_CONFIG userebbe comunque lo stesso atr_multiplier di
V6 per coerenza, ma nessuna calibrazione MR-specifica esiste per GOLD,
RCA Addendum 13/07 sez.22 nota esplicita — non incluso qui).

Due tabelle create (IF NOT EXISTS, mai DROP — append-only):

research_mr_candidates (1 riga per candidato):
  candidate_key (PK: instrument + entry_time ISO)
  instrument, direction, entry_time, signal_bar_time,
  adx_at_entry, atr_at_entry, rsi_at_entry, close_entry,
  persistence_bars,      -- barre consecutive di segnale (stesso lato) prima dell'ingresso
  adx_regime_age_bars,   -- barre consecutive con adx<20 (regime MR, lato OPPOSTO a V6) prima dell'ingresso
  vix_entry, vix3m_entry,
  was_executed, matched_trade_key,
  extraction_run_at

research_mr_candidate_path (48 barre fisse, NEUTRO — mai cappato a
stop/target/max_holding):
  candidate_key, bar_offset, timestamp, open, high, low, close,
  price_r,               -- stop_distance = atr_at_entry * atr_multiplier strumento (stesso di V6, MR lo riusa invariato)
  adx, rsi, ema_fast, ema_slow  -- ema aggiunte qui per compatibilita' incrociata con analisi V6 (non usate dal segnale MR stesso, calcolate a parte)

Nessuna scrittura su trades/backtest_runs/live_*/research_v6_*
esistenti. Nessuna modifica a engine.py, engine_mean_reversion.py o
mean_reversion_signals.py.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import requests
import yfinance as yf

import engine as eng
from engine_mean_reversion import BacktestEngineMeanReversion
from mean_reversion_signals import generate_mean_reversion_signals, _rsi_wilder, RSI_PERIOD

D1_DATABASE_ID = "b9fbd4d6-7837-4d86-9c0f-ca60c0cf69e3"
D1_API_BASE = "https://api.cloudflare.com/client/v4/accounts"

SYMBOLS = ["DAX", "FTSE100"]
PATH_BARS = 48
MR_CAPITAL = 600.0  # sotto-pool reale 30% di 2.000 EUR (Decision Log 17-18/07/2026)

PERIODS = [
    ("2015-2016", "2015-01-05", "2016-12-29"),
    ("2020-covid", "2020-01-02", "2020-12-30"),
    ("2023", "2023-01-02", "2023-12-30"),
    ("2024-2025", "2024-01-03", "2025-12-31"),
    ("2026-ytd", "2026-01-05", None),
]


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
    d1_query("""
        CREATE TABLE IF NOT EXISTS research_mr_candidates (
          candidate_key TEXT PRIMARY KEY,
          instrument TEXT NOT NULL,
          direction TEXT NOT NULL,
          entry_time TEXT NOT NULL,
          signal_bar_time TEXT NOT NULL,
          adx_at_entry REAL,
          atr_at_entry REAL,
          rsi_at_entry REAL,
          close_entry REAL,
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
        CREATE TABLE IF NOT EXISTS research_mr_candidate_path (
          candidate_key TEXT NOT NULL,
          bar_offset INTEGER NOT NULL,
          timestamp TEXT NOT NULL,
          open REAL,
          high REAL,
          low REAL,
          close REAL,
          price_r REAL,
          adx REAL,
          rsi REAL,
          ema_fast REAL,
          ema_slow REAL
        )
    """, account_id, token)
    print("Tabelle research_mr_candidates / research_mr_candidate_path pronte (IF NOT EXISTS, append-only).")


def get_already_extracted_keys(account_id: str, token: str) -> set[str]:
    rows, _ = d1_query("SELECT candidate_key FROM research_mr_candidates", account_id, token)
    return {r["candidate_key"] for r in rows}


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


def slice_period(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return df[(df["timestamp"] >= start) & (df["timestamp"] < end)].reset_index(drop=True)


def compute_executed_keys(signals_full: dict, account_id: str, token: str) -> set[str]:
    """Fa girare il motore REALE di produzione (BacktestEngineMeanReversion,
    mode=rsi, capitale 600 EUR sotto-pool) sui 5 periodi ufficiali, per
    determinare quali candidati sono davvero diventati trade — non esiste
    una tabella D1 di riferimento gia' pronta per MR (a differenza di V6),
    quindi va calcolato qui, stessa metodologia di
    mean_reversion_full_pipeline.py FASE 2."""
    executed = set()
    print("\nCalcolo trade REALMENTE eseguiti (motore MR produzione, 600 EUR, 5 periodi ufficiali)...")
    for label, start_str, end_str in PERIODS:
        start = pd.Timestamp(start_str, tz="UTC")
        end = pd.Timestamp(end_str, tz="UTC") + pd.Timedelta(days=1) if end_str else pd.Timestamp.now(tz="UTC")
        sig_period = {name: slice_period(signals_full[name], start, end) for name in SYMBOLS}
        engine_run = BacktestEngineMeanReversion(capital0=MR_CAPITAL, instruments=eng.INSTRUMENTS)
        trades_df, _ = engine_run.run(sig_period)
        n = len(trades_df) if not trades_df.empty else 0
        print(f"  {label}: {n} trade MR reali")
        if trades_df.empty:
            continue
        for _, t in trades_df.iterrows():
            key = f"{t['instrument']}_{pd.Timestamp(t['entry_time']).isoformat()}"
            executed.add(key)
    print(f"  Totale trade MR reali sui 5 periodi: {len(executed)}\n")
    return executed


def main():
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    token = os.environ.get("CLOUDFLARE_API_TOKEN")
    if not account_id or not token:
        print("ERRORE: CLOUDFLARE_ACCOUNT_ID o CLOUDFLARE_API_TOKEN mancanti.")
        return

    from ohlc_data_source import get_ohlc

    ensure_tables(account_id, token)
    already_extracted = get_already_extracted_keys(account_id, token)
    print(f"Candidati MR gia' estratti in precedenza: {len(already_extracted)}\n")

    vix_by_date, vix3m_by_date = load_vix_lookup()

    print("Scarico/aggiorno storico DAX/FTSE100...")
    raw = {name: get_ohlc(name, account_id, token, log=print) for name in SYMBOLS}
    signals = {name: generate_mean_reversion_signals(raw[name], eng.INSTRUMENTS[name], mode="rsi")
               for name in SYMBOLS}
    print("Fatto.\n")

    # ema aggiunte a parte (non calcolate da generate_mean_reversion_signals,
    # servono solo per arricchire il path — nessun impatto sul segnale MR)
    for name in SYMBOLS:
        signals[name]["ema_fast"] = eng.ema(signals[name]["close"], eng.PARAMS.ema_fast)
        signals[name]["ema_slow"] = eng.ema(signals[name]["close"], eng.PARAMS.ema_slow)

    executed_keys = compute_executed_keys(signals, account_id, token)

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

            persistence = count_consecutive_backward(sdf["signal"] == direction, i - 1)
            adx_regime_age = count_consecutive_backward(sdf["adx"] < eng.PARAMS.adx_min_context, i - 1)

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
                "rsi_at_entry": float(prev_bar["rsi"]) if "rsi" in prev_bar and pd.notna(prev_bar["rsi"]) else None,
                "close_entry": float(prev_bar["close"]),
                "persistence_bars": persistence,
                "adx_regime_age_bars": adx_regime_age,
                "vix_entry": vix_val,
                "vix3m_entry": vix3m_val,
                "was_executed": was_executed,
                "matched_trade_key": candidate_key if was_executed else None,
                "extraction_run_at": extraction_ts,
            })

            atr_entry = prev_bar["atr"]
            stop_distance = (atr_entry * eng.INSTRUMENTS[inst].atr_multiplier
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
                    "rsi": float(bar["rsi"]) if "rsi" in bar and pd.notna(bar["rsi"]) else None,
                    "ema_fast": float(bar["ema_fast"]) if pd.notna(bar["ema_fast"]) else None,
                    "ema_slow": float(bar["ema_slow"]) if pd.notna(bar["ema_slow"]) else None,
                })
            n_new += 1

        print(f"  {inst}: candidati nuovi trovati finora nello scan = {n_new}")

    print(f"\nCandidati MR NUOVI da scrivere: {n_new} ({n_skipped_existing} gia' presenti, saltati)")
    print(f"Righe di path da scrivere: {len(path_rows)}")
    if not cand_rows:
        print("Nulla di nuovo da scrivere. Completato.")
        return

    print("\nScrivo research_mr_candidates su D1...")
    df_cand = pd.DataFrame(cand_rows)
    written_c = insert_chunked_adaptive(df_cand, "research_mr_candidates", account_id, token, start_chunk=200)
    print(f"  {written_c} righe scritte in research_mr_candidates.")

    print("\nScrivo research_mr_candidate_path su D1 (puo' richiedere tempo)...")
    df_path = pd.DataFrame(path_rows)
    written_p = insert_chunked_adaptive(df_path, "research_mr_candidate_path", account_id, token, start_chunk=400)
    print(f"  {written_p} righe scritte in research_mr_candidate_path.")

    n_executed_flagged = int(df_cand["was_executed"].sum())
    print(f"\n=== Completato. {len(df_cand)} candidati MR nuovi, di cui {n_executed_flagged} "
          f"marcati was_executed=1. ===")
    print(f"SANITY CHECK: il numero was_executed=1 qui sopra ({n_executed_flagged}) e' calcolato "
          f"dallo STESSO run del motore usato per generarlo (non c'e' una tabella esterna "
          f"indipendente come per V6) — la verifica indipendente possibile e' ricontare i "
          f"trade con una seconda esecuzione pulita di BacktestEngineMeanReversion(capital0=600) "
          f"sui 5 periodi e confrontare il totale, non il confronto contro dati gia' in D1.")


if __name__ == "__main__":
    main()
