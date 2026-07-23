"""
extract_v6_trade_features_continuous.py — Estensione di
extract_v6_trade_features.py al dataset OHLC continuo (2015-2026 per
DAX dopo il backfill di ieri; FTSE100 resta a 5 periodi finche' non
backfillato anche lui — questo script non richiede/impone nulla,
semplicemente processa quello che c'e' in ohlc_prices per ciascun
simbolo, gap o non gap).

MOTIVAZIONE (22/07/2026): la combinazione VR>1 + corr_dax_ftse alta su
DAX (dentro il regime gia' validato ATR% basso) mostra l'R medio piu'
alto visto in tutta la sessione (+0.596R) ma su un campione di soli
n=13 trade — troppo pochi per qualunque test. Estendere l'estrazione
al continuo dovrebbe aumentare il campione per DAX (nuovi trade nei 6
anni appena colmati: 2017-2019, 2021-2022).

DIFFERENZA CHIAVE rispetto allo script originale: NON tocca
research_v6_trade_features / research_v6_trade_path (il dataset
ufficiale usato per tutti i confronti di oggi, 2156 trade sui 5
periodi ufficiali) — scrive in tabelle NUOVE E SEPARATE
(research_v6_trade_features_continuous / _path_continuous), CREATE
TABLE IF NOT EXISTS (non DROP, idempotente). Additivo, nessun rischio
di invalidare confronti gia' fatti oggi.

Il campo `periodo` distingue i 5 periodi ufficiali (stessa label di
sempre) dai nuovi anni colmati ieri (label "gap_colmato"), cosi' e'
sempre possibile filtrare per restare confrontabili con l'analisi
originale se serve.

Stessa identica logica di calcolo MFE/MAE, breakout_distance,
persistence_bars ecc. dello script originale (copiata 1:1, nessuna
modifica alla metodologia).
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import requests

import engine as eng
from engine_floating_kill_switch import BacktestEngineFloatingKillSwitch

CAPITAL0 = 2000.0
SYMBOLS = ["DAX", "FTSE100"]
D1_DATABASE_ID = "b9fbd4d6-7837-4d86-9c0f-ca60c0cf69e3"
D1_API_BASE = "https://api.cloudflare.com/client/v4/accounts"

# Un solo range continuo invece dei 5 periodi discreti — processa
# qualunque dato sia effettivamente presente in ohlc_prices, gap
# compresi (i gap semplicemente non generano barre/trade li', nessun
# trattamento speciale necessario).
OFFICIAL_PERIODS = [
    ("2015-2016", "2015-01-05", "2017-01-01"),
    ("2020-covid", "2020-01-02", "2021-01-01"),
    ("2023", "2023-01-02", "2024-01-01"),
    ("2024-2025", "2024-01-03", "2026-01-01"),
    ("2026-ytd", "2026-01-05", None),
]
FULL_RANGE_START = "2015-01-01"
FULL_RANGE_END = None  # fino ad oggi


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
    # CREATE TABLE IF NOT EXISTS — MAI drop. Tabelle nuove, separate
    # dall'ufficiale research_v6_trade_features/_path.
    d1_query("""
        CREATE TABLE IF NOT EXISTS research_v6_trade_features_continuous (
          trade_key TEXT PRIMARY KEY,
          periodo TEXT NOT NULL,
          instrument TEXT NOT NULL,
          direction TEXT NOT NULL,
          entry_time TEXT NOT NULL,
          exit_time TEXT,
          exit_reason TEXT,
          pnl REAL,
          r_multiple REAL,
          risk_amount REAL,
          size REAL,
          forced_min_size INTEGER,
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
          hold_bars INTEGER,
          adx_at_exit REAL,
          adx_max REAL,
          adx_max_bar_offset INTEGER,
          adx_min REAL,
          adx_slope_per_bar REAL,
          mfe_r REAL,
          mfe_bar_offset INTEGER,
          mae_r REAL,
          mae_bar_offset INTEGER,
          extraction_run_at TEXT
        )
    """, account_id, token)

    d1_query("""
        CREATE TABLE IF NOT EXISTS research_v6_trade_path_continuous (
          trade_key TEXT NOT NULL,
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
    print("Tabelle _continuous pronte (create se non esistevano, mai droppate).")


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


def slice_period(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return df[(df["timestamp"] >= start) & (df["timestamp"] < end)].reset_index(drop=True)


def label_periodo(ts: pd.Timestamp) -> str:
    for label, start_str, end_str in OFFICIAL_PERIODS:
        start = pd.Timestamp(start_str, tz="UTC")
        end = pd.Timestamp(end_str, tz="UTC") + pd.Timedelta(days=1) if end_str else pd.Timestamp.now(tz="UTC")
        if start <= ts < end:
            return label
    return "gap_colmato"  # 2017-2019, 2021-2022


def count_consecutive_backward(bool_series: pd.Series, end_idx: int) -> int:
    i = end_idx
    count = 0
    while i >= 0 and bool(bool_series.iloc[i]):
        count += 1
        i -= 1
    return count


def main():
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    token = os.environ.get("CLOUDFLARE_API_TOKEN")
    if not account_id or not token:
        print("ERRORE: CLOUDFLARE_ACCOUNT_ID o CLOUDFLARE_API_TOKEN mancanti.")
        return

    from ohlc_data_source import get_ohlc

    ensure_tables(account_id, token)

    print("\nScarico/aggiorno storico DAX/FTSE100 (continuo, quello che c'e' in D1)...")
    raw = {name: get_ohlc(name, account_id, token, log=print) for name in SYMBOLS}
    signals_full = {name: eng.generate_signals(raw[name], eng.INSTRUMENTS[name]) for name in SYMBOLS}
    print("Fatto.\n")

    start = pd.Timestamp(FULL_RANGE_START, tz="UTC")
    end = pd.Timestamp.now(tz="UTC")

    extraction_ts = datetime.now(timezone.utc).isoformat()
    feature_rows = []
    path_rows = []

    sig_period = {name: slice_period(signals_full[name], start, end) for name in SYMBOLS}

    print("Eseguo motore su tutto il range continuo disponibile...")
    engine_run = BacktestEngineFloatingKillSwitch(capital0=CAPITAL0, instruments=eng.INSTRUMENTS)
    trades_df, _ = engine_run.run(sig_period)
    print(f"  {len(trades_df)} trade totali (continuo) — estraggo feature...\n")

    for _, t in trades_df.iterrows():
        inst = t["instrument"]
        sdf = signals_full[inst]
        direction = t["direction"]

        signal_bar_ts = t["entry_time"] - pd.Timedelta(minutes=eng.PARAMS.bar_minutes)
        sig_idx_arr = sdf.index[sdf["timestamp"] == signal_bar_ts]
        if len(sig_idx_arr) == 0:
            continue
        sig_idx = sig_idx_arr[0]
        sig_bar = sdf.loc[sig_idx]

        persistence = count_consecutive_backward(sdf["signal"] == direction, sig_idx)
        adx_regime_age = count_consecutive_backward(sdf["adx"] > eng.PARAMS.adx_min_context, sig_idx)

        breakout_level = sig_bar["rolling_high"] if direction == "long" else sig_bar["rolling_low"]
        breakout_dist = (sig_bar["close"] - breakout_level) if direction == "long" else (breakout_level - sig_bar["close"])

        path_mask = (sdf["timestamp"] >= t["entry_time"]) & (sdf["timestamp"] <= t["exit_time"])
        path = sdf.loc[path_mask].reset_index(drop=True)
        if path.empty:
            continue

        stop_distance = t["atr_at_entry"] * eng.INSTRUMENTS[inst].atr_multiplier

        if direction == "long":
            price_r_close = (path["close"] - t["entry_price"]) / stop_distance
        else:
            price_r_close = (t["entry_price"] - path["close"]) / stop_distance

        trade_key = f"{inst}_{t['entry_time'].isoformat()}"

        n_bars = len(path)
        favorable_candidates = []
        adverse_candidates = []
        for i in range(n_bars):
            is_exit_bar = (i == n_bars - 1)
            is_entry_bar = (i == 0)
            if is_exit_bar:
                r_val = float(t["r_multiple"])
                favorable_candidates.append(r_val)
                adverse_candidates.append(r_val)
            elif is_entry_bar:
                favorable_candidates.append(0.0)
                adverse_candidates.append(0.0)
            else:
                bar = path.iloc[i]
                if direction == "long":
                    favorable_candidates.append((bar["high"] - t["entry_price"]) / stop_distance)
                    adverse_candidates.append((bar["low"] - t["entry_price"]) / stop_distance)
                else:
                    favorable_candidates.append((t["entry_price"] - bar["low"]) / stop_distance)
                    adverse_candidates.append((t["entry_price"] - bar["high"]) / stop_distance)

        mfe_r = max(favorable_candidates)
        mfe_bar_offset = int(np.argmax(favorable_candidates))
        mae_r = min(adverse_candidates)
        mae_bar_offset = int(np.argmin(adverse_candidates))

        bar_offsets = np.arange(len(path))
        adx_values = path["adx"].values
        valid = ~np.isnan(adx_values)
        adx_slope = float(np.polyfit(bar_offsets[valid], adx_values[valid], 1)[0]) if valid.sum() >= 2 else None

        periodo_label = label_periodo(t["entry_time"])

        feature_rows.append({
            "trade_key": trade_key, "periodo": periodo_label, "instrument": inst,
            "direction": direction, "entry_time": t["entry_time"].isoformat(),
            "exit_time": t["exit_time"].isoformat(), "exit_reason": t["exit_reason"],
            "pnl": float(t["pnl"]), "r_multiple": float(t["r_multiple"]),
            "risk_amount": float(t["risk_amount"]), "size": float(t["size"]),
            "forced_min_size": int(bool(t["forced_min_size"])),
            "adx_at_entry": float(t["adx_at_entry"]), "atr_at_entry": float(t["atr_at_entry"]),
            "ema_fast_entry": float(sig_bar["ema_fast"]), "ema_slow_entry": float(sig_bar["ema_slow"]),
            "ema_broad_fast_entry": float(sig_bar["ema_broad_fast"]), "ema_broad_slow_entry": float(sig_bar["ema_broad_slow"]),
            "close_entry": float(sig_bar["close"]), "breakout_level_entry": float(breakout_level),
            "breakout_distance_pts": float(breakout_dist),
            "persistence_bars": persistence, "adx_regime_age_bars": adx_regime_age,
            "hold_bars": len(path) - 1,
            "adx_at_exit": float(path["adx"].iloc[-1]) if pd.notna(path["adx"].iloc[-1]) else None,
            "adx_max": float(np.nanmax(adx_values)), "adx_max_bar_offset": int(np.nanargmax(adx_values)),
            "adx_min": float(np.nanmin(adx_values)), "adx_slope_per_bar": adx_slope,
            "mfe_r": float(mfe_r), "mfe_bar_offset": mfe_bar_offset,
            "mae_r": float(mae_r), "mae_bar_offset": mae_bar_offset,
            "extraction_run_at": extraction_ts,
        })

        for i in range(len(path)):
            path_rows.append({
                "trade_key": trade_key, "bar_offset": i,
                "timestamp": path["timestamp"].iloc[i].isoformat(),
                "open": float(path["open"].iloc[i]) if pd.notna(path["open"].iloc[i]) else None,
                "high": float(path["high"].iloc[i]) if pd.notna(path["high"].iloc[i]) else None,
                "low": float(path["low"].iloc[i]) if pd.notna(path["low"].iloc[i]) else None,
                "close": float(path["close"].iloc[i]), "price_r": float(price_r_close.iloc[i]),
                "adx": float(path["adx"].iloc[i]) if pd.notna(path["adx"].iloc[i]) else None,
                "ema_fast": float(path["ema_fast"].iloc[i]) if pd.notna(path["ema_fast"].iloc[i]) else None,
                "ema_slow": float(path["ema_slow"].iloc[i]) if pd.notna(path["ema_slow"].iloc[i]) else None,
            })

    print(f"\nFeature estratte per {len(feature_rows)} trade, {len(path_rows)} righe di percorso totali.")
    if not feature_rows:
        print("Nessun dato da scrivere.")
        return

    n_gap = sum(1 for r in feature_rows if r["periodo"] == "gap_colmato")
    print(f"  Di cui {n_gap} trade nei nuovi anni colmati (gap_colmato) — "
          f"trade aggiuntivi mai visti nell'estrazione ufficiale.")

    print("\nScrivo research_v6_trade_features_continuous su D1...")
    df_feat = pd.DataFrame(feature_rows)
    total_written = insert_chunked_adaptive(df_feat, "research_v6_trade_features_continuous",
                                             account_id, token, start_chunk=300)
    print(f"  {total_written} righe scritte.")

    print("\nScrivo research_v6_trade_path_continuous su D1...")
    df_path = pd.DataFrame(path_rows)
    total_written_p = insert_chunked_adaptive(df_path, "research_v6_trade_path_continuous",
                                               account_id, token, start_chunk=300)
    print(f"  {total_written_p} righe scritte.")

    print(f"\n=== Completato. research_v6_trade_features ufficiale INVARIATA "
          f"(2156 trade, 5 periodi) — questa e' una tabella aggiuntiva separata. ===")


if __name__ == "__main__":
    main()
