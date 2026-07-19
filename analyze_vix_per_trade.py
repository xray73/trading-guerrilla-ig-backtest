"""
analyze_vix_per_trade.py — Test a livello di SINGOLO TRADE (19/07/2026),
non di media per periodo — lezione imparata dallo spread EMA (sez.
precedente): un pattern che sembra pulito nelle medie per periodo può
sparire completamente quando testato trade per trade. Verifichiamo se
lo stesso destino tocca l'ipotesi VIX prima di crederci.

Unisce il VIX giornaliero (Yahoo Finance) a ciascun trade del "bucket
bloccato" DAX (ADX>30 E ATR%>=0.25, la condizione che
engine_adx_atr_filtro.py blocca) per DATA ESATTA — non per medie di
periodo. Classifica ogni trade come:
  - "calmo"  (VIX del giorno < 15)
  - "panico" (VIX del giorno > 25)
  - "medio"  (15-25, l'ipotesi: "né calmo né in panico" = peggiore)

Calcola il win rate REALE (stop/target fissi, max holding 48 barre —
stessa metodologia già validata oggi, non persistenza a punto fisso)
per ciascuna categoria, su TUTTO lo storico insieme (non periodo per
periodo) — massima potenza statistica disponibile.

Output SOLO aggregato per categoria (win/loss per fascia VIX) — MAI
elenco di singoli trade, coerente con le regole del progetto sui dati
individuali in chat.

Dati: adx_diagnostic_raw (D1, già esteso con tutti gli indicatori) +
VIX storico giornaliero (Yahoo Finance, ^VIX). Nessuna scrittura su D1.
"""

from __future__ import annotations

import os
import time
import pandas as pd
import numpy as np
import requests
import yfinance as yf

DATABASE_ID = "b9fbd4d6-7837-4d86-9c0f-ca60c0cf69e3"
API_BASE = "https://api.cloudflare.com/client/v4/accounts"
CHUNK_SIZE = 5000

ATR_STOP_MULT = 1.5
ATR_TARGET_MULT = 3.0
MAX_HOLD_BARS = 48

VIX_CALMO_SOGLIA = 15.0
VIX_PANICO_SOGLIA = 25.0


def d1_query(sql: str, account_id: str, token: str) -> list[dict]:
    url = f"{API_BASE}/{account_id}/d1/database/{DATABASE_ID}/query"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    resp = requests.post(url, headers=headers, json={"sql": sql}, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if not data.get("success"):
        raise RuntimeError(f"D1 query fallita: {data.get('errors')}")
    return data["result"][0]["results"]


def fetch_dax_full(account_id: str, token: str) -> pd.DataFrame:
    rows = []
    offset = 0
    while True:
        sql = (
            "SELECT bar_index, timestamp, close, high, low, atr, adx, ema20, ema50, "
            "ema100, ema200, rolling_high_20, rolling_low_20 "
            "FROM adx_diagnostic_raw WHERE symbol='DAX' "
            f"ORDER BY bar_index LIMIT {CHUNK_SIZE} OFFSET {offset}"
        )
        batch = d1_query(sql, account_id, token)
        if not batch:
            break
        rows.extend(batch)
        offset += CHUNK_SIZE
        if len(batch) < CHUNK_SIZE:
            break
        time.sleep(0.1)
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("bar_index").reset_index(drop=True)


def simulate_outcome(df: pd.DataFrame, entry_pos: int, direction: str) -> str:
    """Simula l'esito reale (stop/target fissi, max 48 barre) a partire
    dalla posizione entry_pos nel dataframe (indice posizionale, non
    bar_index). Stessa metodologia già validata oggi."""
    row = df.iloc[entry_pos]
    entry_price = row["close"]
    atr = row["atr"]
    if direction == "long":
        stop = entry_price - ATR_STOP_MULT * atr
        target = entry_price + ATR_TARGET_MULT * atr
    else:
        stop = entry_price + ATR_STOP_MULT * atr
        target = entry_price - ATR_TARGET_MULT * atr

    end_pos = min(entry_pos + MAX_HOLD_BARS, len(df) - 1)
    for j in range(entry_pos + 1, end_pos + 1):
        bar = df.iloc[j]
        if direction == "long":
            if bar["low"] <= stop:
                return "STOP"
            if bar["high"] >= target:
                return "TARGET"
        else:
            if bar["high"] >= stop:
                return "STOP"
            if bar["low"] <= target:
                return "TARGET"
    return "TIMEOUT"


def main():
    token = os.environ.get("CLOUDFLARE_API_TOKEN")
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    if not token or not account_id:
        print("ERRORE: CLOUDFLARE_API_TOKEN o CLOUDFLARE_ACCOUNT_ID mancanti.")
        return

    os.makedirs("results", exist_ok=True)
    log_lines = []

    def log(msg):
        print(msg)
        log_lines.append(msg)

    log("=== VIX per-trade — bucket bloccato DAX (ADX>30 + ATR%>=0.25) ===\n")

    log("Scarico dati DAX completi da D1...")
    dax = fetch_dax_full(account_id, token)
    log(f"  {len(dax)} barre caricate.\n")

    log("Scarico storico VIX (Yahoo Finance)...")
    vix = yf.download("^VIX", start="2014-10-01", end="2026-07-19", progress=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    vix = vix.reset_index()
    vix["Date"] = pd.to_datetime(vix["Date"]).dt.tz_localize(None).dt.normalize()
    vix = vix[["Date", "Close"]].rename(columns={"Close": "vix_close"}).sort_values("Date").reset_index(drop=True)
    log(f"  {len(vix)} barre VIX caricate.\n")

    # --- identifica il bucket bloccato: ADX>30, ATR%>=0.25, breakout vero, trend ampio confermato ---
    dax["atr_pct"] = dax["atr"] / dax["close"] * 100
    dax["direction"] = np.where(dax["ema20"] > dax["ema50"], "long", "short")
    dax["dist_r"] = np.where(
        dax["direction"] == "long",
        (dax["close"] - dax["rolling_high_20"]) / dax["atr"],
        (dax["rolling_low_20"] - dax["close"]) / dax["atr"],
    )
    trend_ampio_ok = np.where(
        dax["direction"] == "long", dax["ema100"] > dax["ema200"], dax["ema100"] < dax["ema200"]
    )

    mask_bucket = (
        (dax["adx"] > 30) & (dax["atr_pct"] >= 0.25) & (dax["dist_r"] >= 0) & trend_ampio_ok
    )
    bucket_positions = dax.index[mask_bucket].tolist()
    log(f"Trade nel bucket bloccato trovati: {len(bucket_positions)}\n")

    # --- simula esito reale per ciascun trade del bucket, poi unisce al VIX per data ---
    log("Simulo esito reale (stop/target fissi, max 48 barre) e unisco al VIX per data...")
    vix_by_date = vix.set_index("Date")["vix_close"]

    risultati = []
    for pos in bucket_positions:
        entry_ts = dax.iloc[pos]["timestamp"]
        entry_date = entry_ts.tz_localize(None).normalize()
        direction = dax.iloc[pos]["direction"]
        esito = simulate_outcome(dax, pos, direction)

        # trova il valore VIX del giorno, o del giorno di trading precedente più vicino (fino a 5gg indietro)
        vix_val = None
        for delta in range(0, 6):
            check_date = entry_date - pd.Timedelta(days=delta)
            if check_date in vix_by_date.index:
                vix_val = vix_by_date.loc[check_date]
                break

        if vix_val is None:
            continue

        if vix_val < VIX_CALMO_SOGLIA:
            fascia = "calmo"
        elif vix_val > VIX_PANICO_SOGLIA:
            fascia = "panico"
        else:
            fascia = "medio"

        risultati.append({"esito": esito, "fascia_vix": fascia, "vix_val": vix_val})

    df_ris = pd.DataFrame(risultati)
    log(f"Trade con VIX abbinato con successo: {len(df_ris)} / {len(bucket_positions)}\n")

    log("=" * 70)
    log("WIN RATE REALE PER FASCIA VIX (giorno esatto del trade, non media di periodo)")
    log("=" * 70)
    for fascia in ("calmo", "medio", "panico"):
        sub = df_ris[df_ris["fascia_vix"] == fascia]
        n_target = (sub["esito"] == "TARGET").sum()
        n_stop = (sub["esito"] == "STOP").sum()
        n_timeout = (sub["esito"] == "TIMEOUT").sum()
        n_decisi = n_target + n_stop
        win_rate = n_target / n_decisi * 100 if n_decisi > 0 else float("nan")
        if n_decisi > 0:
            log(f"  {fascia:<8} (n={len(sub):>4}, VIX medio={sub['vix_val'].mean():.2f}): "
                f"TARGET={n_target} STOP={n_stop} TIMEOUT={n_timeout}  win rate={win_rate:.2f}%")
        else:
            log(f"  {fascia:<8}: campione vuoto")

    df_ris.to_csv("results/vix_per_trade_dettaglio.csv", index=False)
    with open("results/analyze_vix_per_trade.txt", "w") as f:
        f.write("\n".join(log_lines))

    print("\n=== Completato. ===")


if __name__ == "__main__":
    main()
