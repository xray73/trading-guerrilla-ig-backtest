"""
analyze_vix_per_trade.py — v3 (19/07/2026): test di robustezza
dell'ipotesi VIX rispetto alla SOGLIA ADX usata per selezionare la
popolazione di trade — stesso principio già applicato alle finestre
di persistenza (5/20/40/60 barre) e ai tetti di durata (24/48/96
barre): se un pattern regge solo a una soglia specifica e sparisce
spostandosi di poco, è fragile quanto le altre ipotesi già smentite
oggi (spread EMA, distanza breakout).

SEMPLIFICAZIONE rispetto alla v2: qui NON uso la regola composta
specifica del filtro ADX×ATR (che mescolava ADX+ATR+breakout+trend
ampio) — uso SOLO la soglia ADX pura, a più livelli (20, 25, 30, 35,
40), per isolare l'effetto VIX senza confondere le variabili. Stessa
metodologia stop/target reale (48 barre, parametro vero del sistema)
di tutti i test precedenti.

Per ciascuna soglia ADX e ciascuno dei due strumenti: divide i trade
per fascia VIX del giorno esatto (calmo <15, medio 15-25, panico >25)
e calcola il win rate reale.

Output SOLO aggregato — mai trade singoli elencati.

Dati: adx_diagnostic_raw (D1) + VIX storico giornaliero (Yahoo
Finance). Nessuna scrittura su D1.
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
MAX_HOLD = 48  # parametro vero del sistema

ADX_THRESHOLDS = [20, 25, 30, 35, 40]

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


def fetch_symbol_full(symbol: str, account_id: str, token: str) -> pd.DataFrame:
    rows = []
    offset = 0
    while True:
        sql = (
            "SELECT bar_index, timestamp, close, high, low, atr, adx, ema20, ema50, "
            "ema100, ema200, rolling_high_20, rolling_low_20, rolling_high_40, rolling_low_40 "
            f"FROM adx_diagnostic_raw WHERE symbol='{symbol}' "
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
    row = df.iloc[entry_pos]
    entry_price = row["close"]
    atr = row["atr"]
    if direction == "long":
        stop = entry_price - ATR_STOP_MULT * atr
        target = entry_price + ATR_TARGET_MULT * atr
    else:
        stop = entry_price + ATR_STOP_MULT * atr
        target = entry_price - ATR_TARGET_MULT * atr

    end_pos = min(entry_pos + MAX_HOLD, len(df) - 1)
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

    log("=== VIX per-trade v3 — robustezza a soglia ADX pura (20/25/30/35/40) ===")
    log("    Solo soglia ADX (nessuna condizione ATR/breakout/trend ampio) per isolare l'effetto VIX.\n")

    log("Scarico storico VIX (Yahoo Finance)...")
    vix = yf.download("^VIX", start="2014-10-01", end="2026-07-19", progress=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    vix = vix.reset_index()
    vix["Date"] = pd.to_datetime(vix["Date"]).dt.tz_localize(None).dt.normalize()
    vix_by_date = vix.set_index("Date")["Close"]
    log(f"  {len(vix)} barre VIX caricate.\n")

    all_rows = []

    for symbol in ("DAX", "FTSE100"):
        log(f"--- {symbol} ---")
        df = fetch_symbol_full(symbol, account_id, token)
        log(f"  {len(df)} barre caricate.")

        direction = pd.Series(np.where(df["ema20"] > df["ema50"], "long", "short"), index=df.index)

        # pre-calcola la fascia VIX per OGNI barra una sola volta (indipendente dalla soglia ADX)
        entry_dates = df["timestamp"].dt.tz_localize(None).dt.normalize()
        unique_dates = entry_dates.unique()
        date_to_fascia = {}
        for d in unique_dates:
            vix_val = None
            for delta in range(0, 6):
                check_date = pd.Timestamp(d) - pd.Timedelta(days=delta)
                if check_date in vix_by_date.index:
                    vix_val = vix_by_date.loc[check_date]
                    break
            if vix_val is None:
                date_to_fascia[d] = None
            elif vix_val < VIX_CALMO_SOGLIA:
                date_to_fascia[d] = "calmo"
            elif vix_val > VIX_PANICO_SOGLIA:
                date_to_fascia[d] = "panico"
            else:
                date_to_fascia[d] = "medio"
        vix_fascia_all = entry_dates.map(date_to_fascia)

        for adx_th in ADX_THRESHOLDS:
            positions = df.index[df["adx"] > adx_th].tolist()
            log(f"\n  ADX > {adx_th}: {len(positions)} barre di contesto")

            esiti = {"calmo": {"TARGET": 0, "STOP": 0, "TIMEOUT": 0},
                     "medio": {"TARGET": 0, "STOP": 0, "TIMEOUT": 0},
                     "panico": {"TARGET": 0, "STOP": 0, "TIMEOUT": 0}}

            for pos in positions:
                fascia = vix_fascia_all.iloc[pos]
                if fascia is None:
                    continue
                esito = simulate_outcome(df, pos, direction.iloc[pos])
                esiti[fascia][esito] += 1

            for fascia in ("calmo", "medio", "panico"):
                t, s, to = esiti[fascia]["TARGET"], esiti[fascia]["STOP"], esiti[fascia]["TIMEOUT"]
                n_decisi = t + s
                n_tot = t + s + to
                if n_decisi > 0:
                    wr = t / n_decisi * 100
                    log(f"    {fascia:<8} (n={n_tot:>5}): win rate={wr:.2f}%")
                else:
                    wr = float("nan")
                    log(f"    {fascia:<8}: vuoto")
                all_rows.append({
                    "symbol": symbol, "adx_threshold": adx_th, "fascia_vix": fascia,
                    "n_target": t, "n_stop": s, "n_timeout": to, "win_rate": wr,
                })
        log("")

    log("=" * 70)
    log("RIEPILOGO — 'calmo' è sempre il peggiore, a ogni soglia ADX?")
    log("=" * 70)
    df_all = pd.DataFrame(all_rows)
    for symbol in ("DAX", "FTSE100"):
        for adx_th in ADX_THRESHOLDS:
            sub = df_all[(df_all["symbol"] == symbol) & (df_all["adx_threshold"] == adx_th)]
            sub_valid = sub.dropna(subset=["win_rate"])
            if sub_valid.empty:
                continue
            peggiore = sub_valid.loc[sub_valid["win_rate"].idxmin(), "fascia_vix"]
            log(f"  {symbol} ADX>{adx_th}: fascia peggiore = {peggiore}")

    df_all.to_csv("results/vix_per_trade_v3_riepilogo.csv", index=False)
    with open("results/analyze_vix_per_trade_v3.txt", "w") as f:
        f.write("\n".join(log_lines))

    print("\n=== Completato. ===")


if __name__ == "__main__":
    main()
