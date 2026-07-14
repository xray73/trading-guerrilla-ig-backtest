"""
mfe_check.py — Calcola la Massima Escursione Favorevole (MFE) in unità R
per tutti i 2.152 trade reali dei 5 periodi validati (run_id 7-11).

Domanda a cui risponde: i livelli 2R-5R testati nella scaletta sono
irraggiungibili per questo segnale/strumento in generale, o è il 2023
(train) ad essere un periodo particolarmente povero di trade che
"corrono"? Puramente descrittivo — nessun filtro, nessuna decisione
qui, solo il dato per interpretare correttamente il risultato della
scaletta.

MFE = quanto lontano il prezzo si è mosso A FAVORE della posizione, in
QUALUNQUE momento tra ingresso e uscita — non dove il trade ha
effettivamente chiuso. Un trade chiuso in perdita può comunque aver
avuto un MFE positivo (è andato in profitto prima di invertire e
tornare in perdita) — è esattamente l'informazione che serve per
capire se un tetto più alto avrebbe mai potuto essere raggiunto.

MFE_R = distanza_massima_a_favore / (ATR_al_ingresso × 1.5)

Output: distribuzione aggregata e per periodo di quanti trade hanno
raggiunto almeno 1R, 2R, 3R, 4R, 5R di MFE in qualunque momento.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
import requests

DATABASE_ID = "b9fbd4d6-7837-4d86-9c0f-ca60c0cf69e3"
API_BASE = "https://api.cloudflare.com/client/v4/accounts"
CHUNK_SIZE = 5000
ATR_MULTIPLIER = 1.5  # stesso del motore, per DAX e FTSE100

RUN_IDS = [7, 8, 9, 10, 11]
RUN_LABELS = {7: "2015-2016", 8: "2020-covid", 9: "2023", 10: "2024-2025", 11: "2026-ytd"}
INSTRUMENTS = ["DAX", "FTSE100"]


def d1_query(sql: str, account_id: str, token: str) -> list[dict]:
    url = f"{API_BASE}/{account_id}/d1/database/{DATABASE_ID}/query"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    resp = requests.post(url, headers=headers, json={"sql": sql}, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if not data.get("success"):
        raise RuntimeError(f"D1 query fallita: {data.get('errors')}")
    return data["result"][0]["results"]


def fetch_all_ohlc(symbol: str, account_id: str, token: str) -> pd.DataFrame:
    rows = []
    offset = 0
    while True:
        sql = (
            f"SELECT timestamp, high, low FROM ohlc_prices "
            f"WHERE symbol='{symbol}' ORDER BY timestamp LIMIT {CHUNK_SIZE} OFFSET {offset}"
        )
        batch = d1_query(sql, account_id, token)
        if not batch:
            break
        rows.extend(batch)
        offset += CHUNK_SIZE
        if len(batch) < CHUNK_SIZE:
            break
        time.sleep(0.15)
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def fetch_all_trades(account_id: str, token: str) -> pd.DataFrame:
    run_list = ",".join(str(r) for r in RUN_IDS)
    sql = (
        f"SELECT run_id, symbol, direction, entry_time, entry_price, exit_time, "
        f"atr_at_entry, pnl FROM trades WHERE run_id IN ({run_list}) ORDER BY symbol, entry_time"
    )
    rows = d1_query(sql, account_id, token)
    df = pd.DataFrame(rows)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)
    return df


def compute_mfe_for_symbol(symbol: str, ohlc: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    ohlc_indexed = ohlc.set_index("timestamp")
    results = []

    for _, t in trades.iterrows():
        entry_time, exit_time = t["entry_time"], t["exit_time"]
        atr = t["atr_at_entry"]
        if pd.isna(atr) or atr == 0:
            continue

        window = ohlc_indexed.loc[entry_time:exit_time]
        if window.empty:
            continue

        r_unit = atr * ATR_MULTIPLIER
        if t["direction"] == "long":
            mfe_price = window["high"].max() - t["entry_price"]
        else:
            mfe_price = t["entry_price"] - window["low"].min()

        mfe_r = mfe_price / r_unit
        results.append({
            "symbol": symbol,
            "run_id": t["run_id"],
            "period": RUN_LABELS[t["run_id"]],
            "direction": t["direction"],
            "pnl": t["pnl"],
            "mfe_r": mfe_r,
        })

    return pd.DataFrame(results)


def main():
    token = os.environ.get("CLOUDFLARE_API_TOKEN")
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    if not token or not account_id:
        print("ERRORE: CLOUDFLARE_API_TOKEN o CLOUDFLARE_ACCOUNT_ID mancanti.", file=sys.stderr)
        sys.exit(1)

    os.makedirs("results", exist_ok=True)

    print("Scarico trade reali (run_id 7-11)...")
    trades = fetch_all_trades(account_id, token)
    print(f"  {len(trades)} trade totali")

    all_results = []
    for symbol in INSTRUMENTS:
        print(f"\nScarico OHLC {symbol}...")
        ohlc = fetch_all_ohlc(symbol, account_id, token)
        print(f"  {len(ohlc)} barre")

        symbol_trades = trades[trades["symbol"] == symbol]
        print(f"Calcolo MFE per {len(symbol_trades)} trade {symbol}...")
        res = compute_mfe_for_symbol(symbol, ohlc, symbol_trades)
        all_results.append(res)
        print(f"  {len(res)} trade elaborati")

    df = pd.concat(all_results, ignore_index=True)
    df.to_csv("results/mfe_by_trade.csv", index=False)

    thresholds = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0]

    print("\n" + "=" * 70)
    print("DISTRIBUZIONE MFE — GLOBALE (tutti i periodi, tutti gli strumenti)")
    print("=" * 70)
    print(df["mfe_r"].describe())
    summary_rows = []
    for th in thresholds:
        pct = (df["mfe_r"] >= th).mean() * 100
        n = (df["mfe_r"] >= th).sum()
        print(f"  Trade che raggiungono almeno {th}R (in qualunque momento): {n}/{len(df)} ({pct:.1f}%)")
        summary_rows.append({"scope": "globale", "period": "tutti", "threshold_r": th,
                              "n_reaching": int(n), "pct_reaching": pct})

    print("\n" + "=" * 70)
    print("DISTRIBUZIONE MFE — PER PERIODO (per capire se 2023 è un'anomalia)")
    print("=" * 70)
    for run_id in RUN_IDS:
        label = RUN_LABELS[run_id]
        sub = df[df["run_id"] == run_id]
        if sub.empty:
            continue
        print(f"\n{label} (n={len(sub)}):")
        for th in thresholds:
            pct = (sub["mfe_r"] >= th).mean() * 100
            n = (sub["mfe_r"] >= th).sum()
            print(f"    >= {th}R: {n}/{len(sub)} ({pct:.1f}%)")
            summary_rows.append({"scope": "periodo", "period": label, "threshold_r": th,
                                  "n_reaching": int(n), "pct_reaching": pct})

    print("\n" + "=" * 70)
    print("PER STRUMENTO")
    print("=" * 70)
    for symbol in INSTRUMENTS:
        sub = df[df["symbol"] == symbol]
        print(f"\n{symbol} (n={len(sub)}):")
        for th in thresholds:
            pct = (sub["mfe_r"] >= th).mean() * 100
            n = (sub["mfe_r"] >= th).sum()
            print(f"    >= {th}R: {n}/{len(sub)} ({pct:.1f}%)")
            summary_rows.append({"scope": "strumento", "period": symbol, "threshold_r": th,
                                  "n_reaching": int(n), "pct_reaching": pct})

    pd.DataFrame(summary_rows).to_csv("results/mfe_summary.csv", index=False)
    print("\nCompletato. File in results/.")


if __name__ == "__main__":
    main()
