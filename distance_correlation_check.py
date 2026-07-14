"""
distance_correlation_check.py — Verifica se la distanza (in unita' ATR) tra
il prezzo di ingresso e il livello di breakout appena rotto correla con
l'esito del trade — SOLO correlazione grezza, nessun filtro costruito qui
(stesso approccio del reversal di prezzo, RCA sez.6.4: prima verificare se
il pattern esiste nei dati reali, poi eventualmente tradurlo in regola).

Metrica (mai look-ahead — usa solo dati fino alla barra segnale, quella
precedente all'ingresso):
    distance_atr = |prezzo_ingresso - livello_breakout_rotto| / ATR_barra_segnale

livello_breakout_rotto = rolling_high (long) o rolling_low (short) della
barra segnale — lo stesso identico livello che il motore usa per decidere
se scatta il breakout (engine.py, generate_signals). distance_atr alto =
il prezzo era gia' andato ben oltre il livello quando e' scattato
l'ingresso ("tardi nel movimento"); basso = ingresso appena sopra/sotto
il livello.

Dati: i 2.152 trade reali dei 5 run validati (run_id 7-11, Variante 6,
motore v2) — non trade simulati, non ipotetici.

Output: results/distance_correlation.csv (dati grezzi trade-level, piccolo
— poche migliaia di righe x 6 colonne) e un riepilogo a console con
correlazione Pearson/Spearman globale e per bucket di distanza.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
import requests
from scipy import stats

import engine as eng

DATABASE_ID = "b9fbd4d6-7837-4d86-9c0f-ca60c0cf69e3"
API_BASE = "https://api.cloudflare.com/client/v4/accounts"
CHUNK_SIZE = 5000

RUN_IDS = [7, 8, 9, 10, 11]  # DAX_FTSE100_V6_{periodo}, i 5 run validati
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
            f"SELECT timestamp, open, high, low, close FROM ohlc_prices "
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
        f"SELECT run_id, symbol, direction, entry_time, entry_price, "
        f"pnl, rr_realized, causa_esito FROM trades "
        f"WHERE run_id IN ({run_list}) ORDER BY symbol, entry_time"
    )
    rows = d1_query(sql, account_id, token)
    df = pd.DataFrame(rows)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    return df


def compute_distance_for_symbol(symbol: str, ohlc: pd.DataFrame,
                                  trades: pd.DataFrame) -> pd.DataFrame:
    inst = eng.INSTRUMENTS[symbol]
    sig = eng.compute_indicators(ohlc, inst)  # rolling_high/low, atr, adx già calcolati

    # indicizza per lookup veloce O(1) invece di scan ripetuti
    sig_indexed = sig.set_index("timestamp")
    ts_index = sig_indexed.index

    results = []
    for _, t in trades.iterrows():
        entry_time = t["entry_time"]
        if entry_time not in ts_index:
            continue
        entry_pos = ts_index.get_loc(entry_time)
        if entry_pos == 0:
            continue
        signal_row = sig.iloc[entry_pos - 1]  # barra segnale = quella precedente
        atr = signal_row["atr"]
        if pd.isna(atr) or atr == 0:
            continue

        if t["direction"] == "long":
            level = signal_row["rolling_high"]
            if pd.isna(level):
                continue
            distance = t["entry_price"] - level
        else:
            level = signal_row["rolling_low"]
            if pd.isna(level):
                continue
            distance = level - t["entry_price"]

        distance_atr = distance / atr
        results.append({
            "symbol": symbol,
            "run_id": t["run_id"],
            "direction": t["direction"],
            "entry_time": str(entry_time),
            "distance_atr": distance_atr,
            "pnl": t["pnl"],
            "rr_realized": t["rr_realized"],
            "causa_esito": t["causa_esito"],
            "win": t["pnl"] > 0,
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
        print(f"Calcolo distance_atr per {len(symbol_trades)} trade {symbol}...")
        res = compute_distance_for_symbol(symbol, ohlc, symbol_trades)
        all_results.append(res)
        print(f"  {len(res)} trade elaborati con successo")

    df = pd.concat(all_results, ignore_index=True)
    df.to_csv("results/distance_correlation.csv", index=False)

    print("\n" + "=" * 70)
    print("RISULTATO — correlazione distance_atr vs esito trade")
    print("=" * 70)

    valid = df.dropna(subset=["distance_atr", "rr_realized"])
    pearson_r, pearson_p = stats.pearsonr(valid["distance_atr"], valid["rr_realized"])
    spearman_r, spearman_p = stats.spearmanr(valid["distance_atr"], valid["rr_realized"])
    print(f"\nGlobale (n={len(valid)}):")
    print(f"  Pearson  r={pearson_r:.4f}  p={pearson_p:.4f}")
    print(f"  Spearman r={spearman_r:.4f}  p={spearman_p:.4f}")

    print(f"\nPer strumento:")
    for symbol in INSTRUMENTS:
        sub = valid[valid["symbol"] == symbol]
        if len(sub) < 10:
            continue
        r, p = stats.pearsonr(sub["distance_atr"], sub["rr_realized"])
        print(f"  {symbol}: Pearson r={r:.4f} p={p:.4f} (n={len(sub)})")

    print(f"\nWin rate per quartile di distance_atr (globale):")
    valid = valid.copy()
    valid["quartile"] = pd.qcut(valid["distance_atr"], 4, labels=["Q1 (più vicino)", "Q2", "Q3", "Q4 (più lontano)"])
    summary = valid.groupby("quartile", observed=True).agg(
        win_rate=("win", "mean"),
        avg_rr=("rr_realized", "mean"),
        n=("win", "count"),
        distance_range=("distance_atr", lambda x: f"{x.min():.2f}..{x.max():.2f}")
    )
    print(summary.to_string())
    summary.to_csv("results/distance_quartile_summary.csv")

    print("\nCompletato. File in results/.")


if __name__ == "__main__":
    main()
