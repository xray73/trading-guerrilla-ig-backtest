"""
analyze_adx_threshold_dax_vs_ftse.py — Analisi ad-hoc (18/07/2026), due
domande distinte:

  1. La soglia ADX>20 (usata da V6 come filtro di contesto) PREDICE
     davvero un movimento favorevole ugualmente bene per DAX e
     FTSE100, o funziona meglio per uno dei due? Riusa
     compute_persistence() da persistence_check_generic.py SENZA
     modifiche (stessa metodologia già validata su ITALY40/SMI/IBEX35)
     — per ciascuno dei 5 periodi ufficiali, calcola la persistenza
     (frazione di barre in cui, dato ADX>20 + direzione EMA20/50, il
     prezzo si muove davvero a favore nelle 20 barre successive).

  2. Come si muove l'ADX dei due strumenti nel tempo? Serie mensile
     (media) di ADX(14) su tutto lo storico 2015-2026 disponibile in
     D1, per entrambi gli strumenti — output compatto (un numero al
     mese per strumento, ~140 mesi) pensato per essere riportato in
     chat e visualizzato come grafico.

MOTIVAZIONE: la diagnostica sul periodo fuori campione giugno-luglio
2026 ha mostrato FTSE100 con win rate 10% contro il 40% di DAX. Prima
di trattarlo come rumore su un campione piccolo, verifichiamo se la
soglia ADX>20 ha una qualità predittiva diversa tra i due strumenti
anche nei 5 periodi storici ufficiali (dove il segnale è già validato
su entrambi).

Dati: D1 (ohlc_prices), nessun fetch Dukascopy nuovo. Nessuna scrittura
su D1. Output aggregato, mai barre/trade singoli.
"""

from __future__ import annotations

import os
import time
import pandas as pd
import requests

import engine as eng
from persistence_check_generic import compute_persistence, PERIODS

DATABASE_ID = "b9fbd4d6-7837-4d86-9c0f-ca60c0cf69e3"
API_BASE = "https://api.cloudflare.com/client/v4/accounts"
CHUNK_SIZE = 5000
SYMBOLS = ("DAX", "FTSE100")
WARMUP_DAYS = 30


def d1_query(sql: str, account_id: str, token: str) -> list[dict]:
    url = f"{API_BASE}/{account_id}/d1/database/{DATABASE_ID}/query"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    resp = requests.post(url, headers=headers, json={"sql": sql}, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if not data.get("success"):
        raise RuntimeError(f"D1 query fallita: {data.get('errors')}")
    return data["result"][0]["results"]


def fetch_symbol_range(symbol: str, start: str, end: str, account_id: str, token: str) -> pd.DataFrame:
    rows = []
    offset = 0
    while True:
        sql = (
            f"SELECT timestamp, open, high, low, close FROM ohlc_prices "
            f"WHERE symbol='{symbol}' AND timeframe='30m' "
            f"AND timestamp >= '{start}' AND timestamp < '{end}' "
            f"ORDER BY timestamp LIMIT {CHUNK_SIZE} OFFSET {offset}"
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
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def main():
    os.makedirs("results", exist_ok=True)
    log_lines = []

    def log(msg):
        print(msg)
        log_lines.append(msg)

    token = os.environ.get("CLOUDFLARE_API_TOKEN")
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    if not token or not account_id:
        log("ERRORE: CLOUDFLARE_API_TOKEN o CLOUDFLARE_ACCOUNT_ID mancanti.")
        return

    # ================================================================
    # PARTE 1 — persistenza (la soglia ADX>20 "funziona" per entrambi?)
    # ================================================================
    log("=" * 70)
    log("PARTE 1 — Persistenza direzionale con ADX>20 (DAX vs FTSE100)")
    log("=" * 70)
    log("Metodologia invariata da persistence_check_generic.py (già usata per ITALY40).\n")

    persistence_rows = []
    for symbol in SYMBOLS:
        log(f"--- {symbol} ---")
        for period_label, (start_str, end_str) in PERIODS.items():
            start = pd.Timestamp(start_str, tz="UTC") - pd.Timedelta(days=WARMUP_DAYS)
            end = pd.Timestamp(end_str, tz="UTC") + pd.Timedelta(days=1)
            df = fetch_symbol_range(symbol, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"),
                                     account_id, token)
            if df.empty:
                log(f"  {period_label}: nessun dato, salto.")
                continue
            result = compute_persistence(df)
            result["symbol"] = symbol
            result["period"] = period_label
            persistence_rows.append(result)
            log(f"  {period_label}: n_trend={result['n_contesto_trend']} "
                f"({result['pct_barre_in_trend']*100:.1f}% del totale) — "
                f"persistenza LONG={result['persistenza_long']*100:.1f}% "
                f"SHORT={result['persistenza_short']*100:.1f}% "
                f"COMBINATA={result['persistenza_combinata']*100:.1f}%")
        log("")

    log("=" * 70)
    log("RIEPILOGO PERSISTENZA — confronto diretto DAX vs FTSE100 per periodo")
    log("=" * 70)
    df_pers = pd.DataFrame(persistence_rows)
    if not df_pers.empty:
        pivot = df_pers.pivot(index="period", columns="symbol", values="persistenza_combinata")
        for period_label in PERIODS:
            if period_label not in pivot.index:
                continue
            row = pivot.loc[period_label]
            dax_v = row.get("DAX", float("nan"))
            ftse_v = row.get("FTSE100", float("nan"))
            log(f"  {period_label:<12} DAX={dax_v*100:.1f}%  FTSE100={ftse_v*100:.1f}%  "
                f"differenza={((dax_v-ftse_v)*100):+.1f}pt")
        log(f"\n  Media 5 periodi — DAX: {pivot['DAX'].mean()*100:.1f}%  "
            f"FTSE100: {pivot['FTSE100'].mean()*100:.1f}%")
        df_pers.to_csv("results/adx_threshold_persistence.csv", index=False)

    # ================================================================
    # PARTE 2 — serie mensile ADX (per grafico nel tempo)
    # ================================================================
    log("\n" + "=" * 70)
    log("PARTE 2 — Serie mensile ADX medio, storico completo (per grafico)")
    log("=" * 70)

    monthly_rows = []
    full_start, full_end = "2015-01-01", "2026-07-19"
    for symbol in SYMBOLS:
        df = fetch_symbol_range(symbol, full_start, full_end, account_id, token)
        if df.empty:
            log(f"  [{symbol}] nessun dato.")
            continue
        df["adx"] = eng.adx_wilder(df, 14)
        df["month"] = df["timestamp"].dt.to_period("M").astype(str)
        monthly_mean = df.groupby("month")["adx"].mean()
        for month, val in monthly_mean.items():
            monthly_rows.append({"symbol": symbol, "month": month, "adx_mean": val})

    df_monthly = pd.DataFrame(monthly_rows)
    if not df_monthly.empty:
        df_monthly.to_csv("results/adx_monthly_series.csv", index=False)
        log("Serie mensile salvata in results/adx_monthly_series.csv "
            f"({len(df_monthly)} righe totali, {df_monthly['month'].nunique()} mesi x 2 strumenti).")
        log("\nAnteprima (prime 10 righe per strumento):")
        for symbol in SYMBOLS:
            sub = df_monthly[df_monthly["symbol"] == symbol].head(10)
            log(f"  {symbol}:")
            for _, r in sub.iterrows():
                log(f"    {r['month']}: {r['adx_mean']:.2f}")

        log("\n--- SERIE COMPLETA (formato compatto, per riportare in chat) ---")
        pivot_monthly = df_monthly.pivot(index="month", columns="symbol", values="adx_mean")
        log("month,DAX,FTSE100")
        for month, row in pivot_monthly.iterrows():
            dax_v = row.get("DAX", float("nan"))
            ftse_v = row.get("FTSE100", float("nan"))
            log(f"{month},{dax_v:.2f},{ftse_v:.2f}")

    with open("results/analyze_adx_threshold_dax_vs_ftse.txt", "w") as f:
        f.write("\n".join(log_lines))

    print("\n=== Completato. ===")


if __name__ == "__main__":
    main()
