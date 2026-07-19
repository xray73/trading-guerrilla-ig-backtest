"""
analyze_period_profile.py — Profilo descrittivo (19/07/2026) dei 5
periodi ufficiali, richiesto esplicitamente come analisi ASETTICA: non
costruisce nessuna regola/filtro, misura solo caratteristiche
strutturali del mercato in ciascun periodo, per capire cosa rende il
2026-ytd diverso dagli altri 4 (dove il blocco ADX×ATR era invece
profittevole).

Già escluso (sez. 51 addendum 19/07): durata trend nel bucket bloccato
(2026-ytd aveva la durata PIÙ LUNGA, il contrario di quanto servirebbe),
intensità ATR/ADX nel bucket, spread EMA (smentito a livello di
singolo trade). Questo script aggiunge misure MAI provate oggi:

  - Rendimento e volatilità realizzata del periodo (il mercato è salito/
    sceso/laterale, quanto si è mosso in assoluto)
  - Distribuzione ADX su TUTTO il periodo (non solo nel bucket
    bloccato) — il mercato nel complesso era più/meno "in trend"?
  - Frequenza di cambio di direzione (EMA20 vs EMA50) — un mercato che
    cambia idea spesso è più "indeciso"/whipsaw di uno che si muove
    in modo persistente
  - Correlazione tra i rendimenti giornalieri di DAX e FTSE100 nel
    periodo — il mercato europeo si muoveva in modo più o meno
    "unificato" (macro-driven) in quel periodo?
  - Statistiche VIX/struttura a termine (già in parte note, riportate
    per completezza del profilo)

NESSUNA regola derivata da queste misure in questo script — solo
descrizione, per decidere DOPO se e come approfondire.

Dati: adx_diagnostic_raw (D1) + VIX/VIX3M storico (Yahoo Finance).
Nessuna scrittura su D1.
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

PERIODS = {
    "2015-2016": ("2015-01-01", "2017-01-01"),
    "2020-covid": ("2020-01-01", "2021-01-01"),
    "2023": ("2023-01-01", "2024-01-01"),
    "2024-2025": ("2024-01-01", "2026-01-01"),
    "2026-ytd": ("2026-01-01", "2026-07-19"),
}


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
            "SELECT bar_index, timestamp, close, adx, ema20, ema50 "
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

    log("=== Profilo descrittivo dei 5 periodi ufficiali (analisi asettica) ===\n")

    log("Scarico DAX e FTSE100 completi da D1...")
    dax = fetch_symbol_full("DAX", account_id, token)
    ftse = fetch_symbol_full("FTSE100", account_id, token)
    dax["direction"] = np.where(dax["ema20"] > dax["ema50"], 1, -1)
    ftse["direction"] = np.where(ftse["ema20"] > ftse["ema50"], 1, -1)

    log("Scarico VIX e VIX3M (Yahoo Finance)...")
    vix = yf.download("^VIX", start="2014-10-01", end="2026-07-19", progress=False)
    vix3m = yf.download("^VIX3M", start="2014-10-01", end="2026-07-19", progress=False)
    for d_ in (vix, vix3m):
        if isinstance(d_.columns, pd.MultiIndex):
            d_.columns = d_.columns.get_level_values(0)
    vix = vix.reset_index()[["Date", "Close"]].rename(columns={"Close": "vix"})
    vix3m = vix3m.reset_index()[["Date", "Close"]].rename(columns={"Close": "vix3m"})
    vix["Date"] = pd.to_datetime(vix["Date"]).dt.tz_localize(None).dt.normalize()
    vix3m["Date"] = pd.to_datetime(vix3m["Date"]).dt.tz_localize(None).dt.normalize()
    term = pd.merge(vix, vix3m, on="Date", how="inner")
    term["backwardation"] = term["vix"] > term["vix3m"]
    log("")

    riepilogo = []

    for label, (start, end) in PERIODS.items():
        log(f"{'='*70}\n{label} ({start} -> {end})\n{'='*70}")

        dax_p = dax[(dax["timestamp"] >= pd.Timestamp(start, tz="UTC")) & (dax["timestamp"] < pd.Timestamp(end, tz="UTC"))]
        ftse_p = ftse[(ftse["timestamp"] >= pd.Timestamp(start, tz="UTC")) & (ftse["timestamp"] < pd.Timestamp(end, tz="UTC"))]

        dax_ret_tot = (dax_p["close"].iloc[-1] / dax_p["close"].iloc[0] - 1) * 100
        ftse_ret_tot = (ftse_p["close"].iloc[-1] / ftse_p["close"].iloc[0] - 1) * 100
        dax_daily = dax_p.set_index("timestamp")["close"].resample("1D").last().dropna()
        ftse_daily = ftse_p.set_index("timestamp")["close"].resample("1D").last().dropna()
        dax_daily_ret = dax_daily.pct_change().dropna()
        ftse_daily_ret = ftse_daily.pct_change().dropna()
        dax_vol_annual = dax_daily_ret.std() * np.sqrt(252) * 100
        ftse_vol_annual = ftse_daily_ret.std() * np.sqrt(252) * 100

        log(f"  Rendimento totale periodo — DAX: {dax_ret_tot:+.1f}%  FTSE100: {ftse_ret_tot:+.1f}%")
        log(f"  Volatilita annualizzata (rendimenti giornalieri) — DAX: {dax_vol_annual:.1f}%  FTSE100: {ftse_vol_annual:.1f}%")

        merged_daily = pd.concat([dax_daily_ret.rename("dax"), ftse_daily_ret.rename("ftse")], axis=1).dropna()
        corr = merged_daily["dax"].corr(merged_daily["ftse"])
        log(f"  Correlazione rendimenti giornalieri DAX-FTSE100: {corr:.3f}")

        log(f"  ADX medio (tutto il periodo, non solo contesto trend) — "
            f"DAX: {dax_p['adx'].mean():.2f}  FTSE100: {ftse_p['adx'].mean():.2f}")
        log(f"  Pct barre ADX>20 — DAX: {(dax_p['adx']>20).mean()*100:.1f}%  "
            f"FTSE100: {(ftse_p['adx']>20).mean()*100:.1f}%")

        dax_flips = (dax_p["direction"].diff() != 0).sum()
        ftse_flips = (ftse_p["direction"].diff() != 0).sum()
        dax_flip_rate = dax_flips / len(dax_p) * 1000
        ftse_flip_rate = ftse_flips / len(ftse_p) * 1000
        log(f"  Cambi di direzione EMA20/50 ogni 1000 barre — DAX: {dax_flip_rate:.1f}  FTSE100: {ftse_flip_rate:.1f}")

        term_p = term[(term["Date"] >= pd.Timestamp(start)) & (term["Date"] < pd.Timestamp(end))]
        vix_medio = term_p["vix"].mean() if not term_p.empty else None
        pct_backward = term_p["backwardation"].mean() * 100 if not term_p.empty else None
        if not term_p.empty:
            log(f"  VIX medio: {vix_medio:.2f}  |  Pct giorni backwardation: {pct_backward:.1f}%")

        riepilogo.append({
            "periodo": label,
            "dax_ret_tot": dax_ret_tot, "ftse_ret_tot": ftse_ret_tot,
            "dax_vol_annual": dax_vol_annual, "ftse_vol_annual": ftse_vol_annual,
            "corr_dax_ftse": corr,
            "dax_adx_medio": dax_p["adx"].mean(), "ftse_adx_medio": ftse_p["adx"].mean(),
            "dax_flip_rate_1000barre": dax_flip_rate, "ftse_flip_rate_1000barre": ftse_flip_rate,
            "vix_medio": vix_medio, "pct_backwardation": pct_backward,
        })
        log("")

    log("=" * 70)
    log("TABELLA RIEPILOGATIVA - tutti i periodi affiancati")
    log("=" * 70)
    df_summary = pd.DataFrame(riepilogo)
    log(df_summary.to_string(index=False))

    df_summary.to_csv("results/period_profile_riepilogo.csv", index=False)
    with open("results/analyze_period_profile.txt", "w") as f:
        f.write("\n".join(log_lines))

    print("\n=== Completato. ===")


if __name__ == "__main__":
    main()
