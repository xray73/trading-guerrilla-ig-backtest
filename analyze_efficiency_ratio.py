"""
analyze_efficiency_ratio.py — Test (19/07/2026): l'Efficiency Ratio di
Kaufman (ER = |movimento netto in N barre| / somma dei movimenti
assoluti barra-per-barra) predice la qualità del trade meglio del
caso? Nato dall'osservazione che il DAX nel 2026-ytd aveva letture
tecniche normali (ADX, EMA) ma un rendimento netto sul periodo
drammaticamente più basso di ogni altro periodo — l'ipotesi è che
l'ER catturi in tempo reale la differenza tra "trend vero" (molto
movimento netto per poco movimento totale) e "rumore che si annulla"
(molto movimento totale, poco progresso netto), diversamente
dall'ATR (che misura solo l'AMPIEZZA, non l'EFFICIENZA del movimento).

METODOLOGIA — protocollo completo fin dall'inizio:
  - Meccanismo di trading reale (stop 1.5xATR, target 3xATR, max 48
    barre), non un proxy.
  - Popolazione: contesto V6 vero (ADX>20 + breakout + trend ampio).
  - Robustezza: ER calcolato su 4 finestre diverse (10/20/30/40 barre)
    — se il pattern regge solo a una finestra specifica, è sospetto.
  - Errore standard riportato per ogni cella.

ER = |close[i] - close[i-N]| / sum(|close[j]-close[j-1]| per j in [i-N+1, i])
Range 0-1: vicino a 1 = movimento efficiente (trend vero), vicino a 0
= molto rumore per poco progresso netto.

Output SOLO aggregato — mai trade singoli elencati.

Dati: adx_diagnostic_raw (D1). Nessun dato esterno, nessuna scrittura
su D1.
"""

from __future__ import annotations

import os
import time
import math
import pandas as pd
import numpy as np
import requests

DATABASE_ID = "b9fbd4d6-7837-4d86-9c0f-ca60c0cf69e3"
API_BASE = "https://api.cloudflare.com/client/v4/accounts"
CHUNK_SIZE = 5000

ATR_STOP_MULT = 1.5
ATR_TARGET_MULT = 3.0
MAX_HOLD = 48
ER_WINDOWS = [10, 20, 30, 40]


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


def se_of_r_total(n_target: int, n_stop: int) -> float:
    n = n_target + n_stop
    if n == 0:
        return float("nan")
    p = n_target / n
    mean_r = p * 2 + (1 - p) * (-1)
    e_r2 = p * 4 + (1 - p) * 1
    var_r = e_r2 - mean_r ** 2
    return math.sqrt(n * var_r)


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

    log("=== Efficiency Ratio di Kaufman — predice la qualita del trade? ===\n")

    all_rows = []

    for symbol in ("DAX", "FTSE100"):
        log(f"{'='*70}\n{symbol}\n{'='*70}")
        df = fetch_symbol_full(symbol, account_id, token)
        log(f"  {len(df)} barre caricate.")

        abs_move = df["close"].diff().abs()

        direction = pd.Series(np.where(df["ema20"] > df["ema50"], "long", "short"), index=df.index)
        trend_ampio_ok = np.where(direction == "long", df["ema100"] > df["ema200"], df["ema100"] < df["ema200"])
        if symbol == "DAX":
            dist_r = np.where(direction == "long",
                               (df["close"] - df["rolling_high_20"]) / df["atr"],
                               (df["rolling_low_20"] - df["close"]) / df["atr"])
        else:
            dist_r = np.where(direction == "long",
                               (df["close"] - df["rolling_high_40"]) / df["atr"],
                               (df["rolling_low_40"] - df["close"]) / df["atr"])

        mask = (df["adx"] > 20) & (dist_r >= 0) & trend_ampio_ok
        positions = df.index[mask].tolist()
        log(f"  Trade nel contesto V6 vero: {len(positions)}\n")

        for window in ER_WINDOWS:
            net_move = df["close"].diff(window).abs()
            sum_move = abs_move.rolling(window).sum()
            er = (net_move / sum_move).values

            log(f"  --- Finestra ER = {window} barre ---")
            esiti = {"basso (ER<0.3)": {"TARGET": 0, "STOP": 0}, "medio (0.3-0.5)": {"TARGET": 0, "STOP": 0},
                     "alto (ER>0.5)": {"TARGET": 0, "STOP": 0}}
            for pos in positions:
                if pos < window or np.isnan(er[pos]):
                    continue
                val = er[pos]
                cat = "basso (ER<0.3)" if val < 0.3 else ("alto (ER>0.5)" if val > 0.5 else "medio (0.3-0.5)")
                esito = simulate_outcome(df, pos, direction.iloc[pos])
                if esito in ("TARGET", "STOP"):
                    esiti[cat][esito] += 1

            for cat in ("basso (ER<0.3)", "medio (0.3-0.5)", "alto (ER>0.5)"):
                t, s = esiti[cat]["TARGET"], esiti[cat]["STOP"]
                n = t + s
                if n == 0:
                    log(f"    {cat:<16}: campione vuoto")
                    continue
                wr = t / n * 100
                se = se_of_r_total(t, s)
                r_total = 2 * t - s
                log(f"    {cat:<16} (n={n:>5}): win rate={wr:.2f}%  R totale={r_total:+.1f}  (errore atteso +/-{se:.1f})")
                all_rows.append({
                    "symbol": symbol, "window": window, "categoria": cat,
                    "n_target": t, "n_stop": s, "win_rate": wr, "r_total": r_total, "se": se,
                })
            log("")

    log("=" * 70)
    log("RIEPILOGO — ER alto batte ER basso, a ogni finestra?")
    log("=" * 70)
    df_all = pd.DataFrame(all_rows)
    for symbol in ("DAX", "FTSE100"):
        for window in ER_WINDOWS:
            sub = df_all[(df_all["symbol"] == symbol) & (df_all["window"] == window)]
            wr_basso = sub[sub["categoria"] == "basso (ER<0.3)"]["win_rate"].values
            wr_alto = sub[sub["categoria"] == "alto (ER>0.5)"]["win_rate"].values
            if len(wr_basso) and len(wr_alto):
                log(f"  {symbol} finestra={window}: alto - basso = {wr_alto[0]-wr_basso[0]:+.2f}pt")

    df_all.to_csv("results/efficiency_ratio_riepilogo.csv", index=False)
    with open("results/analyze_efficiency_ratio.txt", "w") as f:
        f.write("\n".join(log_lines))

    print("\n=== Completato. ===")


if __name__ == "__main__":
    main()
