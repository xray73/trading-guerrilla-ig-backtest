"""
analyze_dax_ftse_agreement.py — Test (19/07/2026): quando DAX e
FTSE100 concordano sulla direzione (EMA20 vs EMA50) nello stesso
momento, il segnale di ciascuno è più affidabile di quando divergono?

Idea: un trend confermato da un secondo indice europeo correlato nello
stesso istante potrebbe riflettere un movimento di mercato genuino
(macro/settoriale europeo), non rumore isolato di un singolo indice —
stessa logica concettuale del VIX (conferma da un contesto più ampio),
ma qui usando un asset che già tradate, zero dati esterni nuovi.

METODOLOGIA (lezioni della sessione precedente applicate fin
dall'inizio, non aggiunte dopo):
  - Meccanismo di trading REALE (stop 1.5xATR, target 3xATR, max 48
    barre) fin dal primo test, non persistenza a punto fisso.
  - Popolazione: contesto V6 vero (ADX>20 + breakout reale + trend
    ampio confermato) per lo strumento "primario" in ciascun confronto.
  - Robustezza verificata su 5 soglie ADX (20/25/30/35/40).
  - Errore standard atteso calcolato e riportato per ogni cella.

Per ogni trade DAX: guarda la direzione di FTSE100 (EMA20 vs EMA50,
NESSUN filtro di contesto richiesto sull'altro strumento — solo la
sua inclinazione direzionale in quel momento) allo STESSO timestamp.
Concorde = stessa direzione. Discorde = direzione opposta. Ripetuto
in modo simmetrico per FTSE100 guardando DAX.

Output SOLO aggregato — mai trade singoli elencati.

Dati: adx_diagnostic_raw (D1, entrambi gli strumenti). Nessuna
scrittura su D1, nessun dato esterno.
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
ADX_THRESHOLDS = [20, 25, 30, 35, 40]


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


def get_dist_r(df: pd.DataFrame, symbol: str, direction: np.ndarray) -> np.ndarray:
    if symbol == "DAX":
        return np.where(direction == "long",
                         (df["close"] - df["rolling_high_20"]) / df["atr"],
                         (df["rolling_low_20"] - df["close"]) / df["atr"])
    return np.where(direction == "long",
                     (df["close"] - df["rolling_high_40"]) / df["atr"],
                     (df["rolling_low_40"] - df["close"]) / df["atr"])


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

    log("=== Accordo DAX-FTSE100 — il segnale è più affidabile quando concordano? ===\n")

    log("Scarico DAX e FTSE100 da D1...")
    dax = fetch_symbol_full("DAX", account_id, token)
    ftse = fetch_symbol_full("FTSE100", account_id, token)
    log(f"  DAX: {len(dax)} barre  |  FTSE100: {len(ftse)} barre")

    dax["direction"] = np.where(dax["ema20"] > dax["ema50"], "long", "short")
    ftse["direction"] = np.where(ftse["ema20"] > ftse["ema50"], "long", "short")

    # unione per timestamp esatto, per sapere la direzione dell'ALTRO strumento in ogni istante
    merged = pd.merge(
        dax[["timestamp", "direction"]].rename(columns={"direction": "dax_dir"}),
        ftse[["timestamp", "direction"]].rename(columns={"direction": "ftse_dir"}),
        on="timestamp", how="inner"
    )
    log(f"  Timestamp allineati tra i due strumenti: {len(merged)}\n")
    other_dir_by_ts = merged.set_index("timestamp")

    all_rows = []

    for primary_name, primary_df, other_col in (("DAX", dax, "ftse_dir"), ("FTSE100", ftse, "dax_dir")):
        log(f"{'='*70}\n{primary_name} (confronto con l'altro strumento)\n{'='*70}")

        primary_df = primary_df.copy()
        primary_df["other_dir"] = primary_df["timestamp"].map(other_dir_by_ts[other_col])
        primary_df["concorde"] = primary_df["direction"] == primary_df["other_dir"]

        trend_ampio_ok = np.where(primary_df["direction"] == "long",
                                   primary_df["ema100"] > primary_df["ema200"],
                                   primary_df["ema100"] < primary_df["ema200"])
        dist_r = get_dist_r(primary_df, primary_name, primary_df["direction"].values)

        for adx_th in ADX_THRESHOLDS:
            mask = (primary_df["adx"] > adx_th) & (dist_r >= 0) & trend_ampio_ok
            positions = primary_df.index[mask].tolist()

            esiti = {"concorde": {"TARGET": 0, "STOP": 0}, "discorde": {"TARGET": 0, "STOP": 0}}
            for pos in positions:
                other_dir = primary_df.iloc[pos]["other_dir"]
                if pd.isna(other_dir):
                    continue
                cat = "concorde" if primary_df.iloc[pos]["concorde"] else "discorde"
                esito = simulate_outcome(primary_df, pos, primary_df.iloc[pos]["direction"])
                if esito in ("TARGET", "STOP"):
                    esiti[cat][esito] += 1

            log(f"\n  ADX > {adx_th}:")
            for cat in ("concorde", "discorde"):
                t, s = esiti[cat]["TARGET"], esiti[cat]["STOP"]
                n = t + s
                if n == 0:
                    log(f"    {cat:<10}: campione vuoto")
                    continue
                wr = t / n * 100
                r_total = 2 * t - s
                se = se_of_r_total(t, s)
                log(f"    {cat:<10} (n={n:>5}): win rate={wr:.2f}%  R totale={r_total:+.1f}  "
                    f"(errore atteso +/-{se:.1f})")
                all_rows.append({
                    "primary": primary_name, "adx_threshold": adx_th, "categoria": cat,
                    "n_target": t, "n_stop": s, "win_rate": wr, "r_total": r_total, "se": se,
                })
        log("")

    log("=" * 70)
    log("RIEPILOGO — concorde batte discorde, a ogni soglia?")
    log("=" * 70)
    df_all = pd.DataFrame(all_rows)
    for primary_name in ("DAX", "FTSE100"):
        for adx_th in ADX_THRESHOLDS:
            sub = df_all[(df_all["primary"] == primary_name) & (df_all["adx_threshold"] == adx_th)]
            if len(sub) < 2:
                continue
            wr_c = sub[sub["categoria"] == "concorde"]["win_rate"].values
            wr_d = sub[sub["categoria"] == "discorde"]["win_rate"].values
            if len(wr_c) and len(wr_d):
                log(f"  {primary_name} ADX>{adx_th}: concorde - discorde = {wr_c[0]-wr_d[0]:+.2f}pt")

    df_all.to_csv("results/dax_ftse_agreement_riepilogo.csv", index=False)
    with open("results/analyze_dax_ftse_agreement.txt", "w") as f:
        f.write("\n".join(log_lines))

    print("\n=== Completato. ===")


if __name__ == "__main__":
    main()
