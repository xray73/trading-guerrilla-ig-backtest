"""
analyze_vix_term_sizing_traintest.py — Test train/test rigoroso
(19/07/2026): la modulazione della size in base a contango/
backwardation (VIX vs VIX3M) regge FUORI CAMPIONE? Stessa identica
metodologia del test già fatto sul livello VIX (calmo/medio/panico) —
quel test non aveva mostrato un segnale distinguibile dal rumore;
questo segnale (backwardation) è risultato molto più forte e
significativo nell'esplorazione (z~2.97 su DAX ADX>20, differenza
6-11 punti percentuali a ogni soglia ADX, sempre nella stessa
direzione) — merita lo stesso controllo rigoroso prima di credergli.

METODOLOGIA — identica al test precedente sul livello VIX:
  TRAIN = 2015-01-01 -> 2024-01-01
  TEST  = 2024-01-01 -> 2026-07-19 (mai visto durante la calibrazione)

REGOLA DI CALIBRAZIONE, fissata PRIMA di vedere i risultati:
  moltiplicatore_regime = win_rate_regime(train) / win_rate_medio(train)
  clip tra 0.5x e 1.5x

Popolazione: contesto V6 vero (ADX>20 + breakout reale + trend ampio
confermato). Stessa metodologia stop/target reale di tutti i test di
oggi (1.5xATR stop, 3xATR target, max 48 barre).

VERDETTO: la size modulata (pesi calibrati SOLO su train) batte la
size flat SUL TEST (2024-2026)? Riportato anche l'errore standard
atteso, per giudicare la significatività senza bisogno di un
controllo successivo.

Output SOLO aggregato — mai trade singoli elencati.

Dati: adx_diagnostic_raw (D1) + VIX e VIX3M storico giornaliero
(Yahoo Finance). Nessuna scrittura su D1.
"""

from __future__ import annotations

import os
import time
import math
import pandas as pd
import numpy as np
import requests
import yfinance as yf

DATABASE_ID = "b9fbd4d6-7837-4d86-9c0f-ca60c0cf69e3"
API_BASE = "https://api.cloudflare.com/client/v4/accounts"
CHUNK_SIZE = 5000

ATR_STOP_MULT = 1.5
ATR_TARGET_MULT = 3.0
MAX_HOLD = 48

TRAIN_END = "2024-01-01"
MULT_MIN, MULT_MAX = 0.5, 1.5


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


def get_v6_context_positions(df: pd.DataFrame, symbol: str) -> tuple[list[int], pd.Series]:
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
    return df.index[mask].tolist(), direction


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

    log("=== Modulazione size per contango/backwardation — TRAIN 2015-2023 / TEST 2024-2026 ===\n")

    log("Scarico VIX e VIX3M (Yahoo Finance)...")
    vix = yf.download("^VIX", start="2014-10-01", end="2026-07-19", progress=False)
    vix3m = yf.download("^VIX3M", start="2014-10-01", end="2026-07-19", progress=False)
    for df_ in (vix, vix3m):
        if isinstance(df_.columns, pd.MultiIndex):
            df_.columns = df_.columns.get_level_values(0)
    vix = vix.reset_index()[["Date", "Close"]].rename(columns={"Close": "vix"})
    vix3m = vix3m.reset_index()[["Date", "Close"]].rename(columns={"Close": "vix3m"})
    vix["Date"] = pd.to_datetime(vix["Date"]).dt.tz_localize(None).dt.normalize()
    vix3m["Date"] = pd.to_datetime(vix3m["Date"]).dt.tz_localize(None).dt.normalize()
    term = pd.merge(vix, vix3m, on="Date", how="inner")
    term["backwardation"] = term["vix"] > term["vix3m"]
    term_by_date = term.set_index("Date")["backwardation"]
    log(f"  {len(term)} giorni disponibili, backwardation {term['backwardation'].mean()*100:.1f}% del tempo.\n")

    riepilogo = []

    for symbol in ("DAX", "FTSE100"):
        log(f"{'='*70}\n{symbol}\n{'='*70}")
        df = fetch_symbol_full(symbol, account_id, token)
        log(f"  {len(df)} barre caricate.")

        positions, direction = get_v6_context_positions(df, symbol)
        log(f"  Trade nel contesto V6 vero: {len(positions)}")

        entry_dates = df["timestamp"].dt.tz_localize(None).dt.normalize()
        unique_dates = pd.Series(entry_dates.unique()).tolist()
        date_to_regime = {}
        for d in unique_dates:
            d = pd.Timestamp(d)
            regime = None
            for delta in range(0, 6):
                check_date = d - pd.Timedelta(days=delta)
                if check_date in term_by_date.index:
                    regime = "backwardation" if bool(term_by_date.loc[check_date]) else "contango"
                    break
            date_to_regime[d] = regime
        regime_all = entry_dates.apply(lambda d: date_to_regime.get(pd.Timestamp(d)))

        log("  Simulo esiti reali per ogni trade del contesto...")
        rows = []
        for pos in positions:
            regime = regime_all.iloc[pos]
            if regime is None or (isinstance(regime, float) and pd.isna(regime)):
                continue
            esito = simulate_outcome(df, pos, direction.iloc[pos])
            rows.append({
                "timestamp": df.iloc[pos]["timestamp"],
                "regime": regime,
                "esito": esito,
            })
        trades = pd.DataFrame(rows)
        trades["r_flat"] = trades["esito"].map({"TARGET": 2.0, "STOP": -1.0, "TIMEOUT": 0.0})

        train = trades[trades["timestamp"] < pd.Timestamp(TRAIN_END, tz="UTC")]
        test = trades[trades["timestamp"] >= pd.Timestamp(TRAIN_END, tz="UTC")]
        log(f"  Train (2015-2023): {len(train)} trade  |  Test (2024-2026): {len(test)} trade\n")

        log("  --- Calibrazione moltiplicatori (SOLO train) ---")
        n_t_train = (train["esito"] == "TARGET").sum()
        n_s_train = (train["esito"] == "STOP").sum()
        wr_train_overall = n_t_train / (n_t_train + n_s_train)
        moltiplicatori = {}
        for regime in ("contango", "backwardation"):
            sub = train[train["regime"] == regime]
            n_t = (sub["esito"] == "TARGET").sum()
            n_s = (sub["esito"] == "STOP").sum()
            wr = n_t / (n_t + n_s) if (n_t + n_s) > 0 else wr_train_overall
            mult = np.clip(wr / wr_train_overall, MULT_MIN, MULT_MAX)
            moltiplicatori[regime] = mult
            log(f"    {regime:<13}: win rate train={wr*100:.2f}% (n={n_t+n_s})  "
                f"(medio train={wr_train_overall*100:.2f}%)  moltiplicatore={mult:.3f}")

        log("\n  --- Applicazione sul TEST (2024-2026, mai visto) ---")
        test = test.copy()
        test["mult"] = test["regime"].map(moltiplicatori)
        test["r_modulato"] = test["r_flat"] * test["mult"]

        r_flat_test = test["r_flat"].sum()
        r_modulato_test = test["r_modulato"].sum()
        diff = r_modulato_test - r_flat_test
        uplift_pct = diff / abs(r_flat_test) * 100 if r_flat_test != 0 else float("nan")

        n_t_test = (test["esito"] == "TARGET").sum()
        n_s_test = (test["esito"] == "STOP").sum()
        se_test = se_of_r_total(n_t_test, n_s_test)

        log(f"    R totali FLAT (size sempre 1x):      {r_flat_test:+.1f}R")
        log(f"    R totali MODULATO (pesi da train):   {r_modulato_test:+.1f}R")
        log(f"    Differenza: {diff:+.1f}R ({uplift_pct:+.1f}%)")
        log(f"    Errore standard atteso per rumore sul totale test: +/-{se_test:.1f}R")
        log(f"    >>> Differenza {'oltre 1 SE, potenzialmente notabile' if abs(diff) > se_test else 'ben dentro il rumore atteso'}\n")

        riepilogo.append({
            "symbol": symbol, "n_train": len(train), "n_test": len(test),
            "mult_contango": moltiplicatori["contango"], "mult_backwardation": moltiplicatori["backwardation"],
            "r_flat_test": r_flat_test, "r_modulato_test": r_modulato_test,
            "diff_r": diff, "uplift_pct": uplift_pct, "se_test": se_test,
        })

    log("=" * 70)
    log("VERDETTO FINALE")
    log("=" * 70)
    for row in riepilogo:
        log(f"  {row['symbol']}: differenza {row['diff_r']:+.1f}R su test "
            f"(errore atteso +/-{row['se_test']:.1f}R) — "
            f"{'sopra il rumore' if abs(row['diff_r']) > row['se_test'] else 'dentro il rumore'}")

    pd.DataFrame(riepilogo).to_csv("results/vix_term_sizing_traintest_riepilogo.csv", index=False)
    with open("results/analyze_vix_term_sizing_traintest.txt", "w") as f:
        f.write("\n".join(log_lines))

    print("\n=== Completato. ===")


if __name__ == "__main__":
    main()
