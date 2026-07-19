"""
analyze_vix_term_structure.py — Test struttura a termine VIX
(19/07/2026): contango (VIX < VIX3M, calma strutturale) vs
backwardation (VIX > VIX3M, stress acuto) come regime — diverso
concettualmente dal livello VIX assoluto già testato (qui si guarda
se il mercato si aspetta che lo stress attuale duri o rientri presto,
non solo "quanto stress c'è oggi").

LEZIONE APPLICATA da subito (non aggiunta dopo come ieri): ogni
cella include una stima dell'errore standard atteso per pura casualità
statistica, per giudicare la significatività SENZA bisogno di un
controllo successivo — un risultato "positivo" ma dentro il rumore
atteso viene segnalato esplicitamente come non significativo.

Popolazione: contesto V6 vero (ADX>20 + breakout reale + trend ampio
confermato), stessa metodologia stop/target reale (1.5xATR stop,
3xATR target, max 48 barre) di tutti i test di oggi. Robustezza
verificata su 5 soglie ADX (20/25/30/35/40), stesso principio già
applicato al VIX di livello.

Output SOLO aggregato — mai trade singoli elencati.

Dati: adx_diagnostic_raw (D1) + VIX e VIX3M storico giornaliero
(Yahoo Finance, ^VIX e ^VIX3M). Nessuna scrittura su D1.
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
    """Errore standard atteso del totale R (esiti +2/-1) per puro rumore,
    dato n_target vincite e n_stop perdite (approssimazione binomiale)."""
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

    log("=== Struttura a termine VIX (VIX vs VIX3M) — contango/backwardation ===\n")

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
    log(f"  {len(term)} giorni con VIX+VIX3M disponibili, "
        f"{term['Date'].min().date()} -> {term['Date'].max().date()}")
    log(f"  Giorni in backwardation: {term['backwardation'].mean()*100:.1f}%\n")

    term_by_date = term.set_index("Date")["backwardation"]

    all_rows = []

    for symbol in ("DAX", "FTSE100"):
        log(f"{'='*70}\n{symbol}\n{'='*70}")
        df = fetch_symbol_full(symbol, account_id, token)
        log(f"  {len(df)} barre caricate.")

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

        for adx_th in ADX_THRESHOLDS:
            mask = (df["adx"] > adx_th) & (dist_r >= 0) & trend_ampio_ok
            positions = df.index[mask].tolist()

            esiti = {"contango": {"TARGET": 0, "STOP": 0}, "backwardation": {"TARGET": 0, "STOP": 0}}
            for pos in positions:
                regime = regime_all.iloc[pos]
                if regime is None or (isinstance(regime, float) and pd.isna(regime)):
                    continue
                esito = simulate_outcome(df, pos, direction.iloc[pos])
                if esito in ("TARGET", "STOP"):
                    esiti[regime][esito] += 1

            log(f"\n  ADX > {adx_th}:")
            for regime in ("contango", "backwardation"):
                t, s = esiti[regime]["TARGET"], esiti[regime]["STOP"]
                n = t + s
                if n == 0:
                    log(f"    {regime:<13}: campione vuoto")
                    continue
                wr = t / n * 100
                r_total = 2 * t - s
                se = se_of_r_total(t, s)
                significativo = abs(r_total) > 2 * se if se > 0 else False
                log(f"    {regime:<13} (n={n:>5}): win rate={wr:.2f}%  R totale={r_total:+.1f}  "
                    f"(errore atteso per rumore +/-{se:.1f}, {'oltre 2 SE, plausibilmente reale' if significativo else 'dentro il rumore atteso'})")
                all_rows.append({
                    "symbol": symbol, "adx_threshold": adx_th, "regime": regime,
                    "n_target": t, "n_stop": s, "win_rate": wr, "r_total": r_total,
                    "se_atteso": se, "oltre_2se": significativo,
                })
        log("")

    log("=" * 70)
    log("RIEPILOGO — backwardation batte contango, a ogni soglia?")
    log("=" * 70)
    df_all = pd.DataFrame(all_rows)
    for symbol in ("DAX", "FTSE100"):
        for adx_th in ADX_THRESHOLDS:
            sub = df_all[(df_all["symbol"] == symbol) & (df_all["adx_threshold"] == adx_th)]
            if len(sub) < 2:
                continue
            wr_contango = sub[sub["regime"] == "contango"]["win_rate"].values
            wr_back = sub[sub["regime"] == "backwardation"]["win_rate"].values
            if len(wr_contango) and len(wr_back):
                diff = wr_back[0] - wr_contango[0]
                log(f"  {symbol} ADX>{adx_th}: backwardation - contango = {diff:+.2f}pt")

    df_all.to_csv("results/vix_term_structure_riepilogo.csv", index=False)
    with open("results/analyze_vix_term_structure.txt", "w") as f:
        f.write("\n".join(log_lines))

    print("\n=== Completato. ===")


if __name__ == "__main__":
    main()
