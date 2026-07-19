"""
analyze_fold2_2021_deepdive.py — Approfondimento (19/07/2026) sul DAX
2021, l'unico anno negativo trovato nell'indagine fold2
(analyze_fold2_investigation.py: Q4 vs resto = -3.73pt, AUC=0.505
quasi casuale, contro +0.97pt/2022 e +10.85pt/2023).

Riusa lo STESSO modello allenato su 2015-2021 (stesso identico codice
di training/feature/simulazione esiti di analyze_fold2_investigation.py
— nessuna ricalibrazione) e isola solo il 2021 DAX per capire COSA
distingue i trade che il modello classifica Q4 (score più alto) dal
resto, in un anno dove il modello non discrimina meglio del caso.

Aggiunge due cose rispetto allo script originale:
1. I coefficienti del modello (quali feature spingono un trade verso
   Q4, in generale — non specifico al 2021).
2. Le medie delle 7 feature per il gruppo Q4 vs il gruppo resto,
   SOLO nel 2021 — per vedere se i trade selezionati come "migliori"
   quell'anno hanno un profilo diverso da un anno buono (2023,
   incluso come controllo).

Output: SOLO aggregati (medie/mediane per gruppo, coefficienti) — MAI
trade individuali, coerente con Regole_Backtest_MonteCarlo.md sez.2-3.

Analisi ESPLORATIVA, nessun criterio di successo/fallimento fissato —
accordo esplicito con l'utente per questo filone (VIX/logistico/
walk-forward/fold2).

Dati: adx_diagnostic_raw (D1, sola lettura) + VIX/VIX3M storico
(Yahoo Finance). Nessuna scrittura su D1. Nessuna modifica a engine.py
o a live_execute.py — puro script di analisi, motore live invariato.
"""

from __future__ import annotations

import os
import time
import pandas as pd
import numpy as np
import requests
import yfinance as yf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

DATABASE_ID = "b9fbd4d6-7837-4d86-9c0f-ca60c0cf69e3"
API_BASE = "https://api.cloudflare.com/client/v4/accounts"
CHUNK_SIZE = 5000

ATR_STOP_MULT = 1.5
ATR_TARGET_MULT = 3.0
MAX_HOLD = 48
ER_WINDOW = 40
FEATURES = ["adx", "atr_pct", "ema_spread_atr", "dist_ema200_atr", "vix", "vix_term_spread", "efficiency_ratio"]

TRAIN_START, TRAIN_END = "2015-01-01", "2021-01-01"
SYMBOL = "DAX"


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


def build_dataset(symbol: str, df: pd.DataFrame, term_by_date) -> pd.DataFrame:
    abs_move = df["close"].diff().abs()
    net_move = df["close"].diff(ER_WINDOW).abs()
    sum_move = abs_move.rolling(ER_WINDOW).sum()
    er_series = (net_move / sum_move).values

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
    entry_dates = df["timestamp"].dt.tz_localize(None).dt.normalize()

    rows = []
    for pos in positions:
        if pos < ER_WINDOW or np.isnan(er_series[pos]):
            continue
        d = entry_dates.iloc[pos]
        vix_val, term_val = None, None
        for delta in range(0, 6):
            check_date = d - pd.Timedelta(days=delta)
            if check_date in term_by_date.index:
                vix_val = term_by_date.loc[check_date, "vix"]
                term_val = term_by_date.loc[check_date, "vix_term_spread"]
                break
        if vix_val is None:
            continue

        row = df.iloc[pos]
        atr_pct = row["atr"] / row["close"] * 100
        ema_spread_atr = abs(row["ema20"] - row["ema50"]) / row["atr"]
        dist_ema200_atr = abs(row["close"] - row["ema200"]) / row["atr"]

        esito = simulate_outcome(df, pos, direction.iloc[pos])
        if esito not in ("TARGET", "STOP"):
            continue

        rows.append({
            "timestamp": row["timestamp"], "direction": direction.iloc[pos],
            "adx": row["adx"], "atr_pct": atr_pct,
            "ema_spread_atr": ema_spread_atr, "dist_ema200_atr": dist_ema200_atr,
            "vix": vix_val, "vix_term_spread": term_val, "efficiency_ratio": er_series[pos],
            "target": 1 if esito == "TARGET" else 0,
        })
    return pd.DataFrame(rows)


def describe_group(df_feat: pd.DataFrame, log):
    n_long = (df_feat["direction"] == "long").sum()
    n_short = (df_feat["direction"] == "short").sum()
    log(f"  n={len(df_feat)}  win_rate={df_feat['target'].mean()*100:.1f}%  long={n_long}  short={n_short}")
    for feat in FEATURES:
        log(f"    {feat:18s} media={df_feat[feat].mean():+.3f}  mediana={df_feat[feat].median():+.3f}")


def score_year(data: pd.DataFrame, raw_dummy, scaler, model, y_start: str, y_end: str, label: str, log):
    test_y = data[(data["timestamp"] >= pd.Timestamp(y_start, tz="UTC")) &
                  (data["timestamp"] < pd.Timestamp(y_end, tz="UTC"))]
    if len(test_y) < 20:
        log(f"  {label}: campione troppo piccolo ({len(test_y)}), salto.")
        return None
    X_test_y = scaler.transform(test_y[FEATURES])
    proba_y = model.predict_proba(X_test_y)[:, 1]
    auc_y = roc_auc_score(test_y["target"].values, proba_y) if test_y["target"].nunique() > 1 else float("nan")

    test_y2 = test_y.copy()
    test_y2["proba"] = proba_y
    q4_cut = test_y2["proba"].quantile(0.75)
    q4 = test_y2[test_y2["proba"] >= q4_cut]
    rest = test_y2[test_y2["proba"] < q4_cut]

    log(f"{label}: n={len(test_y)}  AUC={auc_y:.3f}  cutoff proba Q4={q4_cut:.3f}\n")
    log(f"--- Gruppo Q4 (score piu alto, n={len(q4)}) ---")
    describe_group(q4, log)
    log("")
    log(f"--- Gruppo resto (n={len(rest)}) ---")
    describe_group(rest, log)
    log("")
    log(f"Differenza win rate Q4-resto: {(q4['target'].mean()-rest['target'].mean())*100:+.2f}pt")
    return q4, rest


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

    log("=== Approfondimento DAX 2021 — cosa distingue Q4 dal resto quando il modello non discrimina ===\n")

    log("Scarico VIX e VIX3M...")
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
    term["vix_term_spread"] = term["vix"] - term["vix3m"]
    term_by_date = term.set_index("Date")
    log("")

    log(f"{'='*70}\n{SYMBOL}\n{'='*70}")
    raw = fetch_symbol_full(SYMBOL, account_id, token)
    data = build_dataset(SYMBOL, raw, term_by_date)

    train = data[(data["timestamp"] >= pd.Timestamp(TRAIN_START, tz="UTC")) &
                 (data["timestamp"] < pd.Timestamp(TRAIN_END, tz="UTC"))]
    log(f"Train (2015-2021): {len(train)} trade\n")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[FEATURES])
    model = LogisticRegression(max_iter=1000, C=1.0)
    model.fit(X_train, train["target"].values)

    log("Coefficienti del modello (feature standardizzate — peso/segno nello score, generale, non specifico al 2021):")
    for feat, coef in sorted(zip(FEATURES, model.coef_[0]), key=lambda x: -abs(x[1])):
        log(f"    {feat:18s} {coef:+.3f}")
    log("")

    log("##### 2021 (anno anomalo) #####")
    score_year(data, raw, scaler, model, "2021-01-01", "2022-01-01", "2021", log)
    log("")

    log("##### Controllo: 2023 (anno buono, +10.85pt nell'indagine originale) #####")
    score_year(data, raw, scaler, model, "2023-01-01", "2024-01-01", "2023", log)

    with open("results/analyze_fold2_2021_deepdive.txt", "w") as f:
        f.write("\n".join(log_lines))

    print("\n=== Completato. ===")


if __name__ == "__main__":
    main()
