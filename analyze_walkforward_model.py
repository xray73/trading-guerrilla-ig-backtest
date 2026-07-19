"""
analyze_walkforward_model.py — Estensione (19/07/2026) del modello
logistico: aggiunge l'Efficiency Ratio (finestra 40, dove il DAX
mostrava il pattern più sistematico) alle feature esistenti, e
verifica il segnale Q4 con WALK-FORWARD su 3 finestre storiche
indipendenti invece di un solo split train/test — se il segnale è
vero, deve ripresentarsi su periodi diversi, non solo su quello già
osservato (2024-2026).

FOLD (espansione progressiva, ciascuno allena su tutto lo storico
disponibile FINO a quel punto, testa sul periodo successivo MAI
visto):
  Fold 1: train 2015-2017 -> test 2018-2020
  Fold 2: train 2015-2020 -> test 2021-2023
  Fold 3: train 2015-2023 -> test 2024-2026 (il fold già visto ieri)

Feature: adx, atr_pct, ema_spread_atr, dist_ema200_atr, vix,
vix_term_spread, efficiency_ratio_40 (NUOVA).

Stessa metodologia di base: regressione logistica vincolata,
meccanismo stop/target reale, popolazione contesto V6 vero.

Output SOLO aggregato — mai trade singoli elencati.

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

FOLDS = [
    ("2015-01-01", "2018-01-01", "2020-01-01"),
    ("2015-01-01", "2021-01-01", "2024-01-01"),
    ("2015-01-01", "2024-01-01", "2026-07-19"),
]


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


def build_dataset(symbol: str, account_id: str, token: str, term_by_date, log) -> pd.DataFrame:
    df = fetch_symbol_full(symbol, account_id, token)
    log(f"  {len(df)} barre caricate.")

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
            "timestamp": row["timestamp"], "adx": row["adx"], "atr_pct": atr_pct,
            "ema_spread_atr": ema_spread_atr, "dist_ema200_atr": dist_ema200_atr,
            "vix": vix_val, "vix_term_spread": term_val, "efficiency_ratio": er_series[pos],
            "target": 1 if esito == "TARGET" else 0,
        })
    return pd.DataFrame(rows)


def q4_vs_rest(test_df: pd.DataFrame, proba: np.ndarray):
    test_df = test_df.copy()
    test_df["proba"] = proba
    q4_cut = test_df["proba"].quantile(0.75)
    q4 = test_df[test_df["proba"] >= q4_cut]
    rest = test_df[test_df["proba"] < q4_cut]
    if len(q4) == 0 or len(rest) == 0:
        return float("nan"), 0, 0
    return q4["target"].mean() - rest["target"].mean(), len(q4), len(rest)


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

    log("=== Modello esteso (+ Efficiency Ratio) — walk-forward su 3 finestre ===")
    log(f"    Feature: {FEATURES}\n")

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

    riepilogo = []

    for symbol in ("DAX", "FTSE100"):
        log(f"{'='*70}\n{symbol}\n{'='*70}")
        data = build_dataset(symbol, account_id, token, term_by_date, log)
        log(f"  Dataset totale: {len(data)} trade\n")

        for i, (train_start, split_date, test_end) in enumerate(FOLDS, 1):
            train = data[(data["timestamp"] >= pd.Timestamp(train_start, tz="UTC")) &
                         (data["timestamp"] < pd.Timestamp(split_date, tz="UTC"))]
            test = data[(data["timestamp"] >= pd.Timestamp(split_date, tz="UTC")) &
                        (data["timestamp"] < pd.Timestamp(test_end, tz="UTC"))]

            if len(train) < 100 or len(test) < 40:
                log(f"  Fold {i} ({train_start} a {split_date} | test fino a {test_end}): campione insufficiente, salto.")
                continue

            scaler = StandardScaler()
            X_train = scaler.fit_transform(train[FEATURES])
            X_test = scaler.transform(test[FEATURES])
            model = LogisticRegression(max_iter=1000, C=1.0)
            model.fit(X_train, train["target"].values)
            proba_test = model.predict_proba(X_test)[:, 1]

            diff, n_q4, n_rest = q4_vs_rest(test, proba_test)
            auc = roc_auc_score(test["target"].values, proba_test) if test["target"].nunique() > 1 else float("nan")

            log(f"  Fold {i}: train [{train_start} -> {split_date}) n={len(train)}, "
                f"test [{split_date} -> {test_end}) n={len(test)}")
            log(f"    AUC test: {auc:.3f}  |  Q4 vs resto: {diff*100:+.2f}pt (n_Q4={n_q4}, n_resto={n_rest})")

            riepilogo.append({
                "symbol": symbol, "fold": i, "train_start": train_start, "split": split_date,
                "test_end": test_end, "n_train": len(train), "n_test": len(test),
                "auc_test": auc, "q4_diff": diff,
            })
        log("")

    log("=" * 70)
    log("RIEPILOGO — il segnale Q4 si ripete su tutti i fold, o solo su quello gia noto?")
    log("=" * 70)
    df_r = pd.DataFrame(riepilogo)
    for symbol in ("DAX", "FTSE100"):
        sub = df_r[df_r["symbol"] == symbol]
        log(f"  {symbol}:")
        for _, row in sub.iterrows():
            log(f"    Fold {row['fold']}: Q4 diff={row['q4_diff']*100:+.2f}pt, AUC={row['auc_test']:.3f}")
        positivi = (sub["q4_diff"] > 0).sum()
        log(f"    -> Positivo in {positivi}/{len(sub)} fold")

    df_r.to_csv("results/walkforward_model_riepilogo.csv", index=False)
    with open("results/analyze_walkforward_model.txt", "w") as f:
        f.write("\n".join(log_lines))

    print("\n=== Completato. ===")


if __name__ == "__main__":
    main()
