"""
analyze_logistic_score.py — Test (19/07/2026): un modello statistico
vincolato (regressione logistica, poche variabili) può combinare più
parametri insieme (ADX, ATR%, spread EMA, distanza EMA200, VIX,
struttura a termine VIX) per predire l'esito di un trade meglio del
caso, IN MODO CHE REGGA FUORI CAMPIONE?

METODOLOGIA — principio "parti semplice, aumenta complessità solo se
guadagnata sui dati mai visti":
  - Regressione logistica (pochi gradi di libertà, non un albero o una
    curva libera) — il modello con MENO capacità di inseguire il
    rumore tra quelli ragionevoli.
  - Due modelli separati (DAX, FTSE100) — coerente con l'asimmetria
    già scoperta tra i due strumenti in questa sessione.
  - TRAIN = 2015-01-01 -> 2024-01-01 (fit del modello)
  - TEST  = 2024-01-01 -> 2026-07-19 (MAI visto durante il fit,
    stesso confine già usato per tutti i test train/test di oggi)
  - Feature: solo informazioni disponibili PRIMA dell'ingresso (mai
    "come si è chiuso il trade" — quello è l'etichetta da prevedere,
    non un ingrediente, per non usare informazioni dal futuro).

Target: esito reale (TARGET=1, STOP=0) con meccanismo stop/target
fisso (1.5xATR stop, 3xATR target, max 48 barre) — stesso identico
meccanismo di tutti i test della sessione.

Confronto train vs test: se il modello discrimina bene in train ma
non in test, è overfitting (stesso pattern già visto col filtro
maturità trend in RCA sez.17) — il segnale più diretto per giudicare
se l'idea ha una base reale o no.

Output SOLO aggregato (coefficienti, AUC, win rate per fascia di
punteggio) — mai trade singoli elencati.

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
TRAIN_END = "2024-01-01"

FEATURES = ["adx", "atr_pct", "ema_spread_atr", "dist_ema200_atr", "vix", "vix_term_spread"]


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

    log("=== Regressione logistica multi-parametro — TRAIN 2015-2023 / TEST 2024-2026 ===")
    log(f"    Feature usate: {FEATURES}\n")

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
    term["vix_term_spread"] = term["vix"] - term["vix3m"]
    term_by_date = term.set_index("Date")

    riepilogo = []

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

        mask = (df["adx"] > 20) & (dist_r >= 0) & trend_ampio_ok
        positions = df.index[mask].tolist()
        log(f"  Trade nel contesto V6 vero: {len(positions)}")

        entry_dates = df["timestamp"].dt.tz_localize(None).dt.normalize()

        log("  Costruisco feature + esiti reali per ogni trade...")
        rows = []
        for pos in positions:
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
                "vix": vix_val, "vix_term_spread": term_val,
                "target": 1 if esito == "TARGET" else 0,
            })

        data = pd.DataFrame(rows)
        train = data[data["timestamp"] < pd.Timestamp(TRAIN_END, tz="UTC")].reset_index(drop=True)
        test = data[data["timestamp"] >= pd.Timestamp(TRAIN_END, tz="UTC")].reset_index(drop=True)
        log(f"  Train: {len(train)} trade (win rate {train['target'].mean()*100:.2f}%)  |  "
            f"Test: {len(test)} trade (win rate {test['target'].mean()*100:.2f}%)\n")

        scaler = StandardScaler()
        X_train = scaler.fit_transform(train[FEATURES])
        y_train = train["target"].values
        X_test = scaler.transform(test[FEATURES])
        y_test = test["target"].values

        model = LogisticRegression(max_iter=1000, C=1.0)
        model.fit(X_train, y_train)

        log("  --- Coefficienti (standardizzati, |valore| = importanza relativa) ---")
        for feat, coef in sorted(zip(FEATURES, model.coef_[0]), key=lambda x: -abs(x[1])):
            log(f"    {feat:<18}: {coef:+.3f}")

        train_proba = model.predict_proba(X_train)[:, 1]
        test_proba = model.predict_proba(X_test)[:, 1]
        auc_train = roc_auc_score(y_train, train_proba)
        auc_test = roc_auc_score(y_test, test_proba)
        log(f"\n  AUC-ROC train: {auc_train:.3f}  (0.5 = caso puro, 1.0 = perfetto)")
        log(f"  AUC-ROC test:  {auc_test:.3f}  <-- questo è il numero che conta davvero")
        log(f"  Differenza train-test: {auc_train-auc_test:+.3f} "
            f"({'GAP AMPIO, possibile overfitting' if (auc_train-auc_test) > 0.05 else 'gap contenuto'})")

        log("\n  --- Win rate reale sul TEST, per quartile di punteggio predetto ---")
        test = test.copy()
        test["proba"] = test_proba
        test["quartile"] = pd.qcut(test["proba"], 4, labels=["Q1 (più basso)", "Q2", "Q3", "Q4 (più alto)"])
        for q in ["Q1 (più basso)", "Q2", "Q3", "Q4 (più alto)"]:
            sub = test[test["quartile"] == q]
            wr = sub["target"].mean() * 100
            log(f"    {q}: n={len(sub)}  win rate reale={wr:.2f}%  (punteggio medio predetto: {sub['proba'].mean()*100:.1f}%)")

        riepilogo.append({
            "symbol": symbol, "n_train": len(train), "n_test": len(test),
            "auc_train": auc_train, "auc_test": auc_test,
        })
        log("")

    log("=" * 70)
    log("VERDETTO FINALE")
    log("=" * 70)
    for row in riepilogo:
        segnale_reale = row["auc_test"] > 0.55
        log(f"  {row['symbol']}: AUC test={row['auc_test']:.3f} — "
            f"{'possibile segnale reale (>0.55)' if segnale_reale else 'non distinguibile dal caso (<=0.55)'}")

    pd.DataFrame(riepilogo).to_csv("results/logistic_score_riepilogo.csv", index=False)
    with open("results/analyze_logistic_score.txt", "w") as f:
        f.write("\n".join(log_lines))

    print("\n=== Completato. ===")


if __name__ == "__main__":
    main()
