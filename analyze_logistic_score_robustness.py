"""
analyze_logistic_score_robustness.py — Test di robustezza (19/07/2026)
sul segnale trovato nel quartile 4 (punteggio più alto) del modello
logistico — prima di considerarlo un risultato solido, verifichiamo
se sopravvive a due controlli indipendenti:

  1. BOOTSTRAP: ricampiona il set di test (2024-2026) con
     ripetizione, 1000 volte, ricalcola ogni volta la differenza di
     win rate Q4 vs Q1-Q3. Se l'intervallo di confidenza al 95% non
     include zero, il vantaggio è robusto rispetto al campione
     specifico osservato — non un colpo di fortuna sull'ordine dei
     dati.

  2. ABLATION (rimozione di una feature alla volta): riallena il
     modello 6 volte, ogni volta togliendo UNA feature diversa dalle
     6 originali. Se il segnale Q4 sparisce togliendo una feature
     specifica, il segnale dipende da quel singolo parametro (meno
     interessante, equivalente a un test univariato mascherato). Se
     resta stabile togliendo qualunque singola feature, è davvero la
     combinazione a contare.

Stessa metodologia di base (train 2015-2023, test 2024-2026, stop/
target reale) del test originale. Output SOLO aggregato.

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

DATABASE_ID = "b9fbd4d6-7837-4d86-9c0f-ca60c0cf69e3"
API_BASE = "https://api.cloudflare.com/client/v4/accounts"
CHUNK_SIZE = 5000

ATR_STOP_MULT = 1.5
ATR_TARGET_MULT = 3.0
MAX_HOLD = 48
TRAIN_END = "2024-01-01"
N_BOOTSTRAP = 1000

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


def build_dataset(symbol: str, account_id: str, token: str, term_by_date: pd.DataFrame, log) -> pd.DataFrame:
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
    entry_dates = df["timestamp"].dt.tz_localize(None).dt.normalize()

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
    return pd.DataFrame(rows)


def q4_vs_rest_diff(test_df: pd.DataFrame, proba: np.ndarray) -> float:
    test_df = test_df.copy()
    test_df["proba"] = proba
    q4_cut = test_df["proba"].quantile(0.75)
    q4 = test_df[test_df["proba"] >= q4_cut]
    rest = test_df[test_df["proba"] < q4_cut]
    if len(q4) == 0 or len(rest) == 0:
        return float("nan")
    return q4["target"].mean() - rest["target"].mean()


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

    log("=== Robustezza segnale Q4 — bootstrap + rimozione feature ===\n")

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

    rng = np.random.default_rng(42)
    riepilogo = []

    for symbol in ("DAX", "FTSE100"):
        log(f"{'='*70}\n{symbol}\n{'='*70}")
        data = build_dataset(symbol, account_id, token, term_by_date, log)
        train = data[data["timestamp"] < pd.Timestamp(TRAIN_END, tz="UTC")].reset_index(drop=True)
        test = data[data["timestamp"] >= pd.Timestamp(TRAIN_END, tz="UTC")].reset_index(drop=True)
        log(f"  Train: {len(train)}  |  Test: {len(test)}\n")

        scaler = StandardScaler()
        X_train = scaler.fit_transform(train[FEATURES])
        X_test = scaler.transform(test[FEATURES])
        model = LogisticRegression(max_iter=1000, C=1.0)
        model.fit(X_train, train["target"].values)
        test_proba = model.predict_proba(X_test)[:, 1]

        osservato = q4_vs_rest_diff(test, test_proba)
        log(f"  Differenza osservata Q4 vs resto (tutte le feature): {osservato*100:+.2f}pt\n")

        log(f"  --- Bootstrap ({N_BOOTSTRAP} ricampionamenti del test) ---")
        boot_diffs = []
        test_with_proba = test.copy()
        test_with_proba["proba"] = test_proba
        n = len(test_with_proba)
        for _ in range(N_BOOTSTRAP):
            sample_idx = rng.integers(0, n, n)
            sample = test_with_proba.iloc[sample_idx]
            q4_cut = sample["proba"].quantile(0.75)
            q4 = sample[sample["proba"] >= q4_cut]
            rest = sample[sample["proba"] < q4_cut]
            if len(q4) > 0 and len(rest) > 0:
                boot_diffs.append(q4["target"].mean() - rest["target"].mean())
        boot_diffs = np.array(boot_diffs)
        ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])
        log(f"  Differenza media bootstrap: {boot_diffs.mean()*100:+.2f}pt")
        log(f"  Intervallo di confidenza 95%: [{ci_low*100:+.2f}pt, {ci_high*100:+.2f}pt]")
        include_zero = ci_low <= 0 <= ci_high
        esito_msg = "L'intervallo INCLUDE zero, non possiamo escludere rumore" if include_zero else "L'intervallo NON include zero, vantaggio robusto al ricampionamento"
        log(f"  >>> {esito_msg}\n")

        log("  --- Rimozione di una feature alla volta ---")
        for feat_to_remove in FEATURES:
            remaining = [f for f in FEATURES if f != feat_to_remove]
            scaler_a = StandardScaler()
            X_train_a = scaler_a.fit_transform(train[remaining])
            X_test_a = scaler_a.transform(test[remaining])
            model_a = LogisticRegression(max_iter=1000, C=1.0)
            model_a.fit(X_train_a, train["target"].values)
            proba_a = model_a.predict_proba(X_test_a)[:, 1]
            diff_a = q4_vs_rest_diff(test, proba_a)
            variazione = diff_a - osservato
            log(f"    Senza '{feat_to_remove}': differenza Q4={diff_a*100:+.2f}pt "
                f"(variazione rispetto al modello completo: {variazione*100:+.2f}pt)")

        riepilogo.append({
            "symbol": symbol, "diff_osservata": osservato,
            "boot_mean": boot_diffs.mean(), "ci_low": ci_low, "ci_high": ci_high,
            "include_zero": include_zero,
        })
        log("")

    log("=" * 70)
    log("VERDETTO FINALE")
    log("=" * 70)
    for row in riepilogo:
        stato = "ROBUSTO (non include zero)" if not row["include_zero"] else "NON ROBUSTO (include zero)"
        log(f"  {row['symbol']}: diff={row['diff_osservata']*100:+.2f}pt, "
            f"IC95%=[{row['ci_low']*100:+.2f}, {row['ci_high']*100:+.2f}] — {stato}")

    pd.DataFrame(riepilogo).to_csv("results/logistic_robustness_riepilogo.csv", index=False)
    with open("results/analyze_logistic_score_robustness.txt", "w") as f:
        f.write("\n".join(log_lines))

    print("\n=== Completato. ===")


if __name__ == "__main__":
    main()
