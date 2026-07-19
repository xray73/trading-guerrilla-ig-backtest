"""
analyze_fold2_bootstrap_2021.py — Bootstrap (19/07/2026) sul gap Q4-resto
DAX per 2021/2022/2023, per rispondere alla domanda aperta
dell'approfondimento precedente (analyze_fold2_2021_deepdive.py):
il -3.73pt del 2021 (AUC=0.505, quasi casuale) è un segnale reale o
rientra nel rumore atteso per un anno senza vero potere predittivo?

Riusa lo STESSO modello allenato 2015-2021 (stesso codice di training/
feature/simulazione esiti degli script precedenti — nessuna
ricalibrazione). Per ciascun anno (2021/2022/2023):

1. BOOTSTRAP (incertezza campionaria): ricampiona con reinserimento i
   trade dell'anno, ricalcola il taglio Q4 (75° percentile dello score)
   e il gap win-rate Q4-resto ad ogni iterazione — quantifica quanto
   e' instabile la stima osservata.
2. TEST DI PERMUTAZIONE (nullo casuale): mescola lo score tra i trade
   dell'anno (rompe il legame score-esito, simulando "zero potere
   predittivo reale") e ricalcola lo stesso gap — costruisce la
   distribuzione che ci si aspetterebbe SE il modello non funzionasse
   affatto quell'anno, per calcolare un p-value a due code sul gap
   osservato.

Nessun criterio di successo/fallimento fissato — dati da discutere,
come da accordo per questo filone. Output: SOLO aggregati (percentili,
medie, p-value) — nessun trade individuale.

Dati: adx_diagnostic_raw (D1, sola lettura) + VIX/VIX3M storico
(Yahoo Finance). Nessuna scrittura su D1. Nessuna modifica a engine.py
o a live_execute.py.
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
YEARS = [("2021", "2021-01-01", "2022-01-01"),
         ("2022", "2022-01-01", "2023-01-01"),
         ("2023", "2023-01-01", "2024-01-01")]
N_BOOT = 2000
SEED = 42


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


def q4_gap(target: np.ndarray, proba: np.ndarray) -> float:
    """Gap win-rate Q4 (score >= 75° percentile) meno resto, in punti percentuali."""
    cut = np.quantile(proba, 0.75)
    q4_mask = proba >= cut
    if q4_mask.sum() == 0 or (~q4_mask).sum() == 0:
        return np.nan
    return (target[q4_mask].mean() - target[~q4_mask].mean()) * 100


def bootstrap_ci(target: np.ndarray, proba: np.ndarray, rng: np.random.Generator, n_boot: int) -> np.ndarray:
    n = len(target)
    gaps = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        gaps[b] = q4_gap(target[idx], proba[idx])
    return gaps


def permutation_null(target: np.ndarray, proba: np.ndarray, rng: np.random.Generator, n_perm: int) -> np.ndarray:
    n = len(target)
    gaps = np.empty(n_perm)
    for p in range(n_perm):
        shuffled_target = rng.permutation(target)
        gaps[p] = q4_gap(shuffled_target, proba)
    return gaps


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

    log("=== Bootstrap + test di permutazione — gap Q4-resto DAX 2021/2022/2023 ===\n")

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

    rng = np.random.default_rng(SEED)

    for year_label, y_start, y_end in YEARS:
        test_y = data[(data["timestamp"] >= pd.Timestamp(y_start, tz="UTC")) &
                      (data["timestamp"] < pd.Timestamp(y_end, tz="UTC"))]
        if len(test_y) < 20:
            log(f"{year_label}: campione troppo piccolo, salto.")
            continue

        X_test_y = scaler.transform(test_y[FEATURES])
        proba = model.predict_proba(X_test_y)[:, 1]
        target = test_y["target"].values
        auc_y = roc_auc_score(target, proba) if len(np.unique(target)) > 1 else float("nan")

        observed_gap = q4_gap(target, proba)

        boot_gaps = bootstrap_ci(target, proba, rng, N_BOOT)
        boot_gaps = boot_gaps[~np.isnan(boot_gaps)]
        ci_low, ci_high = np.percentile(boot_gaps, [2.5, 97.5])
        pct_opposite_sign = (boot_gaps < 0).mean() * 100 if observed_gap >= 0 else (boot_gaps > 0).mean() * 100

        perm_gaps = permutation_null(target, proba, rng, N_BOOT)
        perm_gaps = perm_gaps[~np.isnan(perm_gaps)]
        p_value = (np.abs(perm_gaps) >= abs(observed_gap)).mean()
        perm_percentile = (perm_gaps < observed_gap).mean() * 100

        log(f"\n--- {year_label} ---")
        log(f"  n={len(test_y)}  AUC={auc_y:.3f}  gap osservato Q4-resto={observed_gap:+.2f}pt")
        log(f"  Bootstrap ({N_BOOT} iterazioni, ricampionamento con reinserimento):")
        log(f"    IC 95%: [{ci_low:+.2f}pt, {ci_high:+.2f}pt]  media={boot_gaps.mean():+.2f}pt")
        log(f"    % iterazioni con segno opposto all'osservato: {pct_opposite_sign:.1f}%")
        log(f"  Test di permutazione ({N_BOOT} iterazioni, score mescolato = nullo 'zero potere predittivo'):")
        log(f"    nullo: media={perm_gaps.mean():+.2f}pt  sd={perm_gaps.std():.2f}pt")
        log(f"    gap osservato al percentile {perm_percentile:.1f}% del nullo")
        log(f"    p-value a due code={p_value:.3f}")

    with open("results/analyze_fold2_bootstrap_2021.txt", "w") as f:
        f.write("\n".join(log_lines))

    print("\n=== Completato. ===")


if __name__ == "__main__":
    main()
