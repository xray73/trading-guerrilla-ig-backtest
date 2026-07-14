"""
daily_resample_test.py — Testa il filtro "trend ampio" con vero resampling
giornaliero (EMA100/200 su chiusure daily reali) al posto dell'attuale
approssimazione (EMA100/200 calcolate su barre 30min — RCA sez.9, limite
dichiarato ma mai verificato).

Isola UNA sola variabile: tutto il resto del segnale (EMA veloce 20/50,
ADX>20, breakout lookback 20/40, ATR 1.5x, rischio per strumento) resta
IDENTICO al motore v2 in produzione — solo la fonte del filtro trend ampio
cambia da 30min a daily reale.

Criterio di valutazione: STESSO fissato in sessione per l'obiettivo reale
dell'utente (win rate, non z-score) — pavimento trade 90% del baseline,
promozione al test con margine +2pp di win rate. Nessuna simulazione
random necessaria per questo criterio → job molto più leggero (pochi
backtest reali, niente 30 seed per combinazione).

Disciplina walk-forward: selezione/valutazione su 2023 (train) → verifica
2024-2025 (test, out-of-sample, nessun ritocco) → conferma sui 3 periodi
restanti (solo riportati, nessuna soglia pass/fail rigida — coerente con
come sono stati validati DAX/FTSE100 nella RCA originale).

No look-ahead nel resampling: ogni barra intraday usa l'EMA100/200
giornaliera calcolata sulle chiusure fino al giorno di calendario
PRECEDENTE (mai il giorno corrente, che non è ancora chiuso).
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
import requests

import engine as eng

DATABASE_ID = "b9fbd4d6-7837-4d86-9c0f-ca60c0cf69e3"
API_BASE = "https://api.cloudflare.com/client/v4/accounts"
CHUNK_SIZE = 5000

FLOOR = 0.90          # pavimento trade rispetto al baseline (fissato in sessione)
MARGIN_PP = 0.02       # margine minimo win rate per promozione (fissato in sessione)

PERIODS = {
    "2015-2016": ("2015-01-01", "2016-12-31"),
    "2020-covid": ("2020-01-01", "2020-12-31"),
    "2023": ("2023-01-01", "2023-12-31"),          # TRAIN
    "2024-2025": ("2024-01-01", "2025-12-31"),      # TEST
    "2026-ytd": ("2026-01-01", "2026-07-12"),
}
TRAIN_PERIOD = "2023"
TEST_PERIOD = "2024-2025"
CONFIRM_PERIODS = ["2015-2016", "2020-covid", "2026-ytd"]
CAPITAL0 = 900.0


def d1_query(sql: str, account_id: str, token: str) -> list[dict]:
    url = f"{API_BASE}/{account_id}/d1/database/{DATABASE_ID}/query"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    resp = requests.post(url, headers=headers, json={"sql": sql}, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if not data.get("success"):
        raise RuntimeError(f"D1 query fallita: {data.get('errors')}")
    return data["result"][0]["results"]


def fetch_all_ohlc(symbol: str, account_id: str, token: str) -> pd.DataFrame:
    rows = []
    offset = 0
    while True:
        sql = (
            f"SELECT timestamp, open, high, low, close FROM ohlc_prices "
            f"WHERE symbol='{symbol}' ORDER BY timestamp LIMIT {CHUNK_SIZE} OFFSET {offset}"
        )
        batch = d1_query(sql, account_id, token)
        if not batch:
            break
        rows.extend(batch)
        offset += CHUNK_SIZE
        if len(batch) < CHUNK_SIZE:
            break
        time.sleep(0.15)
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def slice_period(df: pd.DataFrame, period_label: str, warmup_days: int = 400) -> tuple[pd.DataFrame, pd.Timestamp]:
    # warmup ampio (400gg) perché l'EMA200 GIORNALIERA ha bisogno di ~200
    # giorni di CALENDARIO di dati precedenti, molto più della vecchia
    # versione 30min (che bastava ~90gg) — è parte del punto da verificare
    start_str, end_str = PERIODS[period_label]
    start = pd.Timestamp(start_str, tz="UTC") - pd.Timedelta(days=warmup_days)
    end = pd.Timestamp(end_str, tz="UTC") + pd.Timedelta(days=1)
    window = df[(df["timestamp"] >= start) & (df["timestamp"] < end)].reset_index(drop=True)
    return window, pd.Timestamp(start_str, tz="UTC")


def generate_signals_daily_broad(df: pd.DataFrame, inst: eng.InstrumentConfig,
                                   p: eng.ChartaParams = eng.PARAMS) -> pd.DataFrame:
    """Identico a eng.generate_signals, TRANNE il filtro trend ampio che usa
    EMA100/200 su resampling giornaliero reale invece che su barre 30min."""
    out = eng.compute_indicators(df, inst, p)

    # --- resampling giornaliero reale ---
    daily = df.copy()
    daily["date"] = daily["timestamp"].dt.date
    daily_close = daily.groupby("date")["close"].last().reset_index()
    daily_close["ema100_d"] = daily_close["close"].ewm(span=100, adjust=False).mean()
    daily_close["ema200_d"] = daily_close["close"].ewm(span=200, adjust=False).mean()
    # ogni valore diventa disponibile SOLO dal giorno di calendario successivo
    # (no look-ahead: il giorno corrente non è ancora chiuso quando si opera)
    daily_close["effective_date"] = pd.to_datetime(daily_close["date"]) + pd.Timedelta(days=1)

    out["date_dt"] = pd.to_datetime(out["timestamp"].dt.date)
    out = out.merge(
        daily_close[["effective_date", "ema100_d", "ema200_d"]],
        left_on="date_dt", right_on="effective_date", how="left"
    )
    out[["ema100_d", "ema200_d"]] = out[["ema100_d", "ema200_d"]].ffill()

    direction_long = out["ema_fast"] > out["ema_slow"]
    direction_short = out["ema_fast"] < out["ema_slow"]
    adx_context_ok = out["adx"] > p.adx_min_context
    breakout_long = out["close"] > out["rolling_high"]
    breakout_short = out["close"] < out["rolling_low"]

    broad_trend_long_ok = out["ema100_d"] > out["ema200_d"]
    broad_trend_short_ok = out["ema100_d"] < out["ema200_d"]

    long_signal = direction_long & adx_context_ok & breakout_long & broad_trend_long_ok
    short_signal = direction_short & adx_context_ok & breakout_short & broad_trend_short_ok

    out["signal"] = None
    out.loc[long_signal, "signal"] = "long"
    out.loc[short_signal, "signal"] = "short"
    return out


def trim_warmup(df: pd.DataFrame, period_start: pd.Timestamp) -> pd.DataFrame:
    return df[df["timestamp"] >= period_start].reset_index(drop=True)


def run_portfolio(data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    engine_ = eng.BacktestEngine(capital0=CAPITAL0)
    return engine_.run(data)


def eval_period(period_label: str, full_data: dict[str, pd.DataFrame],
                 use_daily_broad: bool) -> dict:
    data = {}
    for name, full_df in full_data.items():
        inst = eng.INSTRUMENTS[name]
        window, period_start = slice_period(full_df, period_label)
        if use_daily_broad:
            sig = generate_signals_daily_broad(window, inst)
        else:
            sig = eng.generate_signals(window, inst)
        sig = trim_warmup(sig, period_start)
        data[name] = sig

    trades_df, metrics_df = run_portfolio(data)
    num_trades = int(metrics_df["num_trades"].iloc[0])
    win_rate = float(metrics_df["win_rate"].iloc[0]) if num_trades else 0.0
    return {
        "config": "daily_broad" if use_daily_broad else "baseline_30min_broad",
        "period": period_label,
        "num_trades": num_trades,
        "win_rate": win_rate,
        "pnl_total": float(metrics_df["pnl_total"].iloc[0]),
        "profit_factor": float(metrics_df["profit_factor"].iloc[0]) if num_trades else np.nan,
    }


def main():
    token = os.environ.get("CLOUDFLARE_API_TOKEN")
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    if not token or not account_id:
        print("ERRORE: CLOUDFLARE_API_TOKEN o CLOUDFLARE_ACCOUNT_ID mancanti.", file=sys.stderr)
        sys.exit(1)

    os.makedirs("results", exist_ok=True)

    print("Scarico OHLC DAX + FTSE100...")
    full_data = {
        "DAX": fetch_all_ohlc("DAX", account_id, token),
        "FTSE100": fetch_all_ohlc("FTSE100", account_id, token),
    }
    for name, df in full_data.items():
        print(f"  {name}: {len(df)} barre")

    rows = []

    # --- TRAIN ---
    print(f"\n=== TRAIN ({TRAIN_PERIOD}) ===")
    baseline_train = eval_period(TRAIN_PERIOD, full_data, use_daily_broad=False)
    candidate_train = eval_period(TRAIN_PERIOD, full_data, use_daily_broad=True)
    rows += [baseline_train, candidate_train]
    print(f"  baseline (30min broad): trades={baseline_train['num_trades']} "
          f"win_rate={baseline_train['win_rate']*100:.2f}%")
    print(f"  candidato (daily broad): trades={candidate_train['num_trades']} "
          f"win_rate={candidate_train['win_rate']*100:.2f}%")

    trade_floor = baseline_train["num_trades"] * FLOOR
    wr_threshold = baseline_train["win_rate"] + MARGIN_PP
    print(f"\n  Pavimento trade: {trade_floor:.0f}   Soglia promozione win_rate: {wr_threshold*100:.2f}%")

    promoted = (candidate_train["num_trades"] >= trade_floor and
                candidate_train["win_rate"] >= wr_threshold)

    if not promoted:
        print(f"\n  NON PROMOSSO al test — non soddisfa pavimento trade e/o soglia win_rate.")
        pd.DataFrame(rows).to_csv("results/daily_resample_results.csv", index=False)
        print("\nChiuso qui, come da criterio fissato in sessione. File salvato in results/.")
        return

    print(f"\n  PROMOSSO al test out-of-sample.")

    # --- TEST ---
    print(f"\n=== TEST ({TEST_PERIOD}, out-of-sample, nessun ritocco) ===")
    baseline_test = eval_period(TEST_PERIOD, full_data, use_daily_broad=False)
    candidate_test = eval_period(TEST_PERIOD, full_data, use_daily_broad=True)
    rows += [baseline_test, candidate_test]
    print(f"  baseline (30min broad): trades={baseline_test['num_trades']} "
          f"win_rate={baseline_test['win_rate']*100:.2f}%")
    print(f"  candidato (daily broad): trades={candidate_test['num_trades']} "
          f"win_rate={candidate_test['win_rate']*100:.2f}%")

    trade_floor_test = baseline_test["num_trades"] * FLOOR
    wr_threshold_test = baseline_test["win_rate"] + MARGIN_PP
    survives_test = (candidate_test["num_trades"] >= trade_floor_test and
                      candidate_test["win_rate"] >= wr_threshold_test)

    if not survives_test:
        print(f"\n  NON SUPERA il test out-of-sample (pavimento={trade_floor_test:.0f}, "
              f"soglia_wr={wr_threshold_test*100:.2f}%) — pattern train-vince/test-crolla, "
              f"stesso esito del filtro maturità (Addendum sez.17). Chiuso qui.")
        pd.DataFrame(rows).to_csv("results/daily_resample_results.csv", index=False)
        return

    print(f"\n  SUPERA il test — procedo alla conferma sui 3 periodi restanti.")

    # --- CONFIRM ---
    print(f"\n=== CONFERMA (3 periodi restanti, solo riportati) ===")
    for period in CONFIRM_PERIODS:
        b = eval_period(period, full_data, use_daily_broad=False)
        c = eval_period(period, full_data, use_daily_broad=True)
        rows += [b, c]
        print(f"  {period}: baseline trades={b['num_trades']} win_rate={b['win_rate']*100:.2f}%   "
              f"candidato trades={c['num_trades']} win_rate={c['win_rate']*100:.2f}%")

    pd.DataFrame(rows).to_csv("results/daily_resample_results.csv", index=False)
    print("\nCompletato. File in results/.")


if __name__ == "__main__":
    main()
