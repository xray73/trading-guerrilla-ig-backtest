"""
baseline_by_asset_test.py — Analisi descrittiva del solo motore
baseline (BacktestEngineFloatingKillSwitch, standard, max 3 ordini/
giorno, invariato), con metriche di periodo separate per DAX,
FTSE100, e totale combinato. Nessuno slot extra, nessuna modifica al
motore.

Uso:
  python baseline_by_asset_test.py                 # default 30 giorni
  python baseline_by_asset_test.py --days 90        # 90 giorni
  python baseline_by_asset_test.py --days 999       # troncato a 180

Capitale iniziale: 2.000€ (CAPITAL0), stesso riferimento di tutti i
backtest ufficiali del progetto.

Nota sullo split per asset: le metriche per DAX e FTSE100 sono
calcolate sui trade di quello strumento filtrati dall'unica corsa
combinata del motore (stesso capitale condiviso, stesso kill switch
giornaliero — DAX e FTSE100 competono per gli stessi slot/capitale
esattamente come nel motore reale). Non sono due backtest separati
con capitale indipendente: la size dei trade FTSE100 dipende anche da
cosa ha fatto DAX in precedenza nella stessa corsa (capitale
condiviso), quindi lo split è "quanto ha contribuito ciascun
strumento al risultato reale", non "come sarebbe andato ciascuno
strumento isolato con 2.000€ propri".
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

import dukascopy_python
from dukascopy_python.instruments import INSTRUMENT_IDX_EUROPE_E_DAAX, INSTRUMENT_IDX_EUROPE_E_FUTSEE_100

import engine as eng
from engine_floating_kill_switch import BacktestEngineFloatingKillSwitch

WARMUP_DAYS = 90
MAX_TRADING_DAYS = 180
DEFAULT_TRADING_DAYS = 30
CAPITAL0 = 2000.0
SYMBOLS = {"DAX": INSTRUMENT_IDX_EUROPE_E_DAAX, "FTSE100": INSTRUMENT_IDX_EUROPE_E_FUTSEE_100}


def parse_args() -> int:
    parser = argparse.ArgumentParser(description="Analisi baseline con split per asset su N giorni feriali (tetto 180)")
    parser.add_argument("--days", type=int, default=DEFAULT_TRADING_DAYS,
                         help=f"Numero di giorni feriali della finestra (default {DEFAULT_TRADING_DAYS}, tetto {MAX_TRADING_DAYS})")
    args = parser.parse_args()

    n_days = args.days
    if n_days < 1:
        print(f"ATTENZIONE: --days={n_days} non valido, uso il default {DEFAULT_TRADING_DAYS}.")
        n_days = DEFAULT_TRADING_DAYS
    if n_days > MAX_TRADING_DAYS:
        print(f"ATTENZIONE: --days={n_days} supera il tetto di {MAX_TRADING_DAYS}, troncato a {MAX_TRADING_DAYS}.")
        n_days = MAX_TRADING_DAYS
    return n_days


def fetch_bars(symbol_const, start: datetime, end: datetime, interval) -> pd.DataFrame:
    df = dukascopy_python.fetch(
        symbol_const, interval, dukascopy_python.OFFER_SIDE_BID, start, end,
    ).reset_index()
    ts_col = df.columns[0]
    df = df.rename(columns={ts_col: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def trading_days_window(n_days: int, end_exclusive: datetime) -> tuple[datetime, datetime]:
    day = end_exclusive
    counted = 0
    start = day
    while counted < n_days:
        start -= timedelta(days=1)
        if start.weekday() < 5:
            counted += 1
    return start, end_exclusive


def period_metrics(trades: pd.DataFrame, capital0: float, label: str) -> dict:
    n = len(trades)
    if n == 0:
        return {
            "gruppo": label, "num_trades": 0, "num_wins": 0, "num_losses": 0,
            "win_rate_pct": np.nan, "pnl_total": 0.0, "pnl_avg": np.nan,
            "avg_win": np.nan, "avg_loss": np.nan, "profit_factor": np.nan,
            "avg_r_multiple": np.nan, "max_drawdown_pct": np.nan, "max_drawdown_eur": np.nan,
        }

    wins = trades[trades["pnl"] > 0]
    losses = trades[trades["pnl"] <= 0]
    sum_wins = wins["pnl"].sum()
    sum_losses = losses["pnl"].sum()
    profit_factor = (sum_wins / abs(sum_losses)) if sum_losses != 0 else np.inf

    trades_sorted = trades.sort_values("entry_time")
    equity = capital0 + trades_sorted["pnl"].cumsum()
    running_max = np.maximum.accumulate(equity.values)
    drawdown_eur = equity.values - running_max
    drawdown_pct = drawdown_eur / running_max
    max_dd_eur = drawdown_eur.min()
    max_dd_pct = drawdown_pct.min()

    return {
        "gruppo": label,
        "num_trades": n,
        "num_wins": len(wins),
        "num_losses": len(losses),
        "win_rate_pct": 100 * len(wins) / n,
        "pnl_total": trades["pnl"].sum(),
        "pnl_avg": trades["pnl"].mean(),
        "avg_win": wins["pnl"].mean() if len(wins) > 0 else np.nan,
        "avg_loss": losses["pnl"].mean() if len(losses) > 0 else np.nan,
        "profit_factor": profit_factor,
        "avg_r_multiple": trades["r_multiple"].mean() if "r_multiple" in trades.columns else np.nan,
        "max_drawdown_pct": max_dd_pct * 100,
        "max_drawdown_eur": max_dd_eur,
    }


def main():
    n_trading_days = parse_args()

    yesterday_end = (datetime.utcnow() - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0) \
        + timedelta(days=1)
    window_start, window_end = trading_days_window(n_trading_days, yesterday_end)
    warmup_start = window_start - timedelta(days=WARMUP_DAYS)

    window_start_utc = pd.Timestamp(window_start, tz="UTC")
    window_end_utc = pd.Timestamp(window_end, tz="UTC")

    print(f"Finestra campione: {window_start.date()} -> {window_end.date()} ({n_trading_days} giorni feriali)")
    print(f"Capitale iniziale: {CAPITAL0:.2f}€")

    full_data_30m = {}
    for name, const in SYMBOLS.items():
        print(f"Scarico {name} 30min ({warmup_start.date()} -> {window_end.date()})...")
        full_data_30m[name] = fetch_bars(const, warmup_start, window_end, dukascopy_python.INTERVAL_MIN_30)
        print(f"  {len(full_data_30m[name])} barre 30min")

    signal_data = {}
    for name in SYMBOLS:
        inst = eng.INSTRUMENTS[name]
        signal_data[name] = eng.generate_signals(full_data_30m[name], inst)

    print("\nEseguo motore baseline (max 3 ordini/giorno)...")
    engine_baseline = BacktestEngineFloatingKillSwitch(capital0=CAPITAL0)
    trades_all, _ = engine_baseline.run(signal_data)
    trades_all["entry_time"] = pd.to_datetime(trades_all["entry_time"], utc=True)

    in_window = (trades_all["entry_time"] >= window_start_utc) & (trades_all["entry_time"] < window_end_utc)
    trades_window = trades_all[in_window].copy()

    dax_window = trades_window[trades_window["instrument"] == "DAX"].copy()
    ftse_window = trades_window[trades_window["instrument"] == "FTSE100"].copy()

    metrics_rows = [
        period_metrics(dax_window, CAPITAL0, "DAX"),
        period_metrics(ftse_window, CAPITAL0, "FTSE100"),
        period_metrics(trades_window, CAPITAL0, "totale_combinato"),
    ]
    summary_df = pd.DataFrame(metrics_rows)

    print(f"\n{'='*70}")
    print(f"RIEPILOGO — finestra {n_trading_days} giorni feriali "
          f"({window_start.date()} -> {window_end.date()}), capitale {CAPITAL0:.0f}€")
    print(f"{'='*70}")
    print(f"\n{'Gruppo':<18}{'N':>4}{'Win':>5}{'Loss':>5}{'WR%':>7}{'PnL':>10}{'PF':>7}{'AvgR':>7}{'MaxDD%':>8}")
    for row in metrics_rows:
        wr = f"{row['win_rate_pct']:.1f}" if pd.notna(row['win_rate_pct']) else "n/a"
        pf = f"{row['profit_factor']:.2f}" if pd.notna(row['profit_factor']) and np.isfinite(row['profit_factor']) else "n/a"
        ar = f"{row['avg_r_multiple']:.2f}" if pd.notna(row['avg_r_multiple']) else "n/a"
        dd = f"{row['max_drawdown_pct']:.1f}" if pd.notna(row['max_drawdown_pct']) else "n/a"
        print(f"{row['gruppo']:<18}{row['num_trades']:>4}{row['num_wins']:>5}{row['num_losses']:>5}"
              f"{wr:>7}{row['pnl_total']:>10.1f}{pf:>7}{ar:>7}{dd:>8}")

    suffix = f"{n_trading_days}day"
    trades_window.to_csv(f"baseline_{suffix}_trades.csv", index=False)
    summary_df.to_csv(f"baseline_{suffix}_summary.csv", index=False)

    print(f"\nCompletato. File: baseline_{suffix}_trades.csv, baseline_{suffix}_summary.csv")
    print("\nNota: DAX e FTSE100 condividono lo stesso capitale/kill switch nella corsa — "
          "lo split mostra il contributo di ciascuno al risultato reale, non una simulazione isolata.")


if __name__ == "__main__":
    main()
