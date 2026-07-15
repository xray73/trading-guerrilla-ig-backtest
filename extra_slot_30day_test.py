"""
extra_slot_30day_test.py — Versione estesa di extra_slot_15day_test.py:
finestra a 30 giorni feriali (parametrica) + metriche complete di
periodo (win/loss, profit factor, PnL, drawdown, R medio) sia per il
baseline (slot 1-3) sia per gli slot extra (4°/5°) sia per il totale
combinato. Corregge anche il bug della versione precedente: gli skip
(PnL<=0 / size minima) ora sono loggati con timestamp
(extra_slot_skip_pnl_log / extra_slot_skip_minsize_log, aggiunti in
engine_extended_orders.py il 16/07/2026) e filtrati sulla finestra
esattamente come i trade, non più contatori cumulativi su tutta la
corsa (warmup incluso).

CONTESTO invariato: analisi descrittiva, NON un test di validazione.
Il fronte "slot extra" resta chiuso sulla base del test train/test +
bootstrap del 15/07/2026 (segnale/rumore=0.012 su 2026-ytd completo).
Questo script non riapre quella decisione.

Nessuna modifica al motore standard. Nessuna scrittura su D1. Solo CSV
locali.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

import dukascopy_python
from dukascopy_python.instruments import INSTRUMENT_IDX_EUROPE_E_DAAX, INSTRUMENT_IDX_EUROPE_E_FUTSEE_100

import engine as eng
from engine_floating_kill_switch import BacktestEngineFloatingKillSwitch
from engine_extended_orders import BacktestEngineExtendedOrders

WARMUP_DAYS = 90
N_TRADING_DAYS = 30
EXTRA_SLOT_PCT = 1.0
CAPITAL0 = 2000.0
SYMBOLS = {"DAX": INSTRUMENT_IDX_EUROPE_E_DAAX, "FTSE100": INSTRUMENT_IDX_EUROPE_E_FUTSEE_100}


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
    """Metriche complete di periodo su un set di trade già filtrato
    per finestra. Restituisce un dict pronto per una riga di summary."""
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
    sum_losses = losses["pnl"].sum()  # negativo o zero
    profit_factor = (sum_wins / abs(sum_losses)) if sum_losses != 0 else np.inf

    # drawdown sull'equity curve dei SOLI trade di questo gruppo, in ordine di entry_time
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
    yesterday_end = (datetime.utcnow() - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0) \
        + timedelta(days=1)
    window_start, window_end = trading_days_window(N_TRADING_DAYS, yesterday_end)
    warmup_start = window_start - timedelta(days=WARMUP_DAYS)

    window_start_utc = pd.Timestamp(window_start, tz="UTC")
    window_end_utc = pd.Timestamp(window_end, tz="UTC")

    print(f"Finestra campione: {window_start.date()} -> {window_end.date()} ({N_TRADING_DAYS} giorni feriali)")

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
    trades_baseline, _ = engine_baseline.run(signal_data)
    trades_baseline["entry_time"] = pd.to_datetime(trades_baseline["entry_time"], utc=True)

    print("Eseguo motore con slot extra (max 5 ordini/giorno, extra_slot_pct=1.0)...")
    p_extended = dataclasses.replace(eng.PARAMS, max_new_orders_per_day=5)
    engine_extended = BacktestEngineExtendedOrders(capital0=CAPITAL0, p=p_extended, extra_slot_pct=EXTRA_SLOT_PCT)
    trades_extended, _ = engine_extended.run(signal_data)
    trades_extended["entry_time"] = pd.to_datetime(trades_extended["entry_time"], utc=True)

    extra_keys = set(engine_extended.extra_slot_log)
    trades_extended["is_extra_slot"] = trades_extended.apply(
        lambda r: (r["instrument"], r["entry_time"]) in extra_keys, axis=1
    )

    in_window = lambda ts: (ts >= window_start_utc) & (ts < window_end_utc)

    baseline_window = trades_baseline[in_window(trades_baseline["entry_time"])].copy()
    extra_window = trades_extended[
        trades_extended["is_extra_slot"] & in_window(trades_extended["entry_time"])
    ].copy()
    # baseline "dentro" il motore esteso (slot 1-3, stesso engine_extended run) — per confronto
    # combinato coerente con un unico equity path
    combined_window = trades_extended[in_window(trades_extended["entry_time"])].copy()

    # skip breakdown filtrato sulla finestra (fix rispetto alla versione 15gg:
    # prima erano contatori cumulativi su tutta la corsa, warmup incluso)
    skip_pnl_window = [
        (instr, ts) for instr, ts in engine_extended.extra_slot_skip_pnl_log
        if window_start_utc <= pd.Timestamp(ts) < window_end_utc
    ]
    skip_minsize_window = [
        (instr, ts) for instr, ts in engine_extended.extra_slot_skip_minsize_log
        if window_start_utc <= pd.Timestamp(ts) < window_end_utc
    ]
    n_attempts_window = len(extra_window) + len(skip_pnl_window) + len(skip_minsize_window)

    print(f"\n{'='*70}")
    print(f"RIEPILOGO — finestra {N_TRADING_DAYS} giorni feriali "
          f"({window_start.date()} -> {window_end.date()})")
    print(f"{'='*70}")
    print(f"Tentativi slot extra (chiamate _open_position oltre il 3°) nella finestra: {n_attempts_window}")
    print(f"  aperti:                        {len(extra_window)}")
    print(f"  saltati per PnL giornata<=0:   {len(skip_pnl_window)}")
    print(f"  saltati per size minima:       {len(skip_minsize_window)}")
    print(f"(Nota: non include i casi in cui il 4°/5° segnale non è mai scattato — "
          f"quelli non chiamano _open_position, non sono contabilizzati qui.)")

    metrics_rows = [
        period_metrics(baseline_window, CAPITAL0, "baseline_slot1-3"),
        period_metrics(extra_window, CAPITAL0, "extra_slot4-5"),
        period_metrics(combined_window, CAPITAL0, "combinato_1-5"),
    ]
    summary_df = pd.DataFrame(metrics_rows)

    print(f"\n{'Gruppo':<20}{'N':>4}{'Win':>5}{'Loss':>5}{'WR%':>7}{'PnL':>10}{'PF':>7}{'AvgR':>7}{'MaxDD%':>8}")
    for row in metrics_rows:
        wr = f"{row['win_rate_pct']:.1f}" if pd.notna(row['win_rate_pct']) else "n/a"
        pf = f"{row['profit_factor']:.2f}" if pd.notna(row['profit_factor']) and np.isfinite(row['profit_factor']) else "n/a"
        ar = f"{row['avg_r_multiple']:.2f}" if pd.notna(row['avg_r_multiple']) else "n/a"
        dd = f"{row['max_drawdown_pct']:.1f}" if pd.notna(row['max_drawdown_pct']) else "n/a"
        print(f"{row['gruppo']:<20}{row['num_trades']:>4}{row['num_wins']:>5}{row['num_losses']:>5}"
              f"{wr:>7}{row['pnl_total']:>10.1f}{pf:>7}{ar:>7}{dd:>8}")

    if not extra_window.empty:
        print("\nDettaglio trade slot extra:")
        for _, t in extra_window.iterrows():
            esito = "WIN " if t["pnl"] > 0 else "LOSS"
            print(f"  {t['instrument']:8s} {t['direction']:5s} entry={t['entry_time']} "
                  f"exit_reason={t['exit_reason']:12s} pnl={t['pnl']:+8.2f} "
                  f"r={t.get('r_multiple', float('nan')):+.2f}  [{esito}]")

    extra_window.to_csv("extra_slot_30day_trades.csv", index=False)
    summary_df.to_csv("extra_slot_30day_summary.csv", index=False)

    print(f"\nCompletato. File: extra_slot_30day_trades.csv, extra_slot_30day_summary.csv")
    print("\nNota: risultato descrittivo, non un test di validazione. Il fronte 'slot extra' "
          "resta chiuso sulla base del test train/test + bootstrap del 15/07/2026.")


if __name__ == "__main__":
    main()
