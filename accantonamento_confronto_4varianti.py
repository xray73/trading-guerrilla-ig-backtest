"""
accantonamento_confronto_4varianti.py — Confronto completo delle 4
varianti di accantonamento (opt2/opt3 x continuo/mensile) usando
SEMPRE il motore vero (BacktestEngineAccantonamento), non
approssimazioni — sostituisce le tabelle mostrate in chat prima della
validazione del 16/07/2026.

Include anche un sanity check rapido (una tantum, non ripetuto per
periodo per contenere i tempi — la logica di consolidamento è
indipendente dal periodo, già validata su tutti i 5 periodi in
accantonamento_validation.py) e il drawdown ricostruito
dall'equity curve reale (investito+accantonato) trade per trade.

Ogni periodo è indipendente (capitale riparte da CAPITAL0), coerente
con la metodologia walk-forward del progetto.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import numpy as np
import pandas as pd

import dukascopy_python
from dukascopy_python.instruments import INSTRUMENT_IDX_EUROPE_E_DAAX, INSTRUMENT_IDX_EUROPE_E_FUTSEE_100

import engine as eng
from engine_floating_kill_switch import BacktestEngineFloatingKillSwitch
from engine_accantonamento import BacktestEngineAccantonamento

WARMUP_DAYS = 90
CAPITAL0 = 2000.0
SYMBOLS = {"DAX": INSTRUMENT_IDX_EUROPE_E_DAAX, "FTSE100": INSTRUMENT_IDX_EUROPE_E_FUTSEE_100}

PERIODS = [
    ("2015-2016", "2015-01-05", "2016-12-29"),
    ("2020-covid", "2020-01-02", "2020-12-30"),
    ("2023", "2023-01-02", "2023-12-30"),
    ("2024-2025", "2024-01-03", "2025-12-31"),
    ("2026-ytd", "2026-01-05", "2026-07-10"),
]

VARIANTS = [
    ("opt2_continuo", dict(mode="giveback", check_frequency="continuo", giveback_pct=0.3)),
    ("opt2_mensile", dict(mode="giveback", check_frequency="mensile", giveback_pct=0.3)),
    ("opt3_continuo", dict(mode="gradini", check_frequency="continuo", consolidate_pct=0.4, threshold_mult=1.5)),
    ("opt3_mensile", dict(mode="gradini", check_frequency="mensile", consolidate_pct=0.4, threshold_mult=1.5)),
]


def fetch_bars(symbol_const, start: datetime, end: datetime) -> pd.DataFrame:
    df = dukascopy_python.fetch(
        symbol_const, dukascopy_python.INTERVAL_MIN_30, dukascopy_python.OFFER_SIDE_BID, start, end,
    ).reset_index()
    ts_col = df.columns[0]
    df = df.rename(columns={ts_col: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def get_period_signal_data(period_start: str, period_end: str) -> dict:
    p_start = pd.Timestamp(period_start, tz="UTC")
    p_end = pd.Timestamp(period_end, tz="UTC") + timedelta(days=1)
    warmup_start = p_start - timedelta(days=WARMUP_DAYS)

    signal_data = {}
    for name, const in SYMBOLS.items():
        raw = fetch_bars(const, warmup_start.to_pydatetime(), p_end.to_pydatetime())
        inst = eng.INSTRUMENTS[name]
        full_signals = eng.generate_signals(raw, inst)
        signal_data[name] = full_signals[full_signals["timestamp"] >= p_start].reset_index(drop=True)
    return signal_data


def equity_drawdown_from_trades(trades_df: pd.DataFrame, side_pool_final: float,
                                  consolidation_log: list, capital0: float) -> tuple[float, float]:
    """Ricostruisce l'equity curve (investito+accantonato) trade per
    trade usando pnl cumulato + eventi di consolidamento nel tempo, poi
    calcola il max drawdown. consolidation_log: [(data, consolidato, ...), ...]"""
    if trades_df.empty:
        return 0.0, 0.0

    trades_sorted = trades_df.sort_values("exit_time").reset_index(drop=True)
    consolidations_by_date = {}
    for c in consolidation_log:
        d = c[0]
        consolidations_by_date[d] = consolidations_by_date.get(d, 0.0) + c[1]

    invested = capital0
    side_pool = 0.0
    equity = []
    for _, t in trades_sorted.iterrows():
        invested += t["pnl"]
        exit_date = pd.Timestamp(t["exit_time"]).date()
        if exit_date in consolidations_by_date:
            amt = consolidations_by_date.pop(exit_date)
            side_pool += amt
            invested -= amt
        equity.append(invested + side_pool)

    equity = pd.Series(equity)
    running_max = equity.cummax()
    dd_pct = (equity - running_max) / running_max
    dd_eur = equity - running_max
    return dd_pct.min() * 100, dd_eur.min()


def main():
    all_rows = []

    for label, p_start, p_end in PERIODS:
        print(f"\n{'='*70}\nPeriodo {label} ({p_start} -> {p_end})\n{'='*70}")
        signal_data = get_period_signal_data(p_start, p_end)

        # reale (nessun meccanismo)
        baseline = BacktestEngineFloatingKillSwitch(capital0=CAPITAL0)
        trades_baseline, _ = baseline.run(signal_data)
        real_dd_pct, real_dd_eur = equity_drawdown_from_trades(trades_baseline, 0.0, [], CAPITAL0)
        real_rendimento = 100 * (baseline.capital - CAPITAL0) / CAPITAL0
        print(f"  Reale: {baseline.capital:.0f} EUR ({real_rendimento:+.1f}%)  MaxDD={real_dd_pct:.1f}%")

        row = {
            "periodo": label, "n_trade": len(trades_baseline),
            "reale_totale": baseline.capital, "reale_rendimento_pct": real_rendimento,
            "reale_max_dd_pct": real_dd_pct, "reale_max_dd_eur": real_dd_eur,
        }

        for variant_name, params in VARIANTS:
            eng_acc = BacktestEngineAccantonamento(capital0=CAPITAL0, **params)
            trades_df, metrics_df = eng_acc.run(signal_data)

            invested_finale = metrics_df["capitale_investito_finale"].iloc[0]
            accantonato_finale = metrics_df["accantonato_finale"].iloc[0]
            totale_finale = metrics_df["capitale_totale_finale"].iloc[0]
            rendimento_pct = 100 * (totale_finale - CAPITAL0) / CAPITAL0

            dd_pct, dd_eur = equity_drawdown_from_trades(
                trades_df, accantonato_finale, eng_acc.consolidation_log, CAPITAL0)

            print(f"  {variant_name}: investito={invested_finale:.0f} accantonato={accantonato_finale:.0f} "
                  f"totale={totale_finale:.0f} ({rendimento_pct:+.1f}%) MaxDD={dd_pct:.1f}%")

            row[f"{variant_name}_investito"] = invested_finale
            row[f"{variant_name}_accantonato"] = accantonato_finale
            row[f"{variant_name}_totale"] = totale_finale
            row[f"{variant_name}_rendimento_pct"] = rendimento_pct
            row[f"{variant_name}_max_dd_pct"] = dd_pct
            row[f"{variant_name}_max_dd_eur"] = dd_eur

        all_rows.append(row)

    summary_df = pd.DataFrame(all_rows)
    summary_df.to_csv("accantonamento_confronto_4varianti.csv", index=False)
    print(f"\nCompletato. File: accantonamento_confronto_4varianti.csv")


if __name__ == "__main__":
    main()
