"""
router_combined_test.py — Confronta due modalità di convivenza tra
Variante 6 e mean-reversion (17/07/2026):

  SEPARATI: due motori indipendenti, ciascuno con CAPITAL0 proprio
      (stesso approccio del test di fattibilità) — capitale/rischio
      NON condivisi, i due sistemi non "si vedono" a vicenda.
  COMBINATI: un solo motore, un solo capitale, un solo kill switch,
      segnale scelto dal router (combined_router_signals.py) in base
      all'ADX barra per barra — capitale/rischio DAVVERO condivisi,
      slot in competizione reale.

Stesso periodo (ultimi 180 giorni, non uno dei 5 periodi ufficiali —
solo un primo confronto descrittivo prima di eventuale validazione
formale). Mean-reversion: variante RSI (PF più alto nel test di
fattibilità, 1.12 vs 1.06 di Bollinger) — se serve confrontare anche
Bollinger, si ripete lo stesso script cambiando MR_MODE.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import numpy as np
import pandas as pd

import dukascopy_python
from dukascopy_python.instruments import INSTRUMENT_IDX_EUROPE_E_DAAX, INSTRUMENT_IDX_EUROPE_E_FUTSEE_100

import engine as eng
from engine_floating_kill_switch import BacktestEngineFloatingKillSwitch
from combined_router_signals import generate_combined_signals
from mean_reversion_signals import generate_mean_reversion_signals

WARMUP_DAYS = 30
DAYS_BACK = 180
CAPITAL0 = 2000.0
MR_MODE = "rsi"
SYMBOLS = {"DAX": INSTRUMENT_IDX_EUROPE_E_DAAX, "FTSE100": INSTRUMENT_IDX_EUROPE_E_FUTSEE_100}


def fetch_bars(symbol_const, start: datetime, end: datetime) -> pd.DataFrame:
    df = dukascopy_python.fetch(
        symbol_const, dukascopy_python.INTERVAL_MIN_30, dukascopy_python.OFFER_SIDE_BID, start, end,
    ).reset_index()
    ts_col = df.columns[0]
    df = df.rename(columns={ts_col: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def metrics_summary(trades_df: pd.DataFrame) -> dict:
    n = len(trades_df)
    if n == 0:
        return {"n_trades": 0, "win_rate_pct": np.nan, "profit_factor": np.nan, "pnl_total": 0.0,
                "max_dd_pct": np.nan, "max_dd_eur": np.nan}
    wins = trades_df[trades_df["pnl"] > 0]
    losses = trades_df[trades_df["pnl"] <= 0]
    sum_wins, sum_losses = wins["pnl"].sum(), losses["pnl"].sum()
    pf = sum_wins / abs(sum_losses) if sum_losses != 0 else np.inf

    dd_pct, dd_eur = compute_drawdown(trades_df, CAPITAL0)

    return {"n_trades": n, "win_rate_pct": 100 * len(wins) / n,
            "profit_factor": pf, "pnl_total": trades_df["pnl"].sum(),
            "max_dd_pct": dd_pct, "max_dd_eur": dd_eur}


def compute_combined_wealth_drawdown(trades_a: pd.DataFrame, trades_b: pd.DataFrame, capital0_each: float) -> tuple[float, float]:
    """Per lo scenario SEPARATI: patrimonio totale = capitale_v6 +
    capitale_mr nel tempo (due pool indipendenti sommati), per un
    confronto equo del drawdown contro lo scenario combinato (un solo
    pool). Unisce gli eventi di chiusura di entrambi i motori in
    ordine cronologico e traccia il patrimonio totale ad ogni passo."""
    events = []
    for _, t in trades_a.iterrows():
        events.append((pd.Timestamp(t["exit_time"]), "a", t["pnl"]))
    for _, t in trades_b.iterrows():
        events.append((pd.Timestamp(t["exit_time"]), "b", t["pnl"]))
    events.sort(key=lambda e: e[0])

    cap_a, cap_b = capital0_each, capital0_each
    totals = []
    for _, which, pnl in events:
        if which == "a":
            cap_a += pnl
        else:
            cap_b += pnl
        totals.append(cap_a + cap_b)

    if not totals:
        return 0.0, 0.0
    totals = pd.Series(totals)
    running_max = totals.cummax()
    dd_eur = totals - running_max
    dd_pct = dd_eur / running_max
    return dd_pct.min() * 100, dd_eur.min()
    """Drawdown sull'equity curve ricostruita in ordine di uscita
    (exit_time) — coerente con quando il PnL si realizza davvero."""
    if trades_df.empty:
        return 0.0, 0.0
    trades_sorted = trades_df.sort_values("exit_time")
    equity = capital0 + trades_sorted["pnl"].cumsum()
    running_max = equity.cummax()
    dd_eur = equity - running_max
    dd_pct = dd_eur / running_max
    return dd_pct.min() * 100, dd_eur.min()


def main():
    end = datetime.now()
    start = end - timedelta(days=DAYS_BACK)
    warmup_start = start - timedelta(days=WARMUP_DAYS)

    print(f"=== Separati vs Combinati (router) — mean-reversion mode={MR_MODE} ===")
    print(f"Periodo: {start.date()} -> {end.date()} ({DAYS_BACK} giorni)\n")

    raw_data = {}
    for name, const in SYMBOLS.items():
        print(f"Scarico {name}...")
        raw_data[name] = fetch_bars(const, warmup_start, end)

    # --- SEPARATI: due motori, due capitali indipendenti ---
    v6_signal_data = {name: eng.generate_signals(raw_data[name], eng.INSTRUMENTS[name]) for name in SYMBOLS}
    mr_signal_data = {name: generate_mean_reversion_signals(raw_data[name], eng.INSTRUMENTS[name], mode=MR_MODE)
                       for name in SYMBOLS}

    engine_v6 = BacktestEngineFloatingKillSwitch(capital0=CAPITAL0)
    trades_v6, _ = engine_v6.run(v6_signal_data)

    engine_mr = BacktestEngineFloatingKillSwitch(capital0=CAPITAL0)
    trades_mr, _ = engine_mr.run(mr_signal_data)

    m_v6 = metrics_summary(trades_v6)
    m_mr = metrics_summary(trades_mr)
    separated_total_pnl = m_v6["pnl_total"] + m_mr["pnl_total"]
    separated_total_capital_used = CAPITAL0 * 2  # due pool indipendenti
    separated_wealth_dd_pct, separated_wealth_dd_eur = compute_combined_wealth_drawdown(trades_v6, trades_mr, CAPITAL0)

    print("--- SEPARATI (due capitali indipendenti da 2.000 EUR ciascuno) ---")
    print(f"  V6:              n={m_v6['n_trades']} WR={m_v6['win_rate_pct']:.1f}% "
          f"PF={m_v6['profit_factor']:.2f} PnL={m_v6['pnl_total']:+.2f} MaxDD={m_v6['max_dd_pct']:.1f}%")
    print(f"  Mean-reversion:  n={m_mr['n_trades']} WR={m_mr['win_rate_pct']:.1f}% "
          f"PF={m_mr['profit_factor']:.2f} PnL={m_mr['pnl_total']:+.2f} MaxDD={m_mr['max_dd_pct']:.1f}%")
    print(f"  PnL combinato:   {separated_total_pnl:+.2f} EUR su {separated_total_capital_used:.0f} EUR investiti "
          f"({100*separated_total_pnl/separated_total_capital_used:+.1f}%)")
    print(f"  MaxDD patrimonio totale (V6+MR sommati nel tempo): {separated_wealth_dd_pct:.1f}% "
          f"({separated_wealth_dd_eur:.0f} EUR)\n")

    # --- COMBINATI: un motore, un capitale, router condiviso ---
    combined_signal_data = {name: generate_combined_signals(raw_data[name], eng.INSTRUMENTS[name], mr_mode=MR_MODE)
                             for name in SYMBOLS}

    engine_combined = BacktestEngineFloatingKillSwitch(capital0=CAPITAL0)
    trades_combined, _ = engine_combined.run(combined_signal_data)

    m_combined = metrics_summary(trades_combined)

    print("--- COMBINATI (un solo capitale da 2.000 EUR, router condiviso) ---")
    print(f"  n={m_combined['n_trades']} WR={m_combined['win_rate_pct']:.1f}% "
          f"PF={m_combined['profit_factor']:.2f} PnL={m_combined['pnl_total']:+.2f} "
          f"({100*m_combined['pnl_total']/CAPITAL0:+.1f}% su {CAPITAL0:.0f} EUR) "
          f"MaxDD={m_combined['max_dd_pct']:.1f}%\n")

    print("--- Confronto sintetico ---")
    print(f"  Rendimento % su capitale investito: separati {100*separated_total_pnl/separated_total_capital_used:+.1f}%  "
          f"vs combinati {100*m_combined['pnl_total']/CAPITAL0:+.1f}%")
    print(f"  MaxDD patrimonio totale: separati {separated_wealth_dd_pct:.1f}%  vs combinati {m_combined['max_dd_pct']:.1f}%")
    print(f"  N trade totali: separati {m_v6['n_trades'] + m_mr['n_trades']}  vs combinati {m_combined['n_trades']}")
    print(f"  (Un N trade combinati molto piu' basso della somma indica che il router ha dovuto "
          f"scartare segnali per limite slot/ordini condivisi — competizione reale per il capitale.)")

    trades_combined.to_csv("router_combined_trades.csv", index=False)
    print("\nCompletato. File: router_combined_trades.csv")


if __name__ == "__main__":
    main()
