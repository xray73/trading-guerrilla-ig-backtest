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
        return {"n_trades": 0, "win_rate_pct": np.nan, "profit_factor": np.nan, "pnl_total": 0.0}
    wins = trades_df[trades_df["pnl"] > 0]
    losses = trades_df[trades_df["pnl"] <= 0]
    sum_wins, sum_losses = wins["pnl"].sum(), losses["pnl"].sum()
    pf = sum_wins / abs(sum_losses) if sum_losses != 0 else np.inf
    return {"n_trades": n, "win_rate_pct": 100 * len(wins) / n,
            "profit_factor": pf, "pnl_total": trades_df["pnl"].sum()}


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

    print("--- SEPARATI (due capitali indipendenti da 2.000 EUR ciascuno) ---")
    print(f"  V6:              n={m_v6['n_trades']} WR={m_v6['win_rate_pct']:.1f}% "
          f"PF={m_v6['profit_factor']:.2f} PnL={m_v6['pnl_total']:+.2f}")
    print(f"  Mean-reversion:  n={m_mr['n_trades']} WR={m_mr['win_rate_pct']:.1f}% "
          f"PF={m_mr['profit_factor']:.2f} PnL={m_mr['pnl_total']:+.2f}")
    print(f"  PnL combinato:   {separated_total_pnl:+.2f} EUR su {separated_total_capital_used:.0f} EUR investiti "
          f"({100*separated_total_pnl/separated_total_capital_used:+.1f}%)\n")

    # --- COMBINATI: un motore, un capitale, router condiviso ---
    combined_signal_data = {name: generate_combined_signals(raw_data[name], eng.INSTRUMENTS[name], mr_mode=MR_MODE)
                             for name in SYMBOLS}

    engine_combined = BacktestEngineFloatingKillSwitch(capital0=CAPITAL0)
    trades_combined, _ = engine_combined.run(combined_signal_data)

    m_combined = metrics_summary(trades_combined)

    print("--- COMBINATI (un solo capitale da 2.000 EUR, router condiviso) ---")
    print(f"  n={m_combined['n_trades']} WR={m_combined['win_rate_pct']:.1f}% "
          f"PF={m_combined['profit_factor']:.2f} PnL={m_combined['pnl_total']:+.2f} "
          f"({100*m_combined['pnl_total']/CAPITAL0:+.1f}% su {CAPITAL0:.0f} EUR)\n")

    print("--- Confronto sintetico ---")
    print(f"  Rendimento % su capitale investito: separati {100*separated_total_pnl/separated_total_capital_used:+.1f}%  "
          f"vs combinati {100*m_combined['pnl_total']/CAPITAL0:+.1f}%")
    print(f"  N trade totali: separati {m_v6['n_trades'] + m_mr['n_trades']}  vs combinati {m_combined['n_trades']}")
    print(f"  (Un N trade combinati molto piu' basso della somma indica che il router ha dovuto "
          f"scartare segnali per limite slot/ordini condivisi — competizione reale per il capitale.)")

    trades_combined.to_csv("router_combined_trades.csv", index=False)
    print("\nCompletato. File: router_combined_trades.csv")


if __name__ == "__main__":
    main()
