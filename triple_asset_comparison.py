"""
triple_asset_comparison.py — Confronta il portafoglio a 3 strumenti
(DAX+FTSE100+GOLD, tutti sempre disponibili insieme, nessuna nuova
logica di selezione) contro il baseline attuale (DAX+FTSE100).

Diverso dai test precedenti su GOLD (sez.22.4, run id 37): quelli
testavano la SOSTITUZIONE di FTSE100 con GOLD (coppia alternativa).
Qui GOLD si AGGIUNGE, DAX e FTSE100 restano entrambi — la domanda è
se più opportunità nello stesso paniere aiutano, non se GOLD è
"migliore" di uno dei due esistenti.

Usa il motore standard adottato (BacktestEngineFloatingKillSwitch,
15/07/2026) e la stessa logica di priorità già esistente quando più
trigger scattano insieme (R:R + penalità di correlazione,
_correlation_penalty in engine.py, invariata) — nessuna nuova logica
di instradamento per regime (quella è il punto 4, condizionale
all'esito del test ATR, non ancora avviato).

NOTA sui parametri GOLD: spread=0.90 da RCA sez.22.1, MAI verificato
in orario di mercato attivo (a differenza di SMI, dove è stato
esplicitamente corretto da 6.0 a 2.0 dopo uno screenshot live) — se
questo test promuove, andrebbe verificato prima di qualunque adozione
reale.

Criterio di promozione (fissato in chat 15/07/2026):
  1. PnL/|drawdown| positivo in 5/5 periodi
  2. Rapporto aggregato (somma PnL / |drawdown peggiore tra i 5|)
     almeno +10% sopra il baseline DAX+FTSE100
  3. Nessun periodo con drawdown singolo oltre -35%
Se non soddisfatto, resta il baseline a 2 strumenti — nessuna soglia
abbassata per far vincere il portafoglio a 3.
"""

from __future__ import annotations

import dataclasses
import pandas as pd

import engine as eng
import ema_grid_search as g
from engine_floating_kill_switch import BacktestEngineFloatingKillSwitch

CAPITAL0 = 2000.0
ALL_PERIODS = list(g.PERIODS.keys())
PROMOTION_MARGIN = 0.10
MAX_SINGLE_PERIOD_DD = 0.35

GOLD_CONFIG = eng.InstrumentConfig(
    name="GOLD", tradable=True,
    breakout_lookback=30, atr_multiplier=3.5, risk_pct=0.015,
    point_value=0.88,   # EUR stimato da USD 1 ~ EUR 0.88, RCA sez.22.1 — MAI verificato live
    spread_fixed=0.90,  # punti, RCA sez.22.1 — MAI verificato in orario di mercato attivo
    min_tradable_size=0.10, margin_pct=0.05,
)

PORTFOLIOS = {
    "DAX_FTSE100": ["DAX", "FTSE100"],
    "DAX_FTSE100_GOLD": ["DAX", "FTSE100", "GOLD"],
}
BASELINE_PORTFOLIO = "DAX_FTSE100"


def get_instrument_config(symbol: str) -> eng.InstrumentConfig:
    if symbol == "GOLD":
        return GOLD_CONFIG
    return eng.INSTRUMENTS[symbol]


def run_portfolio_on_period(symbols: list[str], period_label: str,
                             full_data: dict[str, pd.DataFrame]) -> dict:
    instruments: dict[str, eng.InstrumentConfig] = {}
    data: dict[str, pd.DataFrame] = {}

    for sym in symbols:
        inst = dataclasses.replace(get_instrument_config(sym), tradable=True)
        instruments[sym] = inst
        window, period_start = g.slice_period(full_data[sym], period_label)
        sig = eng.generate_signals(window, inst)
        sig = g.trim_warmup(sig, period_start)
        data[sym] = sig

    engine_ = BacktestEngineFloatingKillSwitch(capital0=CAPITAL0, instruments=instruments)
    trades_df, metrics_df = engine_.run(data)

    pnl = float(metrics_df["pnl_total"].iloc[0])
    dd_raw = metrics_df["max_drawdown_pct"].iloc[0]
    dd = float(dd_raw) if pd.notna(dd_raw) else 0.0
    n_trades = int(metrics_df["num_trades"].iloc[0])
    ratio = (pnl / abs(dd)) if dd != 0 else float("nan")

    return {
        "portfolio": "+".join(symbols), "period": period_label,
        "num_trades": n_trades, "pnl_total": pnl,
        "max_drawdown_pct": dd, "pnl_dd_ratio": ratio,
    }


def main():
    import os
    os.makedirs("results", exist_ok=True)

    full_data = {
        "DAX": g.load_full_ohlc("DAX_full.csv"),
        "FTSE100": g.load_full_ohlc("FTSE100_full.csv"),
        "GOLD": g.load_full_ohlc("GOLD_full.csv"),
    }

    rows = []
    for portfolio_name, symbols in PORTFOLIOS.items():
        print(f"\n=== Portafoglio {portfolio_name} ===")
        for period in ALL_PERIODS:
            row = run_portfolio_on_period(symbols, period, full_data)
            rows.append(row)
            print(f"  [{period}] trades={row['num_trades']:4d} "
                  f"pnl={row['pnl_total']:9.1f} dd={row['max_drawdown_pct']*100:6.2f}% "
                  f"pnl/dd={row['pnl_dd_ratio']:9.1f}")

    detail_df = pd.DataFrame(rows)
    detail_df.to_csv("results/triple_asset_comparison_by_period.csv", index=False)

    baseline_rows = detail_df[detail_df["portfolio"] == "+".join(PORTFOLIOS[BASELINE_PORTFOLIO])]
    baseline_sum_pnl = baseline_rows["pnl_total"].sum()
    baseline_worst_dd = baseline_rows["max_drawdown_pct"].min()
    baseline_ratio = baseline_sum_pnl / abs(baseline_worst_dd) if baseline_worst_dd != 0 else float("nan")

    summary_rows = []
    for portfolio_name, symbols in PORTFOLIOS.items():
        key = "+".join(symbols)
        sub = detail_df[detail_df["portfolio"] == key]

        sum_pnl = sub["pnl_total"].sum()
        worst_dd = sub["max_drawdown_pct"].min()
        ratio = sum_pnl / abs(worst_dd) if worst_dd != 0 else float("nan")

        n_positive = int((sub["pnl_dd_ratio"] > 0).sum())
        criterio1_ok = n_positive == len(ALL_PERIODS)

        vs_baseline_pct = ((ratio / baseline_ratio) - 1.0
                            if portfolio_name != BASELINE_PORTFOLIO and baseline_ratio not in (0,) else None)
        criterio2_ok = (portfolio_name == BASELINE_PORTFOLIO) or (
            vs_baseline_pct is not None and vs_baseline_pct >= PROMOTION_MARGIN)

        criterio3_ok = abs(worst_dd) <= MAX_SINGLE_PERIOD_DD

        promosso = (portfolio_name != BASELINE_PORTFOLIO) and criterio1_ok and criterio2_ok and criterio3_ok

        summary_rows.append({
            "portfolio": portfolio_name,
            "sum_pnl_5periodi": sum_pnl, "worst_drawdown_pct": worst_dd,
            "aggregate_pnl_dd_ratio": ratio, "periodi_positivi": f"{n_positive}/{len(ALL_PERIODS)}",
            "criterio1_5su5_positivi": criterio1_ok, "vs_baseline_pct": vs_baseline_pct,
            "criterio2_margine10pct": criterio2_ok, "criterio3_no_dd_oltre_35pct": criterio3_ok,
            "PROMOSSO": promosso,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("results/triple_asset_comparison_summary.csv", index=False)

    print("\n=== VERDETTO FINALE ===")
    print(summary_df.to_string(index=False))

    promoted = summary_df[summary_df["PROMOSSO"]]
    if promoted.empty:
        print("\nNessun portafoglio a 3 strumenti soddisfa tutti i criteri. "
              "Resta DAX+FTSE100 (default conservativo).")
    else:
        print(f"\nPortafoglio/i promosso/i: {', '.join(promoted['portfolio'].tolist())}")
        print("ATTENZIONE: se promosso, verificare lo spread GOLD in orario di "
              "mercato attivo (mai fatto finora) prima di qualunque adozione reale.")


if __name__ == "__main__":
    main()
