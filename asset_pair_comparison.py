"""
asset_pair_comparison.py — Confronto universo asset a 2 strumenti
Agente AI Trading Guerrilla IG — Fase 1 backtest offline
=====================================================================

Confronta 3 coppie candidate per l'universo attivo, con capitale
di riferimento fissato a 2.000€ (limite reale dichiarato dall'utente,
14/07/2026):

    DAX+FTSE100   (coppia attuale, Charter sez.3)
    DAX+GOLD      (esplorata in RCA Addendum 13/07 sez.22.4, mai promossa
                   a decisione — vantaggio +7.2% osservato ma senza
                   criterio di promozione fissato a priori)
    FTSE100+GOLD  (mai testata — correlazione più bassa in assoluto tra
                   le coppie note, RCA Addendum 13/07 sez.19/23: 0.10)

NON modifica engine.py — istanzia BacktestEngine con un dizionario
`instruments` costruito ad-hoc per ciascuna coppia (stesso pattern già
usato da ema_grid_search.py per isolare varianti senza toccare il motore
di produzione).

Parametri GOLD: ATR=3.5/lookback=30/risk=1.5%, quelli già calibrati in
RCA Addendum 13/07 sez.22.3 (grid search train 2023, verifica 4/5
periodi). NOTA — point_value GOLD stimato in EUR da un cambio USD/EUR
indicativo (~0.88, RCA sez.22.1: "USD 1 (~EUR 0.88)"): non è un tasso
di cambio verificato in tempo reale, va aggiornato se questa coppia
viene promossa oltre questo test esplorativo.

Criterio di promozione (fissato PRIMA di vedere i risultati, confermato
in chat 14/07/2026):
    1. PnL/|drawdown massimo aggregato| positivo in 5/5 periodi
    2. Rapporto aggregato (somma PnL / |drawdown peggiore tra i 5 periodi|)
       almeno +10% sopra la coppia attuale DAX+FTSE100
    3. Nessun periodo con drawdown singolo oltre -35% (rete di sicurezza
       assoluta, richiesta esplicitamente per questo test)
Se nessuna coppia soddisfa tutti e 3 i criteri, resta DAX+FTSE100
(default conservativo — nessuna soglia si abbassa per far vincere
qualcuno).

Richiede DAX_full.csv, FTSE100_full.csv, GOLD_full.csv nella working
directory (prodotti da fetch_ohlc_pairs.py, stesso pattern di
fetch_ohlc_d1.py ma esteso a GOLD).

Output: results/asset_pair_comparison_by_period.csv (dettaglio per
periodo/coppia) e results/asset_pair_comparison_summary.csv (verdetto
finale sui 3 criteri).
"""

from __future__ import annotations

import dataclasses
import os
import numpy as np
import pandas as pd

import engine as eng                # motore di produzione, MAI modificato
import ema_grid_search as g         # riuso PERIODS, slice_period, trim_warmup, load_full_ohlc


# =====================================================================
# 0. CONFIGURAZIONE TEST
# =====================================================================

CAPITAL0 = 2000.0   # capitale di riferimento (limite reale dichiarato 14/07/2026)

GOLD_CONFIG = eng.InstrumentConfig(
    name="GOLD", tradable=True,
    breakout_lookback=30, atr_multiplier=3.5, risk_pct=0.015,
    point_value=0.88,          # EUR stimato da USD 1 ~ EUR 0.88, RCA sez.22.1 — verificare se promosso
    spread_fixed=0.90,         # punti, RCA sez.22.1
    min_tradable_size=0.10, margin_pct=0.05,
)

PAIRS: dict[str, list[str]] = {
    "DAX_FTSE100": ["DAX", "FTSE100"],
    "DAX_GOLD": ["DAX", "GOLD"],
    "FTSE100_GOLD": ["FTSE100", "GOLD"],
}

BASELINE_PAIR = "DAX_FTSE100"
ALL_PERIODS = list(g.PERIODS.keys())   # 5 periodi standard del progetto

PROMOTION_MARGIN = 0.10        # +10% sul rapporto aggregato, stessa soglia usata
                                # per i test di uscita (sez.31.2/31.4 RCA 14/07)
MAX_SINGLE_PERIOD_DD = 0.35    # -35%, rete di sicurezza assoluta per questo test


def get_instrument_config(symbol: str) -> eng.InstrumentConfig:
    if symbol == "GOLD":
        return GOLD_CONFIG
    return eng.INSTRUMENTS[symbol]


# =====================================================================
# 1. BACKTEST DI UNA COPPIA SU UN PERIODO
# =====================================================================

def run_pair_on_period(pair_symbols: list[str], period_label: str,
                        full_data: dict[str, pd.DataFrame]) -> dict:
    instruments: dict[str, eng.InstrumentConfig] = {}
    data: dict[str, pd.DataFrame] = {}

    for sym in pair_symbols:
        inst = dataclasses.replace(get_instrument_config(sym), tradable=True)
        instruments[sym] = inst
        window, period_start = g.slice_period(full_data[sym], period_label)
        sig = eng.generate_signals(window, inst)
        sig = g.trim_warmup(sig, period_start)
        data[sym] = sig

    engine_ = eng.BacktestEngine(capital0=CAPITAL0, instruments=instruments)
    trades_df, metrics_df = engine_.run(data)

    pnl = float(metrics_df["pnl_total"].iloc[0])
    dd_raw = metrics_df["max_drawdown_pct"].iloc[0]
    dd = float(dd_raw) if pd.notna(dd_raw) else 0.0
    n_trades = int(metrics_df["num_trades"].iloc[0])
    ratio = (pnl / abs(dd)) if dd != 0 else np.nan

    return {
        "pair": "+".join(pair_symbols),
        "period": period_label,
        "num_trades": n_trades,
        "pnl_total": pnl,
        "max_drawdown_pct": dd,
        "pnl_dd_ratio": ratio,
    }


# =====================================================================
# 2. MAIN — esegue tutte le coppie su tutti i periodi, poi verdetto
# =====================================================================

def main():
    os.makedirs("results", exist_ok=True)

    full_data = {
        "DAX": g.load_full_ohlc("DAX_full.csv"),
        "FTSE100": g.load_full_ohlc("FTSE100_full.csv"),
        "GOLD": g.load_full_ohlc("GOLD_full.csv"),
    }

    rows = []
    for pair_name, symbols in PAIRS.items():
        print(f"\n=== Coppia {pair_name} ===")
        for period in ALL_PERIODS:
            row = run_pair_on_period(symbols, period, full_data)
            rows.append(row)
            print(f"  [{period}] trades={row['num_trades']:4d} "
                  f"pnl={row['pnl_total']:9.1f} dd={row['max_drawdown_pct']*100:6.2f}% "
                  f"pnl/dd={row['pnl_dd_ratio']:9.1f}")

    detail_df = pd.DataFrame(rows)
    detail_df.to_csv("results/asset_pair_comparison_by_period.csv", index=False)

    # --- Verdetto sui 3 criteri, per coppia ---
    baseline_rows = detail_df[detail_df["pair"] == "+".join(PAIRS[BASELINE_PAIR])]
    baseline_sum_pnl = baseline_rows["pnl_total"].sum()
    baseline_worst_dd = baseline_rows["max_drawdown_pct"].min()  # più negativo
    baseline_aggregate_ratio = (baseline_sum_pnl / abs(baseline_worst_dd)
                                 if baseline_worst_dd != 0 else np.nan)

    summary_rows = []
    for pair_name, symbols in PAIRS.items():
        pair_key = "+".join(symbols)
        pair_rows = detail_df[detail_df["pair"] == pair_key]

        sum_pnl = pair_rows["pnl_total"].sum()
        worst_dd = pair_rows["max_drawdown_pct"].min()
        aggregate_ratio = sum_pnl / abs(worst_dd) if worst_dd != 0 else np.nan

        n_positive = int((pair_rows["pnl_dd_ratio"] > 0).sum())
        criterio1_ok = n_positive == len(ALL_PERIODS)

        vs_baseline_pct = ((aggregate_ratio / baseline_aggregate_ratio) - 1.0
                            if pair_name != BASELINE_PAIR and baseline_aggregate_ratio
                            not in (0, np.nan) else np.nan)
        criterio2_ok = (pair_name == BASELINE_PAIR) or (
            pd.notna(vs_baseline_pct) and vs_baseline_pct >= PROMOTION_MARGIN)

        worst_single_dd = pair_rows["max_drawdown_pct"].min()
        criterio3_ok = abs(worst_single_dd) <= MAX_SINGLE_PERIOD_DD

        promosso = (pair_name != BASELINE_PAIR) and criterio1_ok and criterio2_ok and criterio3_ok

        summary_rows.append({
            "pair": pair_name,
            "sum_pnl_5periodi": sum_pnl,
            "worst_drawdown_pct": worst_dd,
            "aggregate_pnl_dd_ratio": aggregate_ratio,
            "periodi_positivi": f"{n_positive}/{len(ALL_PERIODS)}",
            "criterio1_5su5_positivi": criterio1_ok,
            "vs_baseline_pct": vs_baseline_pct,
            "criterio2_margine10pct": criterio2_ok,
            "criterio3_no_dd_oltre_35pct": criterio3_ok,
            "PROMOSSO": promosso,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("results/asset_pair_comparison_summary.csv", index=False)

    print("\n=== VERDETTO FINALE ===")
    print(summary_df.to_string(index=False))

    promoted = summary_df[summary_df["PROMOSSO"]]
    if promoted.empty:
        print("\nNessuna coppia soddisfa tutti e 3 i criteri di promozione. "
              "Resta DAX+FTSE100 (default conservativo, nessuna soglia abbassata).")
    else:
        print(f"\nCoppia/e promossa/e: {', '.join(promoted['pair'].tolist())}")

    print("\nCompletato. Ricorda: se una coppia viene promossa, questa run va "
          "registrata in backtest_runs (D1) con nota distintiva prima di "
          "qualunque modifica al Charter (pre-run checklist, Regole_Backtest_MonteCarlo.md).")


if __name__ == "__main__":
    main()
