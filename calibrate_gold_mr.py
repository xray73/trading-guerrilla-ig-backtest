"""
calibrate_gold_mr.py — Calibrazione walk-forward del moltiplicatore
ATR per GOLD in modalità mean-reversion — MAI fatta finora. I
parametri GOLD attuali (ATR×3,5) sono stati calibrati per V6
(breakout, RCA 13/07 sez.22.3) e semplicemente riusati per MR, stessa
convenzione già in uso per DAX/FTSE100 (nessuno strumento ha mai avuto
un moltiplicatore MR-specifico) — ma il test del 19/07 (V6+GOLD vs
MR+GOLD sui 5 periodi) suggerisce che per GOLD questo riuso è
probabilmente sbagliato: MR+GOLD ha fallito in modo netto (0/5 periodi,
drawdown fino a -33%), mentre V6+GOLD è stato relativamente sano.

ISOLAMENTO: calibra GOLD in mean-reversion DA SOLO (nessuna
competizione con DAX/FTSE100 per gli slot), per misurare la qualità
del segnale MR su GOLD indipendentemente dal meccanismo di selezione
multi-asset — quella è una domanda separata, testata dopo.

Protocollo (identico a RCA 13/07 sez.22.3, stesso standard di rigore):
  - Grid: moltiplicatore ATR in {1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0}
    (range gia' usato altrove nel progetto per sweep ATR)
  - RSI(14)/ADX<20 (trigger di ingresso) INVARIATI — non ricalibrati,
    stessa convenzione di DAX/FTSE100 (nessun parametro di trigger
    mai calibrato per strumento, solo lo stop/target)
  - Metrica: PnL / |max_drawdown%| (criterio corretto del progetto,
    Charter_updates 14/07 — non win_rate/PF isolati)
  - Train: 2023 -> verifica fuori campione: 2024-2025 -> conferma sui
    restanti 3 periodi (2015-16, 2020-covid, 2026-ytd)
  - Capitale 2.000EUR (capitale di riferimento pieno, per isolare la
    qualita' del segnale dagli effetti di size minima — coerente con
    come sono state calibrate le altre coppie ATR/lookback nel progetto)

Nessuna scrittura su D1. Nessuna modifica a engine.py, engine_mean_reversion.py
o mean_reversion_signals.py — usa solo dataclasses.replace su una copia
locale di GOLD_CONFIG.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime, timezone
import numpy as np
import pandas as pd

import dukascopy_python
from dukascopy_python.instruments import INSTRUMENT_FX_METALS_XAU_USD

from engine_mean_reversion import BacktestEngineMeanReversion
from mean_reversion_signals import generate_mean_reversion_signals
from engine_three_asset_gold import GOLD_CONFIG

CAPITAL0 = 2000.0
ATR_GRID = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]

TRAIN_LABEL, TRAIN_START, TRAIN_END = "2023", "2023-01-02", "2023-12-30"
TEST_LABEL, TEST_START, TEST_END = "2024-2025", "2024-01-03", "2025-12-31"
CONFIRM_PERIODS = [
    ("2015-2016", "2015-01-05", "2016-12-29"),
    ("2020-covid", "2020-01-02", "2020-12-30"),
    ("2026-ytd", "2026-01-05", "2026-07-10"),
]

FULL_FETCH_START = datetime(2014, 10, 1, tzinfo=timezone.utc)
FULL_FETCH_END = datetime(2026, 7, 11, tzinfo=timezone.utc)


def fetch_gold_full() -> pd.DataFrame:
    df = dukascopy_python.fetch(
        INSTRUMENT_FX_METALS_XAU_USD, dukascopy_python.INTERVAL_MIN_30, dukascopy_python.OFFER_SIDE_BID,
        FULL_FETCH_START, FULL_FETCH_END,
    ).reset_index()
    ts_col = df.columns[0]
    df = df.rename(columns={ts_col: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def slice_period(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    p_start = pd.Timestamp(start, tz="UTC")
    p_end = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    return df[(df["timestamp"] >= p_start) & (df["timestamp"] < p_end)].reset_index(drop=True)


def run_for_multiplier(signals_period: pd.DataFrame, atr_mult: float) -> dict:
    inst = dataclasses.replace(GOLD_CONFIG, atr_multiplier=atr_mult)
    eng_mr = BacktestEngineMeanReversion(capital0=CAPITAL0, instruments={"GOLD": inst})
    trades, _ = eng_mr.run({"GOLD": signals_period})

    if trades.empty:
        return {"n_trades": 0, "pnl": 0.0, "max_dd_pct": 0.0, "pnl_over_dd": 0.0, "win_rate_pct": np.nan}

    equity = CAPITAL0 + trades["pnl"].cumsum()
    running_max = equity.cummax()
    dd = ((equity - running_max) / running_max).min() * 100
    pnl = trades["pnl"].sum()
    pnl_over_dd = pnl / abs(dd) if dd != 0 else (float("inf") if pnl > 0 else 0.0)
    wins = (trades["pnl"] > 0).sum()

    return {
        "n_trades": len(trades), "pnl": pnl, "max_dd_pct": dd,
        "pnl_over_dd": pnl_over_dd, "win_rate_pct": 100 * wins / len(trades),
    }


def main():
    log_lines = []
    def log(msg):
        print(msg)
        log_lines.append(msg)

    log("=== Calibrazione ATR moltiplicatore — GOLD mean-reversion (isolato) ===\n")

    log("Scarico storico GOLD (fetch unico, 2014-2026)...")
    raw = fetch_gold_full()
    log("Fatto.\n")

    log("Genero segnali mean-reversion RSI su tutta la serie (indipendenti dal moltiplicatore ATR)...")
    signals_full = generate_mean_reversion_signals(raw, GOLD_CONFIG, mode="rsi")
    log("Fatto.\n")

    # ================================================================
    # TRAIN: 2023
    # ================================================================
    log(f"--- TRAIN ({TRAIN_LABEL}) — grid ATR {ATR_GRID} ---")
    train_sig = slice_period(signals_full, TRAIN_START, TRAIN_END)
    train_results = {}
    for mult in ATR_GRID:
        r = run_for_multiplier(train_sig, mult)
        train_results[mult] = r
        log(f"  ATR={mult:.1f}x  n={r['n_trades']:3d}  WR={r['win_rate_pct']:.1f}%  "
            f"PnL={r['pnl']:+.2f}  maxDD={r['max_dd_pct']:.1f}%  PnL/|DD|={r['pnl_over_dd']:+.2f}")

    best_mult = max(train_results, key=lambda m: train_results[m]["pnl_over_dd"])
    log(f"\n  Migliore in train: ATR={best_mult}x (PnL/|DD|={train_results[best_mult]['pnl_over_dd']:+.2f})\n")

    # ================================================================
    # TEST fuori campione: 2024-2025
    # ================================================================
    log(f"--- TEST fuori campione ({TEST_LABEL}) — SOLO il candidato migliore in train ---")
    test_sig = slice_period(signals_full, TEST_START, TEST_END)
    r_test_best = run_for_multiplier(test_sig, best_mult)
    r_test_current = run_for_multiplier(test_sig, 3.5)  # valore attuale (riciclato da V6), per confronto diretto
    log(f"  ATR={best_mult}x (candidato train): n={r_test_best['n_trades']:3d}  WR={r_test_best['win_rate_pct']:.1f}%  "
        f"PnL={r_test_best['pnl']:+.2f}  maxDD={r_test_best['max_dd_pct']:.1f}%  PnL/|DD|={r_test_best['pnl_over_dd']:+.2f}")
    log(f"  ATR=3.5x (attuale, riciclato da V6): n={r_test_current['n_trades']:3d}  WR={r_test_current['win_rate_pct']:.1f}%  "
        f"PnL={r_test_current['pnl']:+.2f}  maxDD={r_test_current['max_dd_pct']:.1f}%  PnL/|DD|={r_test_current['pnl_over_dd']:+.2f}\n")

    train_test_holds = r_test_best["pnl_over_dd"] > 0 and r_test_best["pnl"] > 0
    log(f"  Il candidato di train regge fuori campione: {'SI' if train_test_holds else 'NO'}\n")

    # ================================================================
    # CONFERMA sui restanti 3 periodi
    # ================================================================
    log("--- CONFERMA sui restanti 3 periodi ufficiali ---")
    confirm_rows = []
    for label, start, end in CONFIRM_PERIODS:
        sig = slice_period(signals_full, start, end)
        r_best = run_for_multiplier(sig, best_mult)
        r_current = run_for_multiplier(sig, 3.5)
        log(f"  {label}:")
        log(f"    ATR={best_mult}x: n={r_best['n_trades']:3d}  PnL={r_best['pnl']:+.2f}  "
            f"maxDD={r_best['max_dd_pct']:.1f}%  PnL/|DD|={r_best['pnl_over_dd']:+.2f}")
        log(f"    ATR=3.5x (attuale): n={r_current['n_trades']:3d}  PnL={r_current['pnl']:+.2f}  "
            f"maxDD={r_current['max_dd_pct']:.1f}%  PnL/|DD|={r_current['pnl_over_dd']:+.2f}")
        confirm_rows.append({"periodo": label, "best_pnl": r_best["pnl"], "best_dd": r_best["max_dd_pct"],
                              "current_pnl": r_current["pnl"], "current_dd": r_current["max_dd_pct"]})

    log(f"\n{'='*70}\nRIEPILOGO\n{'='*70}")
    total_current = r_test_current["pnl"] + sum(r["current_pnl"] for r in confirm_rows)
    total_best_pnl = r_test_best["pnl"] + sum(r["best_pnl"] for r in confirm_rows)
    log(f"PnL totale (test+conferma, 4 periodi) — ATR={best_mult}x: {total_best_pnl:+.2f}  "
        f"ATR=3.5x attuale: {total_current:+.2f}")
    log(f"Moltiplicatore raccomandato per GOLD mean-reversion: "
        f"{best_mult}x se regge fuori campione, altrimenti mantenere 3.5x o considerare GOLD non idoneo per MR")

    import os
    os.makedirs("results", exist_ok=True)
    with open("results/calibrate_gold_mr.txt", "w") as f:
        f.write("\n".join(log_lines))

    print("\n=== Completato. ===")


if __name__ == "__main__":
    main()
