"""
orb_stop_grid_test.py — Step 2 del train ORB (17/07/2026): griglia SOLO
sul moltiplicatore ATR dello stop, con ADX threshold=20 (scelto allo
step 1) e R:R target=2.0 tenuti fissi — un parametro alla volta, stessa
disciplina già applicata alla griglia ADX.

PERIODO DI TRAIN: 2015-2016 — l'unico dei 5 periodi ufficiali mai
toccato da nessun test ORB finora (2024-2025 e 2020-covid già usati
per la griglia ADX, gli ultimi 180gg per la fattibilità). Il 2023
resta ancora vergine per il verdetto finale fuori campione, dopo aver
scelto anche questo parametro.

Griglia ATR: [1.5, 2.0, 2.5] — 3 valori, non una scansione fine, stesso
motivo di contenimento overfitting della griglia ADX. 1.5 è il valore
di Variante 6 (per confronto diretto); 2.0 era l'ipotesi di partenza
usata finora nei test ORB; 2.5 testa se serve ancora più margine.

Nessuna modifica a engine.py.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

import dukascopy_python
from dukascopy_python.instruments import INSTRUMENT_IDX_EUROPE_E_DAAX, INSTRUMENT_IDX_EUROPE_E_FUTSEE_100

import engine as eng
from engine_floating_kill_switch import BacktestEngineFloatingKillSwitch
from orb_adx_signals import generate_orb_adx_signals

CAPITAL0 = 2000.0
WARMUP_DAYS = 30
SYMBOLS = {"DAX": INSTRUMENT_IDX_EUROPE_E_DAAX, "FTSE100": INSTRUMENT_IDX_EUROPE_E_FUTSEE_100}

TRAIN_PERIOD = ("2015-2016", "2015-01-05", "2016-12-29")

ADX_THRESHOLD_FIXED = 20.0   # scelto allo step 1
RR_TARGET_FIXED = 2.0

ATR_MULT_GRID = [1.5, 2.0, 2.5]


def fetch_bars(symbol_const, start: datetime, end: datetime) -> pd.DataFrame:
    df = dukascopy_python.fetch(
        symbol_const, dukascopy_python.INTERVAL_MIN_30, dukascopy_python.OFFER_SIDE_BID, start, end,
    ).reset_index()
    ts_col = df.columns[0]
    df = df.rename(columns={ts_col: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def metrics_row(trades_df: pd.DataFrame, instrument: str | None = None) -> dict:
    sub = trades_df if instrument is None else trades_df[trades_df["instrument"] == instrument]
    n = len(sub)
    if n == 0:
        return {"n_trades": 0, "win_rate_pct": np.nan, "profit_factor": np.nan,
                "pnl_total": 0.0, "avg_r": np.nan}
    wins = sub[sub["pnl"] > 0]
    losses = sub[sub["pnl"] <= 0]
    sum_wins = wins["pnl"].sum()
    sum_losses = losses["pnl"].sum()
    pf = sum_wins / abs(sum_losses) if sum_losses != 0 else np.inf
    return {
        "n_trades": n,
        "win_rate_pct": 100 * len(wins) / n,
        "profit_factor": pf,
        "pnl_total": sub["pnl"].sum(),
        "avg_r": sub["r_multiple"].mean() if "r_multiple" in sub.columns else np.nan,
    }


def main():
    label, p_start, p_end = TRAIN_PERIOD
    p_start_ts = pd.Timestamp(p_start, tz="UTC")
    p_end_ts = pd.Timestamp(p_end, tz="UTC") + timedelta(days=1)
    warmup_start = p_start_ts - timedelta(days=WARMUP_DAYS)

    print(f"=== Griglia stop ATR per ORB — periodo train {label} ({p_start} -> {p_end}) ===")
    print(f"ADX threshold fisso: {ADX_THRESHOLD_FIXED}, R:R fisso: 1:{RR_TARGET_FIXED}\n")

    raw_data = {}
    for name, const in SYMBOLS.items():
        print(f"Scarico {name}...")
        raw_data[name] = fetch_bars(const, warmup_start.to_pydatetime(), p_end_ts.to_pydatetime())

    all_rows = []
    for atr_mult in ATR_MULT_GRID:
        revised_params = dataclasses.replace(eng.PARAMS, rr_target=RR_TARGET_FIXED)
        revised_instruments = {
            name: dataclasses.replace(inst, atr_multiplier=atr_mult)
            for name, inst in eng.INSTRUMENTS.items()
        }

        signal_data = {}
        for name in SYMBOLS:
            inst = revised_instruments[name]
            full_signals = generate_orb_adx_signals(
                raw_data[name], inst, name, adx_threshold=ADX_THRESHOLD_FIXED, p=revised_params)
            signal_data[name] = full_signals[full_signals["timestamp"] >= p_start_ts].reset_index(drop=True)

        engine_ = BacktestEngineFloatingKillSwitch(capital0=CAPITAL0, p=revised_params, instruments=revised_instruments)
        trades_df, _ = engine_.run(signal_data)

        combined = metrics_row(trades_df)
        dax = metrics_row(trades_df, "DAX")
        ftse = metrics_row(trades_df, "FTSE100")

        print(f"--- ATR moltiplicatore = {atr_mult} ---")
        print(f"  Combinato: n={combined['n_trades']} WR={combined['win_rate_pct']:.1f}% "
              f"PF={combined['profit_factor']:.2f} PnL={combined['pnl_total']:+.2f} avgR={combined['avg_r']:+.3f}"
              if combined['n_trades'] > 0 else "  Combinato: nessun trade")
        print(f"  DAX:       n={dax['n_trades']} WR={dax['win_rate_pct']:.1f}% "
              f"PF={dax['profit_factor']:.2f} PnL={dax['pnl_total']:+.2f}"
              if dax['n_trades'] > 0 else "  DAX: nessun trade")
        print(f"  FTSE100:   n={ftse['n_trades']} WR={ftse['win_rate_pct']:.1f}% "
              f"PF={ftse['profit_factor']:.2f} PnL={ftse['pnl_total']:+.2f}\n"
              if ftse['n_trades'] > 0 else "  FTSE100: nessun trade\n")

        all_rows.append({"atr_multiplier": atr_mult, "gruppo": "combinato", **combined})
        all_rows.append({"atr_multiplier": atr_mult, "gruppo": "DAX", **dax})
        all_rows.append({"atr_multiplier": atr_mult, "gruppo": "FTSE100", **ftse})

    summary_df = pd.DataFrame(all_rows)
    summary_df.to_csv("orb_stop_grid_train_results_2015_2016.csv", index=False)
    print("Completato. File: orb_stop_grid_train_results_2015_2016.csv")
    print("\nRicorda: 2023 resta vergine per il verdetto finale fuori campione dopo questo step.")


if __name__ == "__main__":
    main()
