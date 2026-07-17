"""
orb_adx_grid_test.py — Step 1 del train dei parametri ORB (17/07/2026):
griglia SOLO su ADX threshold, stop/target tenuti fissi (ma rivisti
rispetto a quelli presi in prestito da Variante 6 nel test di
fattibilità) — un parametro alla volta, non griglia incrociata, per
contenere il rischio di overfitting con questo numero di trade.

PERIODO DI TRAIN: 2024-2025 (uno dei 5 periodi ufficiali), MAI toccato
da nessun test ORB finora — il test di fattibilità aveva usato gli
ultimi 180 giorni (~gen-lug 2026), che si sovrappone al 2026-ytd, non
al 2024-2025. Il 2023 (e il resto) restano ancora "vergini" per il
verdetto finale out-of-sample dopo aver scelto la soglia ADX.

IPOTESI DI PARTENZA per stop/target (fissa in questo test, NON
ottimizzata qui — sarà lo step 2 del train, dopo aver scelto l'ADX):
  - ATR moltiplicatore: 2.0 (invece di 1.5 di Variante 6 — ipotesi che
    un breakout da apertura di sessione abbia bisogno di più margine
    subito dopo l'apertura, più volatile del solito)
  - R:R target: 1:2 (invariato, punto di partenza semplice)
Stop sempre basato su ATR (mai distanza fissa arbitraria), coerente
con la regola del progetto.

Griglia ADX: [0 (nessun filtro), 15, 20, 25] — 4 valori, non una
scansione fine, per lo stesso motivo di contenimento overfitting.

Nessuna modifica a engine.py — i parametri modificati (ATR
moltiplicatore, R:R) sono passati come istanze locali via
dataclasses.replace(), mai scritti nel file condiviso.
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

TRAIN_PERIOD = ("2024-2025", "2024-01-03", "2025-12-31")

ADX_GRID = [0.0, 15.0, 20.0, 25.0]

# ipotesi di partenza stop/target — fisse in questo test
REVISED_ATR_MULT = 2.0
REVISED_RR_TARGET = 2.0


def fetch_bars(symbol_const, start: datetime, end: datetime) -> pd.DataFrame:
    df = dukascopy_python.fetch(
        symbol_const, dukascopy_python.INTERVAL_MIN_30, dukascopy_python.OFFER_SIDE_BID, start, end,
    ).reset_index()
    ts_col = df.columns[0]
    df = df.rename(columns={ts_col: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def build_revised_config():
    """Copie locali di ChartaParams/INSTRUMENTS con ATR/RR rivisti —
    engine.py non viene mai toccato."""
    revised_params = dataclasses.replace(eng.PARAMS, rr_target=REVISED_RR_TARGET)
    revised_instruments = {
        name: dataclasses.replace(inst, atr_multiplier=REVISED_ATR_MULT)
        for name, inst in eng.INSTRUMENTS.items()
    }
    return revised_params, revised_instruments


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

    print(f"=== Griglia ADX per ORB — periodo train {label} ({p_start} -> {p_end}) ===")
    print(f"Stop/target fissi: ATR x{REVISED_ATR_MULT}, R:R 1:{REVISED_RR_TARGET}\n")

    raw_data = {}
    for name, const in SYMBOLS.items():
        print(f"Scarico {name}...")
        raw_data[name] = fetch_bars(const, warmup_start.to_pydatetime(), p_end_ts.to_pydatetime())

    revised_params, revised_instruments = build_revised_config()

    all_rows = []
    for threshold in ADX_GRID:
        signal_data = {}
        for name in SYMBOLS:
            inst = revised_instruments[name]
            full_signals = generate_orb_adx_signals(raw_data[name], inst, name, adx_threshold=threshold, p=revised_params)
            signal_data[name] = full_signals[full_signals["timestamp"] >= p_start_ts].reset_index(drop=True)

        engine_ = BacktestEngineFloatingKillSwitch(capital0=CAPITAL0, p=revised_params, instruments=revised_instruments)
        trades_df, _ = engine_.run(signal_data)

        combined = metrics_row(trades_df)
        dax = metrics_row(trades_df, "DAX")
        ftse = metrics_row(trades_df, "FTSE100")

        print(f"--- ADX threshold = {threshold:.0f} ---")
        print(f"  Combinato: n={combined['n_trades']} WR={combined['win_rate_pct']:.1f}% "
              f"PF={combined['profit_factor']:.2f} PnL={combined['pnl_total']:+.2f} avgR={combined['avg_r']:+.3f}"
              if combined['n_trades'] > 0 else "  Combinato: nessun trade")
        print(f"  DAX:       n={dax['n_trades']} WR={dax['win_rate_pct']:.1f}% "
              f"PF={dax['profit_factor']:.2f} PnL={dax['pnl_total']:+.2f}"
              if dax['n_trades'] > 0 else "  DAX: nessun trade")
        print(f"  FTSE100:   n={ftse['n_trades']} WR={ftse['win_rate_pct']:.1f}% "
              f"PF={ftse['profit_factor']:.2f} PnL={ftse['pnl_total']:+.2f}\n"
              if ftse['n_trades'] > 0 else "  FTSE100: nessun trade\n")

        all_rows.append({"adx_threshold": threshold, "gruppo": "combinato", **combined})
        all_rows.append({"adx_threshold": threshold, "gruppo": "DAX", **dax})
        all_rows.append({"adx_threshold": threshold, "gruppo": "FTSE100", **ftse})

    summary_df = pd.DataFrame(all_rows)
    summary_df.to_csv("orb_adx_grid_train_results.csv", index=False)
    print("Completato. File: orb_adx_grid_train_results.csv")
    print("\nRicorda: questo è il periodo di TRAIN. La soglia scelta qui va verificata su un periodo "
          "mai toccato (es. 2023 o altro dei 5 ufficiali) prima di considerarla definitiva.")


if __name__ == "__main__":
    main()
