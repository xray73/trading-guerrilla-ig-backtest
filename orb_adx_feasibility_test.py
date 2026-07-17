"""
orb_adx_feasibility_test.py — Primo test sul segnale ORB+ADX, criteri
fissati in chat il 17/07/2026 PRIMA di vedere risultati:

  1. FATTIBILITÀ: quanti trade/anno genera? Se troppo pochi (regola
     empirica: <30 trade/anno per strumento è sotto la soglia di
     significatività già usata altrove nel progetto, vedi
     compute_run_metrics "significativo"), il segnale non è
     utilizzabile indipendentemente dalla sua qualità — un ORB con un
     solo trade/giorno possibile per strumento è strutturalmente più
     raro della Variante 6.
  2. CONFRONTO CON/SENZA FILTRO ADX: stesso ORB, adx_threshold=0 (mai
     blocca, sempre armato) vs adx_threshold=20 (soglia proposta) —
     per capire se il filtro sta davvero scartando giorni "cattivi" o
     sta solo riducendo il campione senza migliorare nulla.

NON è ancora un test di validazione vero (quello richiede sanity check
+ i 5 periodi ufficiali + stima del rumore, stesso protocollo di ogni
altro meccanismo del progetto) — è il primo controllo, via/no-via,
prima di investire altro tempo su questo segnale.

Periodo di test: ultimi 180 giorni (stesso campione già usato per
baseline_by_asset_test.py, comodo per un primo sguardo, NON uno dei 5
periodi ufficiali).
"""

from __future__ import annotations

from datetime import datetime, timedelta
import pandas as pd

import dukascopy_python
from dukascopy_python.instruments import INSTRUMENT_IDX_EUROPE_E_DAAX, INSTRUMENT_IDX_EUROPE_E_FUTSEE_100

import engine as eng
from engine_floating_kill_switch import BacktestEngineFloatingKillSwitch
from orb_adx_signals import generate_orb_adx_signals

WARMUP_DAYS = 30  # ADX(14) ha bisogno di molto meno storico del lookback breakout di Variante 6
DAYS_BACK = 180
CAPITAL0 = 2000.0
SYMBOLS = {"DAX": INSTRUMENT_IDX_EUROPE_E_DAAX, "FTSE100": INSTRUMENT_IDX_EUROPE_E_FUTSEE_100}


def fetch_bars(symbol_const, start: datetime, end: datetime) -> pd.DataFrame:
    df = dukascopy_python.fetch(
        symbol_const, dukascopy_python.INTERVAL_MIN_30, dukascopy_python.OFFER_SIDE_BID, start, end,
    ).reset_index()
    ts_col = df.columns[0]
    df = df.rename(columns={ts_col: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def run_variant(signal_data: dict, label: str):
    engine_ = BacktestEngineFloatingKillSwitch(capital0=CAPITAL0)
    trades_df, metrics_df = engine_.run(signal_data)

    print(f"\n--- {label} ---")
    if trades_df.empty:
        print("  Nessun trade generato.")
        return trades_df

    for instr in SYMBOLS:
        n = (trades_df["instrument"] == instr).sum()
        print(f"  {instr}: {n} trade nel periodo ({n * 365 / DAYS_BACK:.0f} trade/anno stimati)")

    wins = (trades_df["pnl"] > 0).sum()
    print(f"  Totale: {len(trades_df)} trade, win rate {100*wins/len(trades_df):.1f}%, "
          f"PnL totale {trades_df['pnl'].sum():+.2f} EUR")
    return trades_df


def main():
    end = datetime.now()
    start = end - timedelta(days=DAYS_BACK)
    warmup_start = start - timedelta(days=WARMUP_DAYS)

    print(f"Periodo test: {start.date()} -> {end.date()} ({DAYS_BACK} giorni)")

    raw_data = {}
    for name, const in SYMBOLS.items():
        print(f"Scarico {name}...")
        raw_data[name] = fetch_bars(const, warmup_start, end)

    # variante A: filtro ADX attivo (soglia 20, come da specifica)
    signal_data_filtered = {}
    for name in SYMBOLS:
        inst = eng.INSTRUMENTS[name]
        signal_data_filtered[name] = generate_orb_adx_signals(
            raw_data[name], inst, name, adx_threshold=20.0)

    # variante B: nessun filtro ADX (soglia 0, sempre armato) — per isolare l'effetto del filtro
    signal_data_unfiltered = {}
    for name in SYMBOLS:
        inst = eng.INSTRUMENTS[name]
        signal_data_unfiltered[name] = generate_orb_adx_signals(
            raw_data[name], inst, name, adx_threshold=0.0)

    trades_filtered = run_variant(signal_data_filtered, "ORB + ADX>20 (specifica)")
    trades_unfiltered = run_variant(signal_data_unfiltered, "ORB senza filtro ADX (adx_threshold=0)")

    trades_filtered.to_csv("orb_adx_feasibility_filtered.csv", index=False)
    trades_unfiltered.to_csv("orb_adx_feasibility_unfiltered.csv", index=False)

    print("\n=== Verdetto fattibilità (soglia empirica: 30 trade/anno per strumento) ===")
    for instr in SYMBOLS:
        n = (trades_filtered["instrument"] == instr).sum() if not trades_filtered.empty else 0
        annualized = n * 365 / DAYS_BACK
        stato = "OK" if annualized >= 30 else "SOTTO SOGLIA — probabilmente non utilizzabile"
        print(f"  {instr}: ~{annualized:.0f} trade/anno stimati — {stato}")

    print("\nCompletato. File: orb_adx_feasibility_filtered.csv, orb_adx_feasibility_unfiltered.csv")


if __name__ == "__main__":
    main()
