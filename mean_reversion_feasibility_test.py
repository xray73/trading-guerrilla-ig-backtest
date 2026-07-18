"""
mean_reversion_feasibility_test.py — Primo test sul mean-reversion,
criteri fissati PRIMA di vedere risultati (17/07/2026):

  1. FATTIBILITÀ: quanti trade/anno genera ciascuna variante
     (Bollinger vs RSI)? Soglia 30 trade/anno/strumento, stessa
     usata per ORB.
  2. SOVRAPPOSIZIONE CON V6: quanti giorni avrebbero segnali attivi
     SIA da V6 SIA dal mean-reversion sullo stesso strumento? Dovrebbe
     essere vicino a zero se il filtro ADX<20/ADX>=20 funziona come
     inteso (i due motori si dividono i regimi di mercato). Se la
     sovrapposizione è alta, il filtro non sta facendo il suo lavoro.

NON è ancora un test di validazione vero (serve sanity check + i 5
periodi ufficiali + stima del rumore, stesso protocollo di sempre) —
è il primo controllo via/no-via.

Periodo di test: ultimi 180 giorni (stesso campione già usato per gli
altri test descrittivi, comodo per un primo sguardo, NON uno dei 5
periodi ufficiali).

Capitale/rischio: CAPITAL0 usato qui SOLO per far girare il motore
standard e ottenere metriche — non è una decisione definitiva
sull'allocazione (ancora aperta, da decidere separatamente).
"""

from __future__ import annotations

from datetime import datetime, timedelta
import pandas as pd

import dukascopy_python
from dukascopy_python.instruments import INSTRUMENT_IDX_EUROPE_E_DAAX, INSTRUMENT_IDX_EUROPE_E_FUTSEE_100

import engine as eng
from engine_floating_kill_switch import BacktestEngineFloatingKillSwitch
from mean_reversion_signals import generate_mean_reversion_signals

WARMUP_DAYS = 30
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


def run_variant(signal_data: dict, label: str) -> pd.DataFrame:
    engine_ = BacktestEngineFloatingKillSwitch(capital0=CAPITAL0)
    trades_df, _ = engine_.run(signal_data)

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

    # segnali V6 (invariato) per il confronto di sovrapposizione
    v6_signal_data = {}
    for name in SYMBOLS:
        inst = eng.INSTRUMENTS[name]
        v6_signal_data[name] = eng.generate_signals(raw_data[name], inst)
    trades_v6 = run_variant(v6_signal_data, "Variante 6 (riferimento)")

    all_results = {}
    for mode, label in [("bollinger", "Mean-reversion BOLLINGER"), ("rsi", "Mean-reversion RSI")]:
        signal_data = {}
        for name in SYMBOLS:
            inst = eng.INSTRUMENTS[name]
            signal_data[name] = generate_mean_reversion_signals(raw_data[name], inst, mode=mode)
        trades = run_variant(signal_data, label)
        all_results[mode] = trades
        trades.to_csv(f"mean_reversion_{mode}_feasibility.csv", index=False)

    # sovrapposizione con V6: posizioni APERTE CONTEMPORANEAMENTE (non
    # "stesso giorno" — l'ADX cambia nel corso della giornata, quindi
    # V6 al mattino e mean-reversion al pomeriggio sullo stesso giorno
    # NON è un conflitto, è il mercato che cambia regime, comportamento
    # atteso. Quello che conta davvero è se le due posizioni
    # occuperebbero lo stesso slot/capitale nello stesso istante.
    print(f"\n=== Sovrapposizione temporale con Variante 6 (posizioni aperte insieme, stesso strumento) ===")
    print("(NON 'stesso giorno' — un conflitto a livello di barra è impossibile per costruzione, "
          "dato che ADX in una barra è sopra O sotto 20, mai entrambi)")

    def to_intervals(trades_df: pd.DataFrame) -> list[tuple]:
        if trades_df.empty:
            return []
        return list(zip(
            trades_df["instrument"],
            pd.to_datetime(trades_df["entry_time"], utc=True),
            pd.to_datetime(trades_df["exit_time"], utc=True),
        ))

    v6_intervals = to_intervals(trades_v6)

    for mode, trades in all_results.items():
        if trades.empty:
            print(f"  {mode}: nessun trade, sovrapposizione non calcolabile.")
            continue
        mr_intervals = to_intervals(trades)

        n_overlap = 0
        for mr_instr, mr_start, mr_end in mr_intervals:
            for v6_instr, v6_start, v6_end in v6_intervals:
                if mr_instr != v6_instr:
                    continue
                # due intervalli si sovrappongono se uno inizia prima che l'altro finisca, in entrambi i sensi
                if mr_start < v6_end and v6_start < mr_end:
                    n_overlap += 1
                    break  # basta un conflitto per contare questo trade come sovrapposto

        pct = 100 * n_overlap / len(mr_intervals) if mr_intervals else 0
        print(f"  {mode}: {n_overlap}/{len(mr_intervals)} trade con posizione V6 aperta nello stesso momento "
              f"({pct:.1f}%)")

    print(f"\n=== Verdetto fattibilità (soglia empirica: 30 trade/anno/strumento) ===")
    for mode, trades in all_results.items():
        print(f"\n{mode}:")
        for instr in SYMBOLS:
            n = (trades["instrument"] == instr).sum() if not trades.empty else 0
            annualized = n * 365 / DAYS_BACK
            stato = "OK" if annualized >= 30 else "SOTTO SOGLIA"
            print(f"  {instr}: ~{annualized:.0f} trade/anno stimati — {stato}")

    print("\nCompletato. File: mean_reversion_bollinger_feasibility.csv, mean_reversion_rsi_feasibility.csv")


if __name__ == "__main__":
    main()
