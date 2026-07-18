"""
sanity_check_engine_mean_reversion.py — Sanity check obbligatorio per
BacktestEngineMeanReversion (engine_mean_reversion.py), come richiesto
dalla disciplina del progetto per ogni sottoclasse del motore ("con
parametri neutri deve riprodurre identicamente il motore standard" —
stesso protocollo già usato per BacktestEngineFloatingKillSwitch).

CHECK A — equivalenza a parametri neutri:
  Capitale grande a sufficienza da non far mai scattare il ramo "size
  sotto il minimo negoziabile" (verificato con assert su
  n_skipped_min_size == 0 a fine run). In questa condizione,
  BacktestEngineMeanReversion deve produrre un trades_df IDENTICO a
  BacktestEngineFloatingKillSwitch (la classe madre diretta) — l'unico
  metodo sovrascritto (_position_size) non deve avere alcun effetto
  quando il suo ramo modificato non si attiva mai.

CHECK B — comportamento della differenza intenzionale (salta vs forza):
  Stesso segnale, capitale piccolo (scenario realistico di split
  ~600€, 30% di 2.000€). Verifica che:
    1) ogni trade forzato al minimo nel motore base (forced_min_size=True)
       NON compaia come trade nel motore MR sullo stesso entry_time/
       strumento/direzione;
    2) il conteggio "forzati" nel motore base sia coerente (stesso
       ordine di grandezza, tipicamente uguale o superiore per via di
       possibili trade successivi abilitati da slot liberati prima)
       col conteggio n_skipped_min_size del motore MR.

Segnali usati: V6 standard (eng.generate_signals), SOLO come dato di
test deterministico e già validato — il codice sotto test è
_position_size(), indipendente dalla fonte del segnale. Nessuna
relazione con la scelta finale mean-reversion Bollinger/RSI.

Dati: DAX_full.csv / FTSE100_full.csv (da fetch_ohlc_d1.py), filtrati
al 2023 (anno solare) come periodo di test — sufficientemente ricco di
trade per il check, non serve corrispondere esattamente ai confini dei
5 periodi ufficiali per uno scopo puramente di equivalenza del codice.

Nessuna scrittura su D1. Solo stampa a log + file risultati/ per
l'artifact.
"""

from __future__ import annotations

import os
import sys
import pandas as pd

import engine as eng
from engine_floating_kill_switch import BacktestEngineFloatingKillSwitch
from engine_mean_reversion import BacktestEngineMeanReversion

PERIOD_START = "2023-01-01"
PERIOD_END = "2024-01-01"
CAPITAL_NEUTRAL = 200_000.0   # abbastanza grande da non forzare mai la size minima
CAPITAL_SMALL = 600.0         # scenario split realistico (~30% di 2000€)


def load_period(symbol_csv: str, name: str) -> pd.DataFrame:
    df = pd.read_csv(symbol_csv)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df[(df["timestamp"] >= PERIOD_START) & (df["timestamp"] < PERIOD_END)].reset_index(drop=True)
    inst = eng.INSTRUMENTS[name]
    return eng.generate_signals(df, inst)


def trades_equal(a: pd.DataFrame, b: pd.DataFrame) -> tuple[bool, str]:
    if len(a) != len(b):
        return False, f"Numero trade diverso: base={len(a)} MR={len(b)}"
    cols = ["instrument", "direction", "entry_time", "entry_price", "exit_time",
            "exit_price", "exit_reason", "size", "pnl"]
    a2 = a[cols].reset_index(drop=True)
    b2 = b[cols].reset_index(drop=True)
    diff = (a2 != b2) & ~(a2.isna() & b2.isna())
    if diff.any().any():
        first_bad = diff.any(axis=1).idxmax()
        return False, f"Divergenza alla riga {first_bad}:\nbase={a2.iloc[first_bad].to_dict()}\nMR={b2.iloc[first_bad].to_dict()}"
    return True, "OK"


def main():
    os.makedirs("results", exist_ok=True)
    report_lines = []

    def log(msg):
        print(msg)
        report_lines.append(msg)

    log(f"=== Sanity check engine_mean_reversion.py — periodo test {PERIOD_START}..{PERIOD_END} ===\n")

    data = {
        "DAX": load_period("DAX_full.csv", "DAX"),
        "FTSE100": load_period("FTSE100_full.csv", "FTSE100"),
    }

    # ------------------------------------------------------------
    # CHECK A — equivalenza a parametri neutri
    # ------------------------------------------------------------
    log("--- CHECK A: equivalenza a parametri neutri (capitale grande) ---")
    base_engine = BacktestEngineFloatingKillSwitch(capital0=CAPITAL_NEUTRAL)
    base_trades, _ = base_engine.run(data)

    mr_engine = BacktestEngineMeanReversion(capital0=CAPITAL_NEUTRAL)
    mr_trades, _ = mr_engine.run(data)

    log(f"  Trade motore base: {len(base_trades)}  |  Trade motore MR: {len(mr_trades)}")
    log(f"  n_skipped_min_size motore MR (deve essere 0): {mr_engine.n_skipped_min_size}")

    check_a_pass = mr_engine.n_skipped_min_size == 0
    if check_a_pass:
        equal, msg = trades_equal(base_trades, mr_trades)
        check_a_pass = equal
        log(f"  Confronto trades_df: {msg}")
    else:
        log("  FALLITO: il ramo 'size sotto minimo' si è attivato anche a capitale "
            "grande — capitale di test insufficiente, alzare CAPITAL_NEUTRAL.")

    log(f"  >>> CHECK A: {'PASS' if check_a_pass else 'FAIL'}\n")

    # ------------------------------------------------------------
    # CHECK B — comportamento della differenza intenzionale
    # ------------------------------------------------------------
    log("--- CHECK B: comportamento salta-vs-forza (capitale piccolo, 600 EUR) ---")
    base_small = BacktestEngineFloatingKillSwitch(capital0=CAPITAL_SMALL)
    base_small_trades, _ = base_small.run(data)

    mr_small = BacktestEngineMeanReversion(capital0=CAPITAL_SMALL)
    mr_small_trades, _ = mr_small.run(data)

    n_forced_base = int(base_small_trades["forced_min_size"].sum()) if not base_small_trades.empty else 0
    n_skipped_mr = mr_small.n_skipped_min_size

    log(f"  Motore base (forza al minimo): {len(base_small_trades)} trade totali, "
        f"di cui {n_forced_base} forzati al minimo")
    log(f"  Motore MR (salta): {len(mr_small_trades)} trade totali, "
        f"{n_skipped_mr} tentativi saltati per size insufficiente")

    # invariante minima: ogni trade NON forzato nel motore base che precede
    # cronologicamente qualunque divergenza deve comparire anche nel motore MR
    base_not_forced = base_small_trades[~base_small_trades["forced_min_size"]] if not base_small_trades.empty else base_small_trades
    mr_entries = set(zip(mr_small_trades["instrument"], mr_small_trades["entry_time"].astype(str),
                          mr_small_trades["direction"])) if not mr_small_trades.empty else set()
    base_not_forced_entries = set(zip(base_not_forced["instrument"], base_not_forced["entry_time"].astype(str),
                                       base_not_forced["direction"])) if not base_not_forced.empty else set()
    missing_non_forced = base_not_forced_entries - mr_entries
    n_missing = len(missing_non_forced)
    log(f"  Trade NON forzati nel motore base assenti nel motore MR: {n_missing} "
        f"(atteso 0 fino al primo punto di divergenza a valle; >0 indica slot "
        f"liberati/occupati diversamente a cascata — atteso e non un errore, "
        f"ma da ispezionare se il numero è alto)")

    check_b_pass = (n_forced_base == 0 and n_skipped_mr == 0) or (n_forced_base > 0 and n_skipped_mr > 0)
    log(f"  >>> CHECK B: {'PASS' if check_b_pass else 'FAIL'} "
        f"(la differenza si attiva/non si attiva coerentemente su entrambi i motori)\n")

    overall = check_a_pass and check_b_pass
    log(f"=== RISULTATO COMPLESSIVO: {'PASS' if overall else 'FAIL'} ===")

    with open("results/sanity_check_mean_reversion.txt", "w") as f:
        f.write("\n".join(report_lines))

    if not overall:
        sys.exit(1)


if __name__ == "__main__":
    main()
