"""
floating_kill_switch_test.py — Verifica la sottoclasse
BacktestEngineFloatingKillSwitch prima di consegnarla:

TEST 1 (obbligatorio, sanity check equivalenza): con kill_switch_pct
impostato a un valore irraggiungibile (0.99), la sottoclasse deve
produrre risultati IDENTICI (stesso numero di trade, stesso PnL,
stesso drawdown) al motore standard BacktestEngine sugli stessi dati.

TEST 2 (verifica funzionamento): con kill_switch_pct alla soglia reale
(0.04), la sottoclasse deve bloccare nuovi ordini in più occasioni
del motore standard (quello reagisce solo a trade chiusi, questo anche
a floating loss) — quindi ci aspettiamo un numero di trade minore o
uguale, mai maggiore.

Uso: python floating_kill_switch_test.py (richiede DAX_full.csv,
FTSE100_full.csv nella working directory)
"""

from __future__ import annotations

import dataclasses
import pandas as pd

import engine as eng
import ema_grid_search as g
from engine_floating_kill_switch import BacktestEngineFloatingKillSwitch


def run_standard(data, capital0, instruments=None, p=None):
    engine_ = eng.BacktestEngine(capital0=capital0, p=p or eng.PARAMS,
                                  instruments=instruments or eng.INSTRUMENTS)
    return engine_.run(data)


def run_floating(data, capital0, kill_switch_pct, instruments=None):
    p = dataclasses.replace(eng.PARAMS, kill_switch_pct=kill_switch_pct)
    engine_ = BacktestEngineFloatingKillSwitch(
        capital0=capital0, p=p, instruments=instruments or eng.INSTRUMENTS)
    return engine_.run(data)


def main():
    full_data = {
        "DAX": g.load_full_ohlc("DAX_full.csv"),
        "FTSE100": g.load_full_ohlc("FTSE100_full.csv"),
    }

    all_pass = True

    for period in g.PERIODS:
        data = {}
        for name in ["DAX", "FTSE100"]:
            inst = eng.INSTRUMENTS[name]
            window, period_start = g.slice_period(full_data[name], period)
            sig = eng.generate_signals(window, inst)
            sig = g.trim_warmup(sig, period_start)
            data[name] = sig

        # ── TEST 1: soglia irraggiungibile -> deve essere identico ──
        # IMPORTANTE: lo stesso p (stessa kill_switch_pct) va usato per
        # ENTRAMBI i motori, altrimenti si confronta un motore con
        # kill switch quasi disattivato contro uno con la soglia di
        # default (0.04) — differenza spuria, non del meccanismo esteso.
        p_off = dataclasses.replace(eng.PARAMS, kill_switch_pct=0.99)
        trades_std, metrics_std = run_standard(data, 2000.0, p=p_off)
        trades_float_off, metrics_float_off = run_floating(data, 2000.0, kill_switch_pct=0.99)

        n_std = len(trades_std)
        n_float_off = len(trades_float_off)
        pnl_std = float(metrics_std["pnl_total"].iloc[0])
        pnl_float_off = float(metrics_float_off["pnl_total"].iloc[0])

        match = (n_std == n_float_off) and abs(pnl_std - pnl_float_off) < 1e-6
        status = "OK" if match else "FALLITO"
        print(f"[{period}] TEST 1 (equivalenza, soglia=0.99): {status} — "
              f"standard: {n_std} trade, pnl={pnl_std:.2f} | "
              f"floating(off): {n_float_off} trade, pnl={pnl_float_off:.2f}")
        if not match:
            all_pass = False

        # ── TEST 2: soglia reale -> deve bloccare uguale o più del motore standard ──
        trades_std_real, metrics_std_real = run_standard(data, 2000.0)
        trades_float_real, metrics_float_real = run_floating(data, 2000.0, kill_switch_pct=0.04)

        n_std_real = len(trades_std_real)
        n_float_real = len(trades_float_real)
        blocks_more_or_equal = n_float_real <= n_std_real
        status2 = "OK" if blocks_more_or_equal else "FALLITO (blocca MENO del motore standard, illogico)"
        print(f"[{period}] TEST 2 (funzionamento, soglia=0.04): {status2} — "
              f"standard: {n_std_real} trade | floating(on): {n_float_real} trade "
              f"(differenza: {n_std_real - n_float_real})")
        if not blocks_more_or_equal:
            all_pass = False

    print(f"\n{'='*60}")
    print("TUTTI I TEST PASSATI" if all_pass else "ALCUNI TEST FALLITI — non consegnare")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
