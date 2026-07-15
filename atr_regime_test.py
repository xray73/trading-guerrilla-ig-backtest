"""
atr_regime_test.py — Sanity check: con tutti i moltiplicatori a 1.0
(nessuna modulazione), BacktestEngineATRRegime deve essere identica a
BacktestEngineFloatingKillSwitch.
"""

from __future__ import annotations

import engine as eng
import ema_grid_search as g
from engine_floating_kill_switch import BacktestEngineFloatingKillSwitch
from engine_atr_regime import BacktestEngineATRRegime, compute_atr_regime


def main():
    full_data = {
        "DAX": g.load_full_ohlc("DAX_full.csv"),
        "FTSE100": g.load_full_ohlc("FTSE100_full.csv"),
    }

    all_pass = True
    for period in g.PERIODS:
        data_ref = {}
        data_atr = {}
        for name in ["DAX", "FTSE100"]:
            inst = eng.INSTRUMENTS[name]
            window, period_start = g.slice_period(full_data[name], period)
            sig = eng.generate_signals(window, inst)
            sig = g.trim_warmup(sig, period_start)
            data_ref[name] = sig

            sig_atr = compute_atr_regime(sig, window_days=40)
            data_atr[name] = sig_atr

        e_ref = BacktestEngineFloatingKillSwitch(capital0=2000.0)
        t_ref, m_ref = e_ref.run(data_ref)

        e_atr = BacktestEngineATRRegime(capital0=2000.0,
                                         tier_multipliers={"low": 1.0, "medium": 1.0, "high": 1.0})
        t_atr, m_atr = e_atr.run(data_atr)

        n_ref, n_atr = len(t_ref), len(t_atr)
        pnl_ref = float(m_ref["pnl_total"].iloc[0])
        pnl_atr = float(m_atr["pnl_total"].iloc[0])
        match = (n_ref == n_atr) and abs(pnl_ref - pnl_atr) < 1e-6
        status = "OK" if match else "FALLITO"
        print(f"[{period}] {status} — riferimento: {n_ref} trade, pnl={pnl_ref:.2f} | "
              f"regime(1.0x): {n_atr} trade, pnl={pnl_atr:.2f}")
        if not match:
            all_pass = False

    print(f"\n{'='*50}")
    print("TUTTI I TEST PASSATI" if all_pass else "FALLITI — non procedere")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
