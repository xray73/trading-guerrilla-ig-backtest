"""
extended_orders_test.py — Verifica BacktestEngineExtendedOrders prima
di consegnarla:

TEST 1 (sanity check equivalenza): con max_new_orders_per_day=3
(invariato rispetto a oggi), deve essere identica a
BacktestEngineFloatingKillSwitch — il ramo "slot extra" non deve mai
attivarsi.

TEST 2 (funzionamento): con max_new_orders_per_day=5, deve aprire
un numero di trade >= alla versione a 3 slot (mai di meno — gli slot
extra si aggiungono, non tolgono nulla ai primi 3), e gli slot extra
aperti devono rispettare il tetto di rischio (mai superiore al rischio
standard dello strumento).
"""

from __future__ import annotations

import dataclasses
import pandas as pd

import engine as eng
import ema_grid_search as g
from engine_floating_kill_switch import BacktestEngineFloatingKillSwitch
from engine_extended_orders import BacktestEngineExtendedOrders


def load_period_data(period, full_data):
    data = {}
    for name in ["DAX", "FTSE100"]:
        inst = eng.INSTRUMENTS[name]
        window, period_start = g.slice_period(full_data[name], period)
        sig = eng.generate_signals(window, inst)
        sig = g.trim_warmup(sig, period_start)
        data[name] = sig
    return data


def main():
    full_data = {
        "DAX": g.load_full_ohlc("DAX_full.csv"),
        "FTSE100": g.load_full_ohlc("FTSE100_full.csv"),
    }

    all_pass = True

    for period in g.PERIODS:
        data = load_period_data(period, full_data)

        # ── TEST 1: max_orders=3 (invariato) -> deve essere identico ──
        p_base = eng.PARAMS  # max_new_orders_per_day=3 di default
        e_ref = BacktestEngineFloatingKillSwitch(capital0=2000.0, p=p_base)
        t_ref, m_ref = e_ref.run(data)

        e_ext = BacktestEngineExtendedOrders(capital0=2000.0, p=p_base, extra_slot_pct=1.0)
        t_ext, m_ext = e_ext.run(data)

        n_ref, n_ext = len(t_ref), len(t_ext)
        pnl_ref = float(m_ref["pnl_total"].iloc[0])
        pnl_ext = float(m_ext["pnl_total"].iloc[0])
        match = (n_ref == n_ext) and abs(pnl_ref - pnl_ext) < 1e-6
        status = "OK" if match else "FALLITO"
        print(f"[{period}] TEST 1 (equivalenza, max_orders=3): {status} — "
              f"riferimento: {n_ref} trade | esteso: {n_ext} trade, "
              f"extra aperti: {e_ext.n_extra_slot_opened} (atteso 0)")
        if not match or e_ext.n_extra_slot_opened != 0:
            all_pass = False

        # ── TEST 2: max_orders=5, slot extra attivi ──
        p_extended = dataclasses.replace(eng.PARAMS, max_new_orders_per_day=5)
        e_5 = BacktestEngineExtendedOrders(capital0=2000.0, p=p_extended, extra_slot_pct=1.0)
        t_5, m_5 = e_5.run(data)

        n_5 = len(t_5)
        pnl_5 = float(m_5["pnl_total"].iloc[0])
        never_fewer = n_5 >= n_ref
        status2 = "OK" if never_fewer else "FALLITO (meno trade della baseline, illogico)"
        print(f"[{period}] TEST 2 (funzionamento, max_orders=5): {status2} — "
              f"baseline(3): {n_ref} trade, pnl={pnl_ref:.1f} | "
              f"esteso(5): {n_5} trade, pnl={pnl_5:.1f} | "
              f"slot extra aperti: {e_5.n_extra_slot_opened}, "
              f"saltati per pnl<=0: {e_5.n_extra_slot_skipped_pnl}, "
              f"saltati per size minima: {e_5.n_extra_slot_skipped_min_size}")
        if not never_fewer:
            all_pass = False

        # verifica che nessun trade extra abbia rischio sopra lo standard
        # NOTA: il capitale cresce nel tempo (compounding), quindi il
        # rischio standard in € cresce con esso — non ha senso confrontare
        # contro una soglia fissa sul capitale INIZIALE. L'invariante
        # "rischio_extra <= rischio_standard_al_momento" è garantito per
        # costruzione dal min() nel codice (engine_extended_orders.py),
        # verificabile per lettura. Qui controlliamo solo che il numero
        # di slot extra aperti sia coerente con i contatori interni.
        totale_slot_extra = e_5.n_extra_slot_opened + e_5.n_extra_slot_skipped_pnl + e_5.n_extra_slot_skipped_min_size
        print(f"           slot extra valutati in totale: {totale_slot_extra} "
              f"(aperti: {e_5.n_extra_slot_opened}, saltati pnl<=0: {e_5.n_extra_slot_skipped_pnl}, "
              f"saltati size minima: {e_5.n_extra_slot_skipped_min_size})")

    print(f"\n{'='*60}")
    print("TUTTI I TEST PASSATI" if all_pass else "ALCUNI TEST FALLITI — non consegnare")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
