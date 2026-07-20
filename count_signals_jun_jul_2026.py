"""
count_signals_jun_jul_2026.py — Conta quanti segnali V6 e MR (RSI) sono
stati generati su DAX/FTSE100 nel periodo 2026-06-01 -> 2026-07-20
(oggi), per verificare se i 2 giorni recenti senza segnale sono in
linea con un periodo più ampio o sono un'anomalia isolata.

Usa ohlc_data_source.py (versione corretta 20/07/2026) per lo storico —
questo run serve anche da verifica pratica che il fix dell'insert D1
funzioni davvero (MAX(timestamp) dovrebbe avanzare oltre il 10/07 dopo
questo run).

Nessuna scrittura su live_positions/live_trades/live_daily_state — SOLO
lettura storico + generate_signals()/generate_mean_reversion_signals()
già esistenti, nessuna modifica alla logica. Output: lista eventi (poche
righe attese) + conteggio aggregato, entrambi piccoli abbastanza per
stare in chat (nessun dato OHLC grezzo in output).
"""

from __future__ import annotations

import os
import pandas as pd

import engine as eng
from mean_reversion_signals import generate_mean_reversion_signals

PERIOD_START = pd.Timestamp("2026-06-01", tz="UTC")
PERIOD_END = pd.Timestamp("2026-07-21", tz="UTC")  # esclusivo, copre fino al 20/07 incluso
SYMBOLS = ["DAX", "FTSE100"]
MR_MODE = "rsi"


def main():
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    token = os.environ.get("CLOUDFLARE_API_TOKEN")
    if not account_id or not token:
        print("ERRORE: CLOUDFLARE_ACCOUNT_ID o CLOUDFLARE_API_TOKEN mancanti.")
        return

    from ohlc_data_source import get_ohlc  # import qui per fallire in modo esplicito se manca

    all_events = []

    for name in SYMBOLS:
        inst = eng.INSTRUMENTS[name]
        print(f"--- {name} ---")
        hist = get_ohlc(name, account_id, token, log=print)

        v6 = eng.generate_signals(hist, inst)
        v6_period = v6[(v6["timestamp"] >= PERIOD_START) & (v6["timestamp"] < PERIOD_END)]
        v6_sig = v6_period[v6_period["signal"].isin(["long", "short"])]
        for _, row in v6_sig.iterrows():
            all_events.append({
                "timestamp": row["timestamp"], "instrument": name, "strategy": "V6",
                "direction": row["signal"], "adx": round(row["adx"], 1),
            })

        mr = generate_mean_reversion_signals(hist, inst, mode=MR_MODE)
        mr_period = mr[(mr["timestamp"] >= PERIOD_START) & (mr["timestamp"] < PERIOD_END)]
        mr_sig = mr_period[mr_period["signal"].isin(["long", "short"])]
        for _, row in mr_sig.iterrows():
            all_events.append({
                "timestamp": row["timestamp"], "instrument": name, "strategy": "MR",
                "direction": row["signal"], "adx": round(row["adx"], 1),
            })

        print(f"  V6: {len(v6_sig)} segnali nel periodo su {len(v6_period)} barre chiuse")
        print(f"  MR: {len(mr_sig)} segnali nel periodo su {len(mr_period)} barre chiuse")

    print(f"\n{'='*60}\nTOTALE eventi 01/06/2026 - 20/07/2026: {len(all_events)}\n{'='*60}")
    if all_events:
        events_df = pd.DataFrame(all_events).sort_values("timestamp")
        for _, ev in events_df.iterrows():
            print(f"  {ev['timestamp'].isoformat()}  [{ev['strategy']}/{ev['instrument']}] "
                  f"{ev['direction'].upper()}  (adx={ev['adx']})")
    else:
        print("  Nessun evento — zero segnali su entrambi gli strumenti/strategie nell'intero periodo.")


if __name__ == "__main__":
    main()
