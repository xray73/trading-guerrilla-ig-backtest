"""
combined_router_signals.py — Router esplicito che unisce Variante 6 e
mean-reversion in un'UNICA colonna segnale, da dare in pasto a UN SOLO
motore (capitale, kill switch, slot condivisi davvero) — a differenza
del test di fattibilità (17/07/2026), dove i due motori giravano
separati, ciascuno con il proprio capitale indipendente.

Logica del router, barra per barra:
  - ADX(14) >= 20  -> usa il segnale di Variante 6 (se presente)
  - ADX(14) < 20   -> usa il segnale del mean-reversion (se presente)
  - Nessuno dei due attivo su quella barra -> nessun segnale

Dato che entrambi i moduli calcolano l'ADX con la STESSA formula
Wilder (riusata da engine.py in entrambi i casi), il valore di soglia
è identico ovunque — non c'è ambiguità su quale motore "ha diritto di
parlare" in un dato momento.

Nessuna modifica a engine.py, engine_floating_kill_switch.py, né ai
moduli di segnale già esistenti — questo file li combina, non li
altera.
"""

from __future__ import annotations

import pandas as pd

import engine as eng
from mean_reversion_signals import generate_mean_reversion_signals, ADX_THRESHOLD


def generate_combined_signals(raw_df: pd.DataFrame, inst: eng.InstrumentConfig,
                               mr_mode: str = "rsi") -> pd.DataFrame:
    """raw_df: colonne timestamp, open, high, low, close (dati grezzi,
    non ancora processati). Ritorna un unico dataframe con colonna
    'signal' che alterna tra V6 e mean-reversion in base all'ADX."""
    v6_signals = eng.generate_signals(raw_df, inst)
    mr_signals = generate_mean_reversion_signals(raw_df, inst, mode=mr_mode)

    # entrambi calcolati sugli stessi timestamp/OHLC di partenza, quindi
    # allineati riga per riga senza bisogno di merge per timestamp
    combined = v6_signals.copy()
    regime_trend = combined["adx"] >= ADX_THRESHOLD

    # dove il regime è "laterale" (ADX<20), sovrascrivo con il segnale mean-reversion
    combined.loc[~regime_trend, "signal"] = mr_signals.loc[~regime_trend, "signal"]

    # traccia diagnostica: quale motore ha "parlato" su ciascuna barra con segnale
    combined["fonte_segnale"] = None
    combined.loc[regime_trend & combined["signal"].notna(), "fonte_segnale"] = "V6"
    combined.loc[~regime_trend & combined["signal"].notna(), "fonte_segnale"] = f"mean_reversion_{mr_mode}"

    return combined


if __name__ == "__main__":
    import sys
    print("Questo file va importato come modulo. Vedi router_combined_test.py per il test.")
    sys.exit(0)
