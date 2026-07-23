"""
discover_dukascopy_eurusd.py — Segue lo stesso approccio cauto di
discover_dukascopy_smi_v2.py: NON si costruisce il nome della costante
Dukascopy a partire da input utente. Si filtra prima la lista di
costanti FX (FX_MAJORS/FOREX), poi si cerca EUR/USD dentro quella
lista ristretta, e si stampa il valore reale della costante trovata
per verifica visiva prima di usarla in SYMBOL_MAP.

Nota: GBPUSD e' stato aggiunto in precedenza usando
INSTRUMENT_FX_MAJORS_GBP_USD trovato via documentazione ufficiale,
MAI verificato con un discover dedicato. Questo script chiude quel
gap anche per GBPUSD, oltre a confermare EURUSD.

Nessun dato scaricato, nessuna scrittura su D1 — solo enumerazione
delle costanti disponibili nel pacchetto dukascopy-python installato.
"""

import dukascopy_python.instruments as instr

all_attrs = [a for a in dir(instr) if a.startswith("INSTRUMENT_")]

fx_attrs = [a for a in all_attrs if "FX" in a.upper() or "FOREX" in a.upper()]
print(f"=== Tutte le costanti FX trovate: {len(fx_attrs)} ===\n")
for a in sorted(fx_attrs):
    print(" ", a, "=", getattr(instr, a))

print("\n=== Filtro su EUR/USD dentro la lista FX ===")
eurusd_matches = [a for a in fx_attrs if "EUR" in a.upper() and "USD" in a.upper()]
if eurusd_matches:
    for a in eurusd_matches:
        print(" ", a, "=", getattr(instr, a))
else:
    print("  Nessun match diretto — nessuna costante EUR/USD trovata nel pacchetto.")

print("\n=== Verifica incrociata: conferma GBP/USD gia' in uso (non ancora verificata via discover) ===")
gbpusd_matches = [a for a in fx_attrs if "GBP" in a.upper() and "USD" in a.upper()]
for a in gbpusd_matches:
    print(" ", a, "=", getattr(instr, a))

print("\nCompletato — nessun dato scaricato, nessuna scrittura su D1.")
