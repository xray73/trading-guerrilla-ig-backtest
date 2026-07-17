"""
discover_dukascopy_cac40.py — Trova il nome esatto della costante
dukascopy_python per il CAC40 (Francia), necessaria prima di poter
testare la cointegrazione DAX-CAC40. Nessun nome plausibile trovato
con certezza via ricerca web, quindi lo verifichiamo direttamente
ispezionando il modulo invece di indovinare.

Cerca tra tutti gli attributi di dukascopy_python.instruments quelli
che contengono 'CAC', 'FRA', 'FR40', 'FR_40' (case-insensitive) —
pattern osservati per gli altri indici europei: INSTRUMENT_IDX_EUROPE_E_DAAX
(DAX), INSTRUMENT_IDX_EUROPE_E_FUTSEE_100 (FTSE100).
"""

import dukascopy_python.instruments as inst_module

SEARCH_TERMS = ["CAC", "FRA", "FR40", "FR_40", "FRANCE"]

print("=== Ricerca costante CAC40 in dukascopy_python.instruments ===\n")

all_attrs = [a for a in dir(inst_module) if a.startswith("INSTRUMENT_")]
print(f"Totale costanti INSTRUMENT_* nel modulo: {len(all_attrs)}\n")

matches = []
for attr in all_attrs:
    if any(term in attr.upper() for term in SEARCH_TERMS):
        matches.append(attr)

if matches:
    print("Possibili corrispondenze trovate:")
    for m in matches:
        print(f"  {m} = {getattr(inst_module, m)!r}")
else:
    print("NESSUNA corrispondenza trovata con i termini di ricerca.")
    print("\nStampo tutte le costanti IDX_EUROPE per ispezione manuale:")
    for attr in all_attrs:
        if "IDX_EUROPE" in attr:
            print(f"  {attr} = {getattr(inst_module, attr)!r}")

print("\n=== Fine ricerca. ===")
