"""
discover_dukascopy_instrument.py — Trova il nome esatto della costante
per uno strumento in dukascopy-python, senza scaricare nessun dato e
senza toccare D1. Puramente di scoperta, sicuro da eseguire.

Motivo: la documentazione pubblica della libreria mostra solo esempi
con GBP_USD — non c'è un elenco pubblico affidabile delle costanti per
gli indici. Indovinare il nome rischierebbe di caricare dati sbagliati
(o nessun dato) in D1. Meglio ispezionare il modulo direttamente.
"""

import dukascopy_python
import dukascopy_python.instruments as instr

SEARCH_TERMS = ["ITA", "MIB", "IT40", "ITALY"]

print("=== Ricerca costanti strumento che contengono:", SEARCH_TERMS, "===\n")

all_attrs = [a for a in dir(instr) if a.startswith("INSTRUMENT_")]
print(f"Totale costanti INSTRUMENT_* trovate nel modulo: {len(all_attrs)}\n")

matches = []
for term in SEARCH_TERMS:
    found = [a for a in all_attrs if term in a.upper()]
    if found:
        print(f"Match per '{term}': {found}")
        matches.extend(found)

matches = sorted(set(matches))

if not matches:
    print("\nNESSUN match trovato per Italy/MIB. Stampo tutte le costanti "
          "INSTRUMENT_INDICES_* disponibili, per cercare a occhio:")
    indices_attrs = [a for a in all_attrs if "INDICES" in a.upper() or "INDEX" in a.upper()]
    for a in sorted(indices_attrs):
        print(" ", a, "=", getattr(instr, a))
else:
    print(f"\n=== CANDIDATI TROVATI ({len(matches)}) ===")
    for m in matches:
        print(f"  {m} = {getattr(instr, m)!r}")

print("\n=== Costanti INTERVAL_* disponibili (per confermare 30 minuti) ===")
interval_attrs = [a for a in dir(dukascopy_python) if a.startswith("INTERVAL_")]
for a in sorted(interval_attrs):
    print(" ", a, "=", getattr(dukascopy_python, a))

print("\nCompletato — nessun dato scaricato, nessuna scrittura su D1.")
