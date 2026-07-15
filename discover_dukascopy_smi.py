"""
discover_dukascopy_smi.py — Trova il nome esatto della costante per
SMI/Switzerland Blue Chip in dukascopy-python. Nessun dato scaricato,
nessuna scrittura su D1. Stesso approccio già usato per ITALY40.
"""

import dukascopy_python
import dukascopy_python.instruments as instr

SEARCH_TERMS = ["SMI", "SWISS", "SWITZ", "CH20", "SUI"]

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
    print("\nNESSUN match diretto. Stampo tutte le costanti "
          "INSTRUMENT_INDICES_* / INSTRUMENT_IDX_* disponibili, per cercare a occhio:")
    indices_attrs = [a for a in all_attrs if "INDICES" in a.upper() or "IDX" in a.upper()
                      or "INDEX" in a.upper()]
    for a in sorted(indices_attrs):
        print(" ", a, "=", getattr(instr, a))
else:
    print(f"\n=== CANDIDATI TROVATI ({len(matches)}) ===")
    for m in matches:
        print(f"  {m} = {getattr(instr, m)!r}")

print("\nCompletato — nessun dato scaricato, nessuna scrittura su D1.")
