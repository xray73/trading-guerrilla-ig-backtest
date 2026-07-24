"""
discover_dukascopy_bund_future.py — Follow-up a discover_dukascopy_bund_gilt.py
(24/07/2026): quello script ha trovato solo BND_CFD_BUND_TR_EUR ("Total
Return" CFD, scala ~144), ma IG offre il Bund SOLO come Future (scala
~12.446, verificato su IG demo — screenshot utente). Sono probabilmente
due prodotti diversi con struttura di prezzo diversa, non lo stesso
strumento in unita' diverse.

Questo script allarga la ricerca includendo "FUT" tra le keyword, per
vedere se Dukascopy ha anche il future vero (che corrisponderebbe al
prodotto IG), non solo il CFD Total Return trovato prima.

Nessun dato scaricato, nessuna scrittura su D1.
"""

import dukascopy_python.instruments as instr

all_attrs = [a for a in dir(instr) if a.startswith("INSTRUMENT_")]

keywords = ["BOND", "BUND", "GILT", "RATE", "TREASURY", "NOTE", "YIELD", "FUT", "DBUND", "SCHATZ", "BOBL"]
bond_attrs = [a for a in all_attrs if any(k in a.upper() for k in keywords)]

print(f"=== Costanti relative a bond/tassi/futures trovate: {len(bond_attrs)} ===\n")
for a in sorted(bond_attrs):
    print(" ", a, "=", getattr(instr, a))

print("\n=== Filtro esplicito su BUND ===")
bund_matches = [a for a in bond_attrs if "BUND" in a.upper() or "DBUND" in a.upper()]
for a in bund_matches:
    print(" ", a, "=", getattr(instr, a))

print("\nCompletato — nessun dato scaricato, nessuna scrittura su D1.")
