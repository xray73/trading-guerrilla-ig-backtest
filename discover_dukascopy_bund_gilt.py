"""
discover_dukascopy_bund_gilt.py — Stesso approccio cauto degli altri
discover script del progetto (SMI, EURUSD): NON si costruisce il nome
della costante Dukascopy a partire da input utente. Si filtra prima la
lista di costanti relative a bond/tassi/futures, poi si cerca Bund/Gilt
dentro quella lista ristretta.

Contesto (24/07/2026): primo passo per esplorare i futures obbligazionari
(Bund tedesco, Gilt britannico) come possibile nuova classe di asset —
teoricamente il terreno piu' favorevole al trend-following secondo la
letteratura CTA, e la scelta a minor rischio tra le opzioni considerate
(energia, agricole, crypto) per la natura piu' graduale dei movimenti
guidati da tassi/banche centrali.

Nessun dato scaricato, nessuna scrittura su D1 — solo enumerazione delle
costanti disponibili nel pacchetto dukascopy-python installato.
"""

import dukascopy_python.instruments as instr

all_attrs = [a for a in dir(instr) if a.startswith("INSTRUMENT_")]

# Filtro largo: bond, tassi, futures governativi
keywords = ["BOND", "BUND", "GILT", "RATE", "TREASURY", "NOTE", "YIELD"]
bond_attrs = [a for a in all_attrs if any(k in a.upper() for k in keywords)]

print(f"=== Costanti relative a bond/tassi trovate: {len(bond_attrs)} ===\n")
for a in sorted(bond_attrs):
    print(" ", a, "=", getattr(instr, a))

print("\n=== Filtro su BUND (Germania) ===")
bund_matches = [a for a in bond_attrs if "BUND" in a.upper()]
if bund_matches:
    for a in bund_matches:
        print(" ", a, "=", getattr(instr, a))
else:
    print("  Nessun match diretto per BUND.")

print("\n=== Filtro su GILT (UK) ===")
gilt_matches = [a for a in bond_attrs if "GILT" in a.upper()]
if gilt_matches:
    for a in gilt_matches:
        print(" ", a, "=", getattr(instr, a))
else:
    print("  Nessun match diretto per GILT.")

print("\nCompletato — nessun dato scaricato, nessuna scrittura su D1.")
