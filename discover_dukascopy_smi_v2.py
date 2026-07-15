"""
discover_dukascopy_smi_v2.py — Il primo giro ha trovato solo azioni
svizzere singole (INSTRUMENT_SWITZERLAND_*), non l'indice SMI: "SWITZ"
matchava il prefisso sbagliato. Qui filtriamo SOLO le costanti indice
(IDX/INDICES), poi cerchiamo Svizzera/SMI/Europa dentro quella lista
ristretta — stesso approccio che ha funzionato per ITALY40
(INSTRUMENT_IDX_EUROPE_ITA_IDX_EUR).
"""

import dukascopy_python.instruments as instr

all_attrs = [a for a in dir(instr) if a.startswith("INSTRUMENT_")]

idx_attrs = [a for a in all_attrs if "IDX" in a.upper() or "INDICES" in a.upper()]
print(f"=== Tutte le costanti indice (IDX/INDICES) trovate: {len(idx_attrs)} ===\n")
for a in sorted(idx_attrs):
    print(" ", a, "=", getattr(instr, a))

print("\n=== Filtro su Svizzera/SMI dentro la lista indici ===")
swiss_idx = [a for a in idx_attrs if any(t in a.upper() for t in ["SWI", "SUI", "SMI", "CHE"])]
if swiss_idx:
    for a in swiss_idx:
        print(" ", a, "=", getattr(instr, a))
else:
    print("  Nessun match diretto — stampo l'elenco EUROPE_* per cercare a occhio:")
    europe_idx = [a for a in idx_attrs if "EUROPE" in a.upper()]
    for a in sorted(europe_idx):
        print("   ", a, "=", getattr(instr, a))

print("\nCompletato — nessun dato scaricato, nessuna scrittura su D1.")
