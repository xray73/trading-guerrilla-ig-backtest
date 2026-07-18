"""
engine_orario_filtro.py — Sottoclasse isolata di
BacktestEngineFloatingKillSwitch che aggiunge UN SOLO comportamento:
blocca l'apertura di NUOVE posizioni se l'ora UTC della barra di
ingresso rientra in blocked_hours (default: 20,21,22,23 — la "zona
oraria morta" scoperta il 18/07/2026 tramite esplorazione sui dati
grezzi, poi confermata col meccanismo di trading reale — stop/target
fissi, non persistenza a punto fisso — su entrambi DAX e FTSE100,
sotto la soglia di pareggio).

DECISIONE ESPLICITA (18/07/2026): stessa fascia oraria per ENTRAMBI
gli strumenti, anche se la scomposizione ora per ora suggeriva fasce
leggermente diverse tra DAX e FTSE100 — scartato per evitare di
inseguire rumore su campioni orari troppo piccoli (es. l'ora 21 aveva
solo 12-17 osservazioni, troppo poche per fidarsene). Meno parametri
liberi, coerente con la disciplina "la complessità fallisce quasi
sempre out-of-sample" già validata più volte in questo progetto.

COMPORTAMENTO: blocca SOLO nuovi ingressi. Le posizioni già aperte
prima dell'inizio della fascia bloccata continuano a essere gestite
normalmente da stop/target/max holding (stessa filosofia del kill
switch — mai chiusure forzate).

Nessuna modifica a engine.py né a engine_floating_kill_switch.py —
override di un solo metodo, _open_position().

SANITY CHECK (in fondo al file, eseguito da
engine_orario_filtro_test.py): con blocked_hours=set() (vuoto),
questa sottoclasse deve produrre risultati IDENTICI al motore
standard (BacktestEngineFloatingKillSwitch) — stesso protocollo già
usato per ogni altra sottoclasse del motore in questo progetto.
"""

from __future__ import annotations

from engine_floating_kill_switch import BacktestEngineFloatingKillSwitch

# Zona oraria morta scoperta il 18/07/2026, stessa fascia per DAX e FTSE100
# (decisione esplicita, vedi docstring sopra)
DEFAULT_BLOCKED_HOURS_UTC = {20, 21, 22, 23}


class BacktestEngineOrarioFiltro(BacktestEngineFloatingKillSwitch):

    def __init__(self, *args, blocked_hours: set[int] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.blocked_hours = blocked_hours if blocked_hours is not None else set(DEFAULT_BLOCKED_HOURS_UTC)
        self.n_blocked_by_hour = 0  # diagnostico: quanti ingressi saltati per fascia oraria

    def _open_position(self, instrument: str, direction: str, bar, atr_at_entry: float, adx_at_entry: float):
        """Identica alla base class, con un solo controllo aggiuntivo
        PRIMA di aprire: se l'ora UTC della barra di ingresso è
        bloccata, salta silenziosamente (nessuna posizione aperta,
        nessun incremento di orders_today — esattamente come se non
        ci fosse stato nessun segnale su quella barra)."""
        entry_hour = bar["timestamp"].hour
        if entry_hour in self.blocked_hours:
            self.n_blocked_by_hour += 1
            return
        super()._open_position(instrument, direction, bar, atr_at_entry, adx_at_entry)


if __name__ == "__main__":
    import sys
    print("Questo file va importato come modulo. Vedi engine_orario_filtro_test.py "
          "per il sanity check e il test di impatto sui 5 periodi ufficiali.")
    sys.exit(0)
