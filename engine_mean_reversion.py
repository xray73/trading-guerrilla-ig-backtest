"""
engine_mean_reversion.py — Sottoclasse isolata di
BacktestEngineFloatingKillSwitch che cambia UN SOLO comportamento
rispetto allo standard: se la size calcolata dal rischio % è sotto la
size minima negoziabile, il trade viene SALTATO invece di forzato al
minimo (decisa in chat 18/07/2026, dopo aver scoperto che su DAX il
mean-reversion forzava la size nel 100% dei trade con capitale ridotto
— il sistema non stava più operando al rischio % previsto).

Questo comportamento è l'OPPOSTO di quello standard di Variante 6
(che forza al minimo per scelta esplicita, RCA sez.15, "nettamente la
migliore" delle 4 alternative confrontate — decisione che resta
INVARIATA per V6). Qui è diverso perché il mean-reversion, con un
capitale tipicamente più piccolo (split), ha una probabilità molto
più alta di over-forzare — la logica "salta" si auto-regola in base
all'ATR del momento senza bisogno di una soglia di capitale fissa
decisa a tavolino (stessa filosofia già usata in
engine_extended_orders.py per gli slot 4-5).

Nessuna modifica a engine.py né a
engine_floating_kill_switch.py — questa classe sovrascrive SOLO
_position_size(), un singolo metodo isolato.

USO: pensata per il motore mean-reversion nello scenario SEPARATO
(capitale/motore indipendente da V6) — il router combinato resta in
pausa (18/07/2026), quindi questa classe non è ancora stata pensata
per quel caso d'uso.
"""

from __future__ import annotations

import engine as eng
from engine_floating_kill_switch import BacktestEngineFloatingKillSwitch


class BacktestEngineMeanReversion(BacktestEngineFloatingKillSwitch):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_skipped_min_size = 0  # diagnostico: quanti trade saltati per size insufficiente

    def _position_size(self, entry_price: float, stop_price: float,
                        inst: eng.InstrumentConfig) -> tuple[float, float, bool, bool]:
        """Identica alla base class FINO al controllo size minima — lì
        SALTA (size=0) invece di forzare. Il controllo margine
        successivo resta invariato (non ancora rilevante se size=0,
        _open_position si ferma comunque su size<=0)."""
        risk_amount = self.capital * inst.risk_pct
        risk_distance = abs(entry_price - stop_price)
        if risk_distance <= 0:
            return 0.0, 0.0, False, False

        size = risk_amount / (risk_distance * inst.point_value)
        if size < inst.min_tradable_size:
            self.n_skipped_min_size += 1
            return 0.0, 0.0, False, False  # SALTA, non forza — unica differenza da BacktestEngine

        # controllo margine invariato (identico alla base class)
        margin_required = size * entry_price * inst.point_value * inst.margin_pct
        margin_reduced = False
        if margin_required > self.capital:
            max_size_by_margin = self.capital / (entry_price * inst.point_value * inst.margin_pct)
            if max_size_by_margin < size:
                size = max(max_size_by_margin, 0.0)
                margin_reduced = True

        return size, risk_amount, False, margin_reduced


if __name__ == "__main__":
    import sys
    print("Questo file va importato come modulo. Vedi mean_reversion_full_pipeline.py per il test completo.")
    sys.exit(0)
