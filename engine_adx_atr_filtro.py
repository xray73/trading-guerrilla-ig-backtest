"""
engine_adx_atr_filtro.py — Sottoclasse isolata di
BacktestEngineFloatingKillSwitch che blocca l'apertura di NUOVE
posizioni in base a una combinazione ADX/volatilità (ATR), con regole
DIVERSE per DAX e FTSE100 — decisione esplicita 18/07/2026, basata
sull'asimmetria scoperta esplorando i dati grezzi e confermata col
meccanismo di trading reale (stop/target fissi, non persistenza a
punto fisso):

  DAX: salta se ADX > 30 E ATR% >= 0,25 (la COMBINAZIONE è pericolosa
       — ADX alto da solo, con volatilità contenuta, resta affidabile
       per il DAX, anzi è il caso migliore trovato).
  FTSE100: salta se ADX > 40, INDIPENDENTEMENTE dalla volatilità —
       per il FTSE100 l'ADX estremo è debole anche a bassa volatilità.

Soglie riprese ESATTAMENTE da quelle già esplorate nella sessione del
18/07/2026, non raffinate ulteriormente — raffinarle rischierebbe di
adattarle al rumore invece che al pattern reale.

AVVERTENZA ESPLICITA (18/07/2026): questo filtro ha più parametri
liberi (2 per DAX, 1 per FTSE100) del filtro orario appena bocciato —
rischio di overfitting più alto, atteso e accettato prima di vedere
i risultati.

COMPORTAMENTO: blocca SOLO nuovi ingressi. Le posizioni già aperte
continuano a essere gestite normalmente da stop/target/max holding.

Nessuna modifica a engine.py né a engine_floating_kill_switch.py —
override di un solo metodo, _open_position().

SANITY CHECK (in engine_adx_atr_filtro_test.py): con soglie
irraggiungibili (ADX>999 per entrambi), deve produrre risultati
IDENTICI al motore standard — stessa convenzione già usata per
BacktestEngineFloatingKillSwitch.
"""

from __future__ import annotations

from engine_floating_kill_switch import BacktestEngineFloatingKillSwitch

DEFAULT_DAX_ADX_THRESHOLD = 30.0
DEFAULT_DAX_ATR_THRESHOLD_PCT = 0.25
DEFAULT_FTSE_ADX_THRESHOLD = 40.0


class BacktestEngineADXATRFiltro(BacktestEngineFloatingKillSwitch):

    def __init__(self, *args,
                 dax_adx_threshold: float = DEFAULT_DAX_ADX_THRESHOLD,
                 dax_atr_threshold_pct: float = DEFAULT_DAX_ATR_THRESHOLD_PCT,
                 ftse_adx_threshold: float = DEFAULT_FTSE_ADX_THRESHOLD,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.dax_adx_threshold = dax_adx_threshold
        self.dax_atr_threshold_pct = dax_atr_threshold_pct
        self.ftse_adx_threshold = ftse_adx_threshold
        self.n_blocked_dax = 0    # diagnostico
        self.n_blocked_ftse = 0   # diagnostico

    def _should_block(self, instrument: str, bar, atr_at_entry: float, adx_at_entry: float) -> bool:
        if instrument == "DAX":
            if adx_at_entry <= self.dax_adx_threshold:
                return False
            atr_pct = (atr_at_entry / bar["close"]) * 100.0
            return atr_pct >= self.dax_atr_threshold_pct
        elif instrument == "FTSE100":
            return adx_at_entry > self.ftse_adx_threshold
        return False

    def _open_position(self, instrument: str, direction: str, bar, atr_at_entry: float, adx_at_entry: float):
        """Identica alla base class, con un controllo aggiuntivo PRIMA
        di aprire: se la combinazione ADX/ATR (regola specifica per
        strumento) è nella zona bloccata, salta silenziosamente —
        nessuna posizione aperta, nessun incremento di orders_today."""
        if self._should_block(instrument, bar, atr_at_entry, adx_at_entry):
            if instrument == "DAX":
                self.n_blocked_dax += 1
            else:
                self.n_blocked_ftse += 1
            return
        super()._open_position(instrument, direction, bar, atr_at_entry, adx_at_entry)


if __name__ == "__main__":
    import sys
    print("Questo file va importato come modulo. Vedi engine_adx_atr_filtro_test.py "
          "per il sanity check e il test di impatto sui 5 periodi ufficiali.")
    sys.exit(0)
