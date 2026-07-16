"""
engine_accantonamento.py — Estensione isolata di
BacktestEngineFloatingKillSwitch che aggiunge un meccanismo di
accantonamento periodico ("opzione 3 mensile", decisa in chat
16/07/2026): ogni volta che il capitale investito supera per la prima
volta +50% rispetto all'ultimo riferimento fissato, il 40% del
guadagno sopra quella soglia esce PERMANENTEMENTE dal capitale a
rischio e va in un "accantonato" (side_pool) — non più investito, non
più rischiato, disponibile per prelievo (in Fase 4) o semplicemente
come cuscinetto di sicurezza.

Il check avviene UNA volta al mese (al cambio di mese di calendario),
non ad ogni trade — decisione presa in chat dopo aver verificato che
il check mensile batte sempre quello continuo in rendimento finale, su
tutti i 5 periodi storici testati (16/07/2026).

DESIGN — perché self.capital rappresenta l'investito, non il totale:
Questa classe NON tocca `_position_size`, `_open_position`, né la
logica del kill switch — che continuano a usare `self.capital` esattamente
come nel motore standard. La scelta di design è che `self.capital`
rappresenta il capitale ANCORA A RISCHIO (l'"investito"), non il
patrimonio totale — quando l'accantonamento consolida una quota, quella
quota viene SOTTRATTA da `self.capital` e aggiunta a `self.side_pool`.
Questo significa che il kill switch giornaliero (-4%/-5%) e il sizing
dei trade si applicano automaticamente solo al capitale ancora in
gioco, senza bisogno di override separati — esattamente il
comportamento voluto ("l'accantonato non è più a rischio").
Il patrimonio totale (quello che conta per l'utente) è sempre
`self.capital + self.side_pool`.

SANITY CHECK OBBLIGATORIO (in fondo al file): con threshold_mult
irraggiungibile (es. 999), questa sottoclasse deve produrre risultati
IDENTICI a BacktestEngineFloatingKillSwitch — il ramo di
accantonamento non viene mai raggiunto.

STATO: costruito ma NON ANCORA validato secondo il protocollo standard
del progetto (sanity check + verifica di non-patologia sui 5 periodi).
Vedi accantonamento_validation.py per il test completo. Non usare in
demo/live finché quel test non è passato.
"""

from __future__ import annotations

import pandas as pd
from engine_floating_kill_switch import BacktestEngineFloatingKillSwitch


class BacktestEngineAccantonamento(BacktestEngineFloatingKillSwitch):

    def __init__(self, *args, consolidate_pct: float = 0.4, threshold_mult: float = 1.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.consolidate_pct = consolidate_pct
        self.threshold_mult = threshold_mult
        self.side_pool = 0.0
        self.reference = self.capital0
        self.threshold = self.reference * self.threshold_mult
        self._last_month: tuple | None = None

        # log diagnostico: (data, capitale_investito_prima, consolidato, nuovo_investito, side_pool_dopo)
        self.consolidation_log: list[tuple] = []

    def _try_consolidate(self):
        """Consolida a gradini finché il capitale investito non scende
        sotto la soglia corrente (while, non if — gestisce anche salti
        di più gradini in un colpo solo, es. dopo un mese eccezionale)."""
        while self.capital > self.threshold:
            capital_before = self.capital
            gain = self.capital - self.reference
            consolidated = self.consolidate_pct * gain
            self.side_pool += consolidated
            self.capital -= consolidated
            self.reference = self.capital
            self.threshold = self.reference * self.threshold_mult
            self.consolidation_log.append(
                (self._current_day, capital_before, consolidated, self.capital, self.side_pool)
            )

    def _reset_day_if_needed(self, ts: pd.Timestamp):
        day = ts.date()
        if self._current_day != day:
            month = (day.year, day.month)
            if self._last_month is not None and month != self._last_month:
                self._try_consolidate()  # check al cambio di mese, su capitale di fine mese precedente
            self._last_month = month
        super()._reset_day_if_needed(ts)

    def run(self, data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
        trades_df, metrics_df = super().run(data)
        self._try_consolidate()  # check finale di chiusura periodo (cattura l'ultimo mese parziale)

        metrics_df = metrics_df.copy()
        metrics_df["capitale_investito_finale"] = self.capital
        metrics_df["accantonato_finale"] = self.side_pool
        metrics_df["capitale_totale_finale"] = self.capital + self.side_pool
        metrics_df["n_consolidamenti"] = len(self.consolidation_log)
        return trades_df, metrics_df


if __name__ == "__main__":
    import sys
    print("Questo file va importato come modulo dal workflow di validazione, "
          "non eseguito direttamente. Vedi accantonamento_validation.py")
    sys.exit(0)
