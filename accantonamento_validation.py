"""
engine_accantonamento.py — Estensione isolata di
BacktestEngineFloatingKillSwitch con meccanismo di accantonamento
periodico, generalizzata per coprire le 4 varianti esplorate in chat
(16/07/2026):

  mode="giveback"  (opzione 2): ogni volta che il capitale investito
      supera il proprio massimo storico, `giveback_pct` dell'incremento
      esce e va accantonato.
  mode="gradini"   (opzione 3): ogni volta che il capitale investito
      supera +50% (threshold_mult) rispetto all'ultimo riferimento,
      `consolidate_pct` del guadagno sopra quella soglia esce e va
      accantonato.

  check_frequency="continuo": il controllo avviene dopo OGNI trade
      chiuso (agganciato a _close_position).
  check_frequency="mensile":  il controllo avviene solo al cambio di
      mese di calendario (agganciato a _reset_day_if_needed, una volta
      per nuovo giorno, quindi anche per nuovo mese).

DESIGN invariato dalla versione precedente: `self.capital` rappresenta
il capitale ANCORA A RISCHIO (l'investito) — quando l'accantonamento
consolida una quota, esce da `self.capital` ed entra in
`self.side_pool`. Kill switch e sizing dei trade continuano a usare
`self.capital` senza override separati: l'accantonato esce
automaticamente dal loro raggio d'azione. Patrimonio totale =
`self.capital + self.side_pool`.

SANITY CHECK (obbligatorio, vedi accantonamento_validation.py):
  - mode="gradini" con threshold_mult=999 (irraggiungibile) -> side_pool
    resta 0, risultati identici al motore standard.
  - mode="giveback" con giveback_pct=0 -> side_pool resta 0 (ogni skim
    è 0), risultati identici al motore standard.
  Entrambi verificati per check_frequency sia "continuo" sia "mensile"
  (la frequenza del check è irrilevante se il meccanismo non consolida
  mai nulla).
"""

from __future__ import annotations

import pandas as pd
from engine_floating_kill_switch import BacktestEngineFloatingKillSwitch


class BacktestEngineAccantonamento(BacktestEngineFloatingKillSwitch):

    def __init__(self, *args, mode: str = "gradini", check_frequency: str = "mensile",
                 giveback_pct: float = 0.3, consolidate_pct: float = 0.4,
                 threshold_mult: float = 1.5, **kwargs):
        super().__init__(*args, **kwargs)
        if mode not in ("gradini", "giveback"):
            raise ValueError(f"mode deve essere 'gradini' o 'giveback', ricevuto '{mode}'")
        if check_frequency not in ("continuo", "mensile"):
            raise ValueError(f"check_frequency deve essere 'continuo' o 'mensile', ricevuto '{check_frequency}'")

        self.mode = mode
        self.check_frequency = check_frequency
        self.giveback_pct = giveback_pct
        self.consolidate_pct = consolidate_pct
        self.threshold_mult = threshold_mult

        self.side_pool = 0.0
        self.peak = self.capital0            # usato solo da mode="giveback"
        self.reference = self.capital0        # usato solo da mode="gradini"
        self.threshold = self.reference * self.threshold_mult
        self._last_month: tuple | None = None

        # log diagnostico: (data, consolidato, capitale_investito_dopo, side_pool_dopo)
        self.consolidation_log: list[tuple] = []

    def _try_consolidate(self):
        if self.mode == "giveback":
            if self.capital > self.peak:
                increment = self.capital - self.peak
                skim = self.giveback_pct * increment
                if skim <= 0:
                    self.peak = self.capital
                    return
                self.side_pool += skim
                self.capital -= skim
                self.peak = self.capital + skim
                self.consolidation_log.append((self._current_day, skim, self.capital, self.side_pool))
        else:  # gradini
            while self.capital > self.threshold:
                gain = self.capital - self.reference
                consolidated = self.consolidate_pct * gain
                if consolidated <= 0:
                    break
                self.side_pool += consolidated
                self.capital -= consolidated
                self.reference = self.capital
                self.threshold = self.reference * self.threshold_mult
                self.consolidation_log.append((self._current_day, consolidated, self.capital, self.side_pool))

    def _close_position(self, pos, exit_time, exit_price: float, exit_reason: str):
        super()._close_position(pos, exit_time, exit_price, exit_reason)
        if self.check_frequency == "continuo":
            self._try_consolidate()

    def _reset_day_if_needed(self, ts: pd.Timestamp):
        day = ts.date()
        if self._current_day != day:
            if self.check_frequency == "mensile":
                month = (day.year, day.month)
                if self._last_month is not None and month != self._last_month:
                    self._try_consolidate()
                self._last_month = (day.year, day.month)
        super()._reset_day_if_needed(ts)

    def run(self, data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
        trades_df, metrics_df = super().run(data)
        self._try_consolidate()  # check finale di chiusura periodo

        metrics_df = metrics_df.copy()
        metrics_df["capitale_investito_finale"] = self.capital
        metrics_df["accantonato_finale"] = self.side_pool
        metrics_df["capitale_totale_finale"] = self.capital + self.side_pool
        metrics_df["n_consolidamenti"] = len(self.consolidation_log)
        return trades_df, metrics_df


if __name__ == "__main__":
    import sys
    print("Questo file va importato come modulo, non eseguito direttamente. "
          "Vedi accantonamento_validation.py e accantonamento_confronto_4varianti.py")
    sys.exit(0)
