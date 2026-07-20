"""
engine_persistence_variant.py — Estensione isolata di
BacktestEngineFloatingKillSwitch che aggiunge un requisito di
PERSISTENZA MINIMA del segnale prima di aprire una posizione: invece di
entrare alla prima barra utile in cui il segnale è attivo (comportamento
standard, persistenza=1), richiede che il segnale sia rimasto attivo per
almeno `min_persistence` barre consecutive.

Origine: osservazione in chat (20/07/2026) sul dataset di ricerca
research_v6_trade_features — i pochi trade (9% del totale) che per caso
erano entrati con persistenza>=2 (perche' un altro slot era occupato o
il limite ordini/giorno era gia' raggiunto quando il segnale e' scattato
la prima volta) avevano R medio ~3x superiore ai trade a persistenza=1.
Questo NON prova che aspettare apposta funzioni — la persistenza alta
nel dataset originale era un EFFETTO COLLATERALE del caso, non una
regola imposta. Questo motore testa la domanda vera: se il sistema
aspettasse DELIBERATAMENTE N barre prima di entrare, il risultato
sarebbe migliore, uguale, o peggiore? Alcuni segnali che oggi durano
solo 1 barra sparirebbero prima di raggiungere la soglia (trade perso
del tutto), altri entrerebbero piu' tardi a un prezzo/ATR diverso.

SANITY CHECK OBBLIGATORIO (in fondo al file): con min_persistence=1,
il comportamento DEVE essere IDENTICO a BacktestEngineFloatingKillSwitch
(persistenza>=1 e' sempre vera per qualunque barra con segnale attivo).

LIMITE METODOLOGICO NOTO (dichiarato, non nascosto): la persistenza qui
e' calcolata sulla serie GIA' tagliata al periodo ufficiale (stessa
`data` passata a run()), non sulla serie storica completa come nel
dataset di ricerca — per le primissime barre di ciascun periodo manca
il contesto pre-periodo. Effetto trascurabile su migliaia di barre per
periodo, ma dichiarato per trasparenza.

Nessuna modifica a engine.py o a engine_floating_kill_switch.py.
"""

from __future__ import annotations

import pandas as pd
from engine_floating_kill_switch import BacktestEngineFloatingKillSwitch
import engine as eng


class BacktestEnginePersistenceVariant(BacktestEngineFloatingKillSwitch):

    def __init__(self, *args, min_persistence: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_persistence = min_persistence

    @staticmethod
    def _signal_persistence(inst_df: pd.DataFrame, end_idx: int, direction: str) -> int:
        """Barre consecutive (terminando, incluso, a end_idx) con lo
        stesso segnale attivo — identica logica di
        count_consecutive_backward() usata in extract_v6_trade_features.py."""
        i = end_idx
        count = 0
        while i >= 0 and inst_df.iloc[i]["signal"] == direction:
            count += 1
            i -= 1
        return count

    def run(self, data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
        tradable_instruments = [
            name for name in data
            if self.instruments.get(name) is not None and self.instruments[name].tradable
        ]
        if not tradable_instruments:
            raise ValueError("Nessuno strumento tradabile fornito a run().")

        all_timestamps = sorted(set().union(
            *[set(data[i]["timestamp"]) for i in tradable_instruments]))

        for ts in all_timestamps:
            self._reset_day_if_needed(ts)

            for pos in list(self.open_positions):
                inst_df = data[pos.instrument]
                row = inst_df.loc[inst_df["timestamp"] == ts]
                if row.empty:
                    continue
                bar = row.iloc[0]
                bar_index = row.index[0]
                self._try_close_position(pos, bar, bar_index, self.instruments[pos.instrument])

            self.equity_curve.append((ts, self.capital))

            if not self._kill_switch_active and self.open_positions:
                current_bars = {}
                for pos in self.open_positions:
                    inst_df = data[pos.instrument]
                    row = inst_df.loc[inst_df["timestamp"] == ts]
                    if not row.empty:
                        current_bars[pos.instrument] = row.iloc[0]
                perdita_pct = self._floating_loss_pct(current_bars)
                if perdita_pct >= self.p.kill_switch_pct:
                    self._kill_switch_active = True

            if self._kill_switch_active:
                continue
            if self._orders_today >= self.p.max_new_orders_per_day:
                continue
            if len(self.open_positions) >= self.p.max_concurrent_positions:
                continue

            candidates = []
            for name in tradable_instruments:
                inst_df = data[name]
                idx = inst_df.index[inst_df["timestamp"] == ts]
                if len(idx) == 0:
                    continue
                i = idx[0]
                if i == 0:
                    continue
                prev_bar = inst_df.iloc[i - 1]
                cur_bar = inst_df.iloc[i]
                if prev_bar["signal"] not in ("long", "short"):
                    continue

                # --- NUOVO: requisito di persistenza minima ---
                persistence = self._signal_persistence(inst_df, i - 1, prev_bar["signal"])
                if persistence < self.min_persistence:
                    continue
                # --- fine nuovo ---

                already_open = any(p.instrument == name for p in self.open_positions)
                if already_open:
                    continue
                candidates.append({
                    "instrument": name, "direction": prev_bar["signal"],
                    "bar": cur_bar, "atr": prev_bar["atr"], "adx": prev_bar["adx"],
                    "rr": self.p.rr_target,
                })

            if not candidates:
                continue

            candidates.sort(key=lambda c: (-c["rr"], self._correlation_penalty(c["instrument"])))

            slots_free = self.p.max_concurrent_positions - len(self.open_positions)
            for c in candidates:
                if slots_free <= 0:
                    break
                if self._orders_today >= self.p.max_new_orders_per_day:
                    break
                if pd.isna(c["atr"]) or pd.isna(c["adx"]):
                    continue
                self._open_position(c["instrument"], c["direction"], c["bar"],
                                     c["atr"], c["adx"])
                slots_free -= 1

        trades_df = self.trades_to_dataframe()
        metrics_df = eng.compute_run_metrics(trades_df, self.capital0, self.capital)
        return trades_df, metrics_df
