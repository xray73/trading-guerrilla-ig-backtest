"""
engine_extended_orders.py — Estensione isolata (sopra
BacktestEngineFloatingKillSwitch, già adottato come standard il
15/07/2026) che permette slot di trade oltre il 3° giornaliero
(Charter sez.3), ciascuno modulato dinamicamente in base al PnL netto
di giornata già realizzato al momento in cui quello slot si aprirebbe.

REGOLA (definita in chat 15/07/2026):
  - Slot 1-3 (BASE_MAX_ORDERS): invariati, rischio standard dello
    strumento (2% DAX, 1.5% FTSE100), come oggi.
  - Slot 4, 5, ... (fino a max_new_orders_per_day del run): il rischio
    massimo concesso è
        min(EXTRA_SLOT_PCT × PnL_netto_giornata_finora, rischio_standard)
    ricalcolato ogni volta, iterativamente — non un valore fissato
    all'inizio della giornata. Se il PnL netto finora è <= 0, lo slot
    extra NON si apre (protezione esplicita: "non trasformare una
    giornata vincente in perdente" non si applica nemmeno al contrario,
    ma qui il punto è non aggiungere rischio quando la giornata non lo
    giustifica).
  - IMPORTANTE — comportamento diverso dal fallback standard: se il
    rischio modulato non basta a coprire la size minima negoziabile,
    lo slot extra viene SALTATO, non forzato al minimo come fa il
    motore standard per gli slot 1-3. Forzare al minimo violerebbe il
    tetto di rischio esplicitamente voluto per questi slot.
  - Il vincolo di posizioni concorrenti massime (max_concurrent_positions,
    default 2) resta invariato — gli slot extra riguardano solo il
    conteggio di NUOVI ordini nella giornata, non quante posizioni
    possono essere aperte insieme.

Per attivare gli slot extra, il chiamante deve passare un ChartaParams
con max_new_orders_per_day alzato (es. 5) — il gate BASE_MAX_ORDERS
dentro questa classe si occupa di applicare comunque la modulazione a
tutto ciò che supera i primi 3.

Sanity check: con max_new_orders_per_day=3 (invariato), questa classe
deve produrre risultati IDENTICI a BacktestEngineFloatingKillSwitch
(il ramo "slot extra" non viene mai raggiunto).

MODIFICA 16/07/2026 (solo diagnostica, NESSUN impatto sulla logica di
trading/decisione): aggiunti extra_slot_skip_pnl_log e
extra_slot_skip_minsize_log, stesso formato di extra_slot_log
[(instrument, bar_timestamp), ...], per poter filtrare gli skip per
finestra temporale in analisi esterne invece di leggere solo il
contatore cumulativo su tutta la corsa. I contatori n_extra_slot_*
restano invariati e continuano a funzionare come prima.
"""

from __future__ import annotations

import dataclasses
import pandas as pd

import engine as eng
from engine_floating_kill_switch import BacktestEngineFloatingKillSwitch


class BacktestEngineExtendedOrders(BacktestEngineFloatingKillSwitch):

    BASE_MAX_ORDERS = 3

    def __init__(self, *args, extra_slot_pct: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.extra_slot_pct = extra_slot_pct
        self.n_extra_slot_opened = 0
        self.n_extra_slot_skipped_pnl = 0
        self.n_extra_slot_skipped_min_size = 0
        self.extra_slot_log: list[tuple] = []  # (instrument, entry_time) per correlazione con trades_df
        self.extra_slot_skip_pnl_log: list[tuple] = []  # (instrument, bar_timestamp) skip per PnL<=0
        self.extra_slot_skip_minsize_log: list[tuple] = []  # (instrument, bar_timestamp) skip per size minima

    def _open_position(self, instrument: str, direction: str, bar: pd.Series,
                        atr_at_entry: float, adx_at_entry: float):
        order_index = self._orders_today + 1

        if order_index <= self.BASE_MAX_ORDERS:
            return super()._open_position(instrument, direction, bar, atr_at_entry, adx_at_entry)

        # ── slot extra (4°, 5°, ...) ──
        inst = self.instruments[instrument]
        pnl_netto_finora = self.capital - self._day_start_capital

        if pnl_netto_finora <= 0:
            self.n_extra_slot_skipped_pnl += 1
            self.extra_slot_skip_pnl_log.append((instrument, bar["timestamp"]))
            return

        standard_risk_amount = self.capital * inst.risk_pct
        modulated_risk_amount = min(self.extra_slot_pct * pnl_netto_finora, standard_risk_amount)
        if modulated_risk_amount <= 0:
            self.n_extra_slot_skipped_pnl += 1
            self.extra_slot_skip_pnl_log.append((instrument, bar["timestamp"]))
            return

        spread = inst.spread_fixed
        raw_price = bar["open"]
        entry_price = raw_price + spread / 2 if direction == "long" else raw_price - spread / 2
        stop_distance = atr_at_entry * inst.atr_multiplier
        if direction == "long":
            stop_loss = entry_price - stop_distance
            take_profit = entry_price + stop_distance * self.p.rr_target
        else:
            stop_loss = entry_price + stop_distance
            take_profit = entry_price - stop_distance * self.p.rr_target

        risk_distance = abs(entry_price - stop_loss)
        if risk_distance <= 0:
            return

        raw_size = modulated_risk_amount / (risk_distance * inst.point_value)
        if raw_size < inst.min_tradable_size:
            # rischio modulato insufficiente per la size minima -> salta,
            # NON forzare (violerebbe il tetto voluto per gli slot extra)
            self.n_extra_slot_skipped_min_size += 1
            self.extra_slot_skip_minsize_log.append((instrument, bar["timestamp"]))
            return

        size = raw_size
        margin_required = size * entry_price * inst.point_value * inst.margin_pct
        margin_reduced = False
        if margin_required > self.capital:
            max_size_by_margin = self.capital / (entry_price * inst.point_value * inst.margin_pct)
            if max_size_by_margin < inst.min_tradable_size:
                return
            size = max_size_by_margin
            margin_reduced = True
        if margin_reduced:
            self.n_margin_reduced += 1

        pos = eng.Position(
            instrument=instrument, direction=direction,
            entry_bar_index=bar.name, entry_time=bar["timestamp"],
            entry_price=entry_price, stop_loss=stop_loss, take_profit=take_profit,
            size=size, risk_amount=modulated_risk_amount, atr_at_entry=atr_at_entry,
            adx_at_entry=adx_at_entry, rr_planned=self.p.rr_target,
            forced_min_size=False, max_holding_bars=self.p.max_holding_bars,
        )
        self.open_positions.append(pos)
        self._orders_today += 1
        self.n_extra_slot_opened += 1
        self.extra_slot_log.append((instrument, bar["timestamp"]))
