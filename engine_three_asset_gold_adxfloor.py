"""
engine_three_asset_gold_adxfloor.py — Variante di engine_three_asset_gold.py
con un pavimento di protezione ADX RELATIVO, discusso in chat il
19/07/2026 dopo l'osservazione che la selezione a sola correlazione
può scartare un segnale forte per uno debole solo perché più
scorrelato (visto concretamente nel test mirato giugno-luglio 2026,
dove GOLD ha scalzato il DAX che quel mese era il segnale sano).

REGOLA (fissata prima di vedere i risultati, un solo parametro nuovo,
la soglia ADX 30 — STESSA soglia già usata altrove nel progetto per
"trend forte", RCA 18/07 filtro ADX×ATR sul DAX, non un nuovo numero
scelto a piacere):

  Un candidato è "intoccabile" (non escludibile per motivi di
  correlazione) SOLO SE:
    (a) il suo ADX > 30, E
    (b) il suo ADX è maggiore dell'ADX di ogni candidato che
        prenderebbe il suo posto nel sottoinsieme scelto

  Tra i sottoinsiemi che rispettano questo vincolo, si sceglie
  comunque quello a correlazione minima (stesso criterio di
  engine_three_asset_gold.py) — la correlazione resta il criterio
  guida, ha solo un limite sopra. Se nessun sottoinsieme rispetta il
  vincolo (raro, gestito come rete di sicurezza), si ricade sulla
  selezione a sola correlazione per non bloccare il motore.

Questo NON è uno score composito (niente pesi, niente somma di più
fattori) — è un confronto diretto tra due numeri già esistenti (ADX
del candidato scartato vs ADX del candidato tenuto), applicato solo
nel caso raro di 3 segnali simultanei con capacità insufficiente.

Nessuna modifica a engine.py, engine_three_asset_gold.py o alle altre
sottoclassi esistenti.
"""

from __future__ import annotations

import itertools
import pandas as pd

import engine as eng
from engine_floating_kill_switch import BacktestEngineFloatingKillSwitch
from engine_mean_reversion import BacktestEngineMeanReversion
from engine_three_asset_gold import pair_corr, GOLD_CONFIG, instruments_with_gold

ADX_PROTECTION_THRESHOLD = 30.0


def _subset_respects_adx_floor(candidates: list[dict], subset: tuple[dict, ...]) -> bool:
    """Un sottoinsieme e' valido se nessun candidato ESCLUSO con
    ADX>soglia ha un ADX maggiore di un candidato INCLUSO nel
    sottoinsieme."""
    included_instruments = {c["instrument"] for c in subset}
    excluded = [c for c in candidates if c["instrument"] not in included_instruments]
    for exc in excluded:
        if exc["adx"] > ADX_PROTECTION_THRESHOLD:
            for inc in subset:
                if inc["adx"] < exc["adx"]:
                    return False
    return True


def _best_subset_adx_floor(candidates: list[dict], already_open_instruments: list[str], slots_free: int) -> list[dict]:
    if len(candidates) <= slots_free:
        return candidates

    valid_subsets_by_size = {}
    for size in range(min(slots_free, len(candidates)), 0, -1):
        valid = [combo for combo in itertools.combinations(candidates, size)
                 if _subset_respects_adx_floor(candidates, combo)]
        if valid:
            valid_subsets_by_size[size] = valid
            break  # dimensione massima possibile con almeno un sottoinsieme valido

    if not valid_subsets_by_size:
        # rete di sicurezza: nessun sottoinsieme rispetta il vincolo ADX,
        # ricade sulla selezione a sola correlazione per non bloccare il motore
        from engine_three_asset_gold import _best_subset
        return _best_subset(candidates, already_open_instruments, slots_free)

    size = next(iter(valid_subsets_by_size))
    best_subset, best_score = [], float("inf")
    for combo in valid_subsets_by_size[size]:
        instruments_in_combo = [c["instrument"] for c in combo]
        score = 0.0
        for a, b in itertools.combinations(instruments_in_combo, 2):
            score += pair_corr(a, b)
        for a in instruments_in_combo:
            for open_inst in already_open_instruments:
                score += pair_corr(a, open_inst)
        if score < best_score:
            best_score = score
            best_subset = list(combo)
    return best_subset


def _run_with_gold_adxfloor(self, data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
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

        slots_free = self.p.max_concurrent_positions - len(self.open_positions)
        already_open_instruments = [p.instrument for p in self.open_positions]

        selected = _best_subset_adx_floor(candidates, already_open_instruments, slots_free)

        for c in selected:
            if pd.isna(c["atr"]) or pd.isna(c["adx"]):
                continue
            if self._orders_today >= self.p.max_new_orders_per_day:
                break
            self._open_position(c["instrument"], c["direction"], c["bar"], c["atr"], c["adx"])

    trades_df = self.trades_to_dataframe()
    metrics_df = eng.compute_run_metrics(trades_df, self.capital0, self.capital)
    return trades_df, metrics_df


class BacktestEngineV6GoldADXFloor(BacktestEngineFloatingKillSwitch):
    """V6 su 3 strumenti con selezione per correlazione minima E
    pavimento di protezione ADX relativo (ADX>30 e maggiore del
    sostituto -> mai escludibile)."""

    def run(self, data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
        return _run_with_gold_adxfloor(self, data)


if __name__ == "__main__":
    import sys
    print("Questo file va importato come modulo. Vedi engine_three_asset_gold_compare_test.py "
          "per il confronto a 3 vie (baseline / correlazione pura / correlazione+pavimento ADX).")
    sys.exit(0)
