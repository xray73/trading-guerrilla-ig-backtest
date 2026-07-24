"""
engine_three_asset_gold_longonly.py — Variante di engine_three_asset_gold.py
che disattiva lo short SOLO su GOLD (DAX/FTSE100 restano long+short come
sempre). Nasce dalla scoperta del 24/07/2026: il segnale V6 su GOLD standalone
ha un edge quasi interamente asimmetrico — long R medio netto (spread reale
applicato) +0,154 (bootstrap z=2,44, IC 95% esclude lo zero), short -0,102
(negativo). L'edge complessivo misto (z=1,19) era quindi in gran parte un
artefatto del bull market strutturale di GOLD nel periodo, non una vera
capacita' di trend-following simmetrica del segnale.

Verificato PRIMA su research_v6_candidate_path (query D1 dirette, R-multiple
puro, nessun capitale/size) — qui si passa al motore vero perche' disattivare
un lato del segnale cambia QUANDO si aprono/chiudono le posizioni e quindi
la size, il capitale disponibile per i trade successivi e la selezione
multi-candidato (_best_subset) — lo stesso tipo di effetto a cascata gia'
scoperto il 23-24/07 con l'idea 1 (uscita dinamica): una query SQL su trade
gia' avvenuti non puo' vederlo, serve il motore vero fin da subito.

Nessuna modifica a engine.py, engine_floating_kill_switch.py,
engine_mean_reversion.py o engine_three_asset_gold.py — questo file copia
SOLO _run_with_gold (gia' essa stessa una copia dichiarata di run(), stesso
principio) aggiungendo un unico filtro esplicito nel loop di raccolta
candidati: se lo strumento e' GOLD e la direzione e' 'short', il candidato
non viene nemmeno generato. DAX e FTSE100 restano invariati in tutto.

SANITY CHECK OBBLIGATORIO (prima di qualunque conclusione sul risultato):
con GOLD_LONGONLY_FILTER_ACTIVE=False (parametro neutro), questo motore deve
riprodurre ESATTAMENTE gli stessi risultati di BacktestEngineV6Gold — vedi
engine_three_asset_gold_longonly_test.py.
"""

from __future__ import annotations

import itertools
import pandas as pd

import engine as eng
from engine_floating_kill_switch import BacktestEngineFloatingKillSwitch
from engine_mean_reversion import BacktestEngineMeanReversion
from engine_three_asset_gold import GOLD_CONFIG, CORRELATION, pair_corr, _best_subset


def _run_with_gold_longonly(self, data: dict[str, pd.DataFrame], signal_fn,
                             gold_longonly_filter_active: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Copia di _run_with_gold (engine_three_asset_gold.py) con UNA sola
    modifica esplicita, commentata inline dove appare. Tutto il resto e'
    identico, incluso _best_subset per la selezione multi-candidato quando
    ci sono piu' segnali degli slot liberi."""
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

            # <<< UNICA MODIFICA rispetto a _run_with_gold originale >>>
            # Filtro GOLD long-only (24/07/2026): scoperto che lo short su
            # GOLD ha R medio netto negativo (-0,102, spread reale applicato)
            # mentre il long e' positivo (+0,154, z=2,44) — l'edge misto era
            # quasi interamente drift strutturale del bull market. Il
            # candidato short su GOLD non viene generato quando il filtro e'
            # attivo. DAX/FTSE100 non toccati in nessun caso.
            if gold_longonly_filter_active and name == "GOLD" and prev_bar["signal"] == "short":
                continue
            # <<< fine modifica >>>

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

        selected = _best_subset(candidates, already_open_instruments, slots_free)

        for c in selected:
            if pd.isna(c["atr"]) or pd.isna(c["adx"]):
                continue
            if self._orders_today >= self.p.max_new_orders_per_day:
                break
            self._open_position(c["instrument"], c["direction"], c["bar"], c["atr"], c["adx"])

    trades_df = self.trades_to_dataframe()
    metrics_df = eng.compute_run_metrics(trades_df, self.capital0, self.capital)
    return trades_df, metrics_df


class BacktestEngineV6GoldLongOnly(BacktestEngineFloatingKillSwitch):
    """V6 su 3 strumenti, GOLD limitato al solo long. Parametro
    gold_longonly_filter_active=False riproduce esattamente
    BacktestEngineV6Gold (per il sanity check)."""

    def __init__(self, *args, gold_longonly_filter_active: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.gold_longonly_filter_active = gold_longonly_filter_active

    def run(self, data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
        return _run_with_gold_longonly(self, data, eng.generate_signals,
                                        gold_longonly_filter_active=self.gold_longonly_filter_active)


class BacktestEngineMeanReversionGoldLongOnly(BacktestEngineMeanReversion):
    """Equivalente mean-reversion — incluso per completezza/coerenza con
    engine_three_asset_gold.py, ma il filone MR+GOLD e' gia' chiuso
    (22/07/2026, campione insufficiente n=1) — non usare senza prima
    riconsiderare quella chiusura."""

    def __init__(self, *args, gold_longonly_filter_active: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.gold_longonly_filter_active = gold_longonly_filter_active

    def run(self, data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
        return _run_with_gold_longonly(self, data, None,
                                        gold_longonly_filter_active=self.gold_longonly_filter_active)


if __name__ == "__main__":
    import sys
    print("Questo file va importato come modulo. Vedi "
          "engine_three_asset_gold_longonly_test.py per sanity check e test causale.")
    sys.exit(0)
