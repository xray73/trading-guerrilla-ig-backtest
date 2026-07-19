"""
engine_three_asset_gold.py — Estende l'universo attivo da 2 a 3
strumenti (DAX, FTSE100, GOLD) SENZA aggiungere un terzo sotto-pool di
capitale: GOLD entra come candidato aggiuntivo nello STESSO pool (V6 o
mean-reversion), competendo per gli stessi 2 slot concorrenti già
esistenti — nessuna diluizione di capitale, coerente con la richiesta
in chat del 19/07/2026.

PROBLEMA RISOLTO (segnalato ma mai integrato, RCA Addendum 13/07 sez.23):
la selezione tra candidati quando ce ne sono più degli slot liberi
usava _correlation_penalty(), che conta la correlazione SOLO contro le
posizioni GIA' aperte — con 2 strumenti non serviva mai scegliere (se
c'era 1 slot libero e 2 segnali, non serviva discriminare). Con 3
strumenti candidati e 2 slot, questo diventa un caso reale e frequente:
serve scegliere quale SOTTOINSIEME di candidati aprire, non solo
ordinarli uno alla volta.

SOLUZIONE: quando i candidati in un ciclo superano gli slot liberi,
si prova ogni sottoinsieme possibile (al massimo pochi, con 3
strumenti e max 2 slot non supera mai 3 combinazioni) e si sceglie
quello con la SOMMA di correlazione pairwise più bassa, includendo sia
le posizioni già aperte sia i candidati nel sottoinsieme — non solo il
confronto candidato-per-candidato del motore base.

Matrice di correlazione REALE (RCA Addendum 13/07 sez.19, rendimenti
giornalieri storico 2015-2026):
    DAX-FTSE100:  0,8103
    DAX-GOLD:     0,1790
    FTSE100-GOLD: 0,1005

Parametri GOLD (RCA Addendum 13/07 sez.22, calibrazione walk-forward
dedicata, train 2023 -> verifica 2024-25 -> conferma sui restanti 3):
ATR moltiplicatore 3,5×, lookback breakout 30 barre, rischio 1,5%
(allineato a FTSE100), spread 0,90pt (osservato IG), size minima 0,10,
margine 5%, point_value USD1 (~EUR0,88, approssimazione dichiarata).
Costante Dukascopy: INSTRUMENT_FX_METALS_XAU_USD.

**NOTA IMPORTANTE**: i parametri GOLD calibrati in RCA sez.22 non hanno
un secondo livello di verifica indipendente fuori campione nel senso
più stretto (grid ATR/lookback con train 2023, ma la scelta finale tra
i due candidati migliori non è stata riverificata su un terzo taglio
di dati) — trattare come ipotesi di lavoro solida, non parametro
definitivo. Lo stesso vale per l'uso di questi parametri nel ramo
mean-reversion: MAI calibrati specificamente per GOLD in modalità
mean-reversion, riusano lo stesso atr_multiplier di V6 per coerenza
con quanto già fatto per DAX/FTSE100 (nessuna calibrazione MR-specifica
esiste nemmeno per quei due strumenti).

Nessuna modifica a engine.py, engine_floating_kill_switch.py o
engine_mean_reversion.py.
"""

from __future__ import annotations

import itertools
import pandas as pd

import engine as eng
from engine_floating_kill_switch import BacktestEngineFloatingKillSwitch
from engine_mean_reversion import BacktestEngineMeanReversion

GOLD_CONFIG = eng.InstrumentConfig(
    name="GOLD", tradable=True,
    breakout_lookback=30, atr_multiplier=3.5, risk_pct=0.015,
    point_value=0.88, spread_fixed=0.90,
    min_tradable_size=0.10, margin_pct=0.05,
)

CORRELATION = {
    frozenset(["DAX", "FTSE100"]): 0.8103,
    frozenset(["DAX", "GOLD"]): 0.1790,
    frozenset(["FTSE100", "GOLD"]): 0.1005,
}


def instruments_with_gold() -> dict:
    """Copia di eng.INSTRUMENTS con GOLD aggiunto — non modifica
    engine.py, coerente con la convenzione già usata per lo spread
    realistico (dataclasses.replace su una copia locale)."""
    out = dict(eng.INSTRUMENTS)
    out["GOLD"] = GOLD_CONFIG
    return out


def pair_corr(a: str, b: str) -> float:
    if a == b:
        return 1.0
    return CORRELATION.get(frozenset([a, b]), 0.5)  # fallback prudente se coppia ignota


def _best_subset(candidates: list[dict], already_open_instruments: list[str], slots_free: int) -> list[dict]:
    """Tra tutti i sottoinsiemi di `candidates` di dimensione <=
    slots_free, ritorna quello con la SOMMA di correlazione pairwise
    piu' bassa (candidati tra loro + candidati con le posizioni gia'
    aperte). A parita' di correlazione, preferisce il sottoinsieme piu'
    grande (usa piu' slot disponibili)."""
    if len(candidates) <= slots_free:
        return candidates

    best_subset, best_score = [], float("inf")
    for size in range(min(slots_free, len(candidates)), 0, -1):
        for combo in itertools.combinations(candidates, size):
            instruments_in_combo = [c["instrument"] for c in combo]
            score = 0.0
            # correlazione tra i candidati del sottoinsieme, a coppie
            for a, b in itertools.combinations(instruments_in_combo, 2):
                score += pair_corr(a, b)
            # correlazione tra ciascun candidato e le posizioni gia' aperte
            for a in instruments_in_combo:
                for open_inst in already_open_instruments:
                    score += pair_corr(a, open_inst)
            if score < best_score:
                best_score = score
                best_subset = list(combo)
        if best_subset:
            break  # trovato il miglior sottoinsieme alla dimensione massima possibile
    return best_subset


def _run_with_gold(self, data: dict[str, pd.DataFrame], signal_fn) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Corpo comune di run() per V6+GOLD e MR+GOLD — signal_fn e' la
    funzione che genera i segnali sul dataframe grezzo (eng.generate_signals
    per V6, generate_mean_reversion_signals per MR), passata dal chiamante
    per non duplicare due volte la stessa struttura di ciclo."""
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

        # --- NUOVO: selezione multi-candidato per correlazione minima (invece di sort+greedy) ---
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


class BacktestEngineV6Gold(BacktestEngineFloatingKillSwitch):
    """V6 su 3 strumenti (DAX/FTSE100/GOLD), stesso pool di capitale,
    stessi 2 slot concorrenti — selezione per correlazione minima."""

    def run(self, data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
        return _run_with_gold(self, data, eng.generate_signals)


class BacktestEngineMeanReversionGold(BacktestEngineMeanReversion):
    """Mean-reversion su 3 strumenti, stessa logica salta-invece-di-forza
    di engine_mean_reversion.py (ereditata, _position_size invariato),
    con la stessa selezione multi-candidato per correlazione minima."""

    def run(self, data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
        return _run_with_gold(self, data, None)


if __name__ == "__main__":
    import sys
    print("Questo file va importato come modulo. Vedi engine_three_asset_gold_test.py "
          "per il sanity check e il test di impatto sui 5 periodi ufficiali.")
    sys.exit(0)
