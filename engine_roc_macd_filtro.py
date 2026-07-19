"""
engine_roc_macd_filtro.py — Sottoclasse isolata di
BacktestEngineFloatingKillSwitch che aggiunge un filtro di CONFERMA
momentum via ROC e MACD, mai testato prima in questo progetto (diverso
dai filtri ADX×volatilità e fascia oraria già bocciati — quelli erano
varianti del filtro di regime ADX già incorporato nel segnale; ROC/MACD
sono indicatori di momentum indipendenti, angolo genuinamente nuovo).

IPOTESI (fissata PRIMA di vedere i risultati, principio 5 Protocollo
Anti-Rumore): un breakout V6 è più affidabile se il momentum a breve
termine (ROC) e la relazione tra medie mobili veloci (MACD) concordano
già con la direzione del segnale — un breakout che scatta mentre
ROC/MACD sono ancora piatti o contrari potrebbe essere rumore/falso
segnale in un mercato che si sta muovendo poco.

REGOLA (fissata prima, nessun parametro libero oltre le impostazioni
standard/canoniche di ROC e MACD, per minimizzare il rischio di
overfitting):
  ROC(14) = (close - close_14_barre_fa) / close_14_barre_fa * 100
  MACD standard: EMA12 - EMA26, signal = EMA9 del MACD, hist = MACD-signal

  LONG: blocca se NON (MACD_hist > 0 E ROC > 0) — calcolati sulla barra
        di segnale (N), stessa barra usata per adx/atr, coerente con
        l'anti look-ahead del motore (esecuzione alla barra N+1)
  SHORT: blocca se NON (MACD_hist < 0 E ROC < 0)

COMPORTAMENTO: blocca SOLO nuovi ingressi. Le posizioni già aperte
continuano a essere gestite normalmente da stop/target/max holding.

Richiede l'override completo di run() (non solo _open_position come il
filtro ADX×ATR) perché ROC/MACD vanno letti dalla barra di SEGNALE
(prev_bar), non dalla barra di esecuzione (cur_bar) — dati non
disponibili nel dizionario candidato costruito dal motore base.

Nessuna modifica a engine.py né a engine_floating_kill_switch.py.

SANITY CHECK (in engine_roc_macd_filtro_test.py): con roc_threshold e
macd_threshold impostati a valori sempre soddisfatti (soglia 0 su
entrambi, cioè la regola diventa "richiedi solo lo stesso segno del
segnale" — per un vero bypass servirebbe disattivare il filtro del
tutto, gestito con un flag enabled=False), deve produrre risultati
IDENTICI al motore standard.
"""

from __future__ import annotations

import pandas as pd
from engine_floating_kill_switch import BacktestEngineFloatingKillSwitch

ROC_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9


def add_roc_macd_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Aggiunge colonne 'roc' e 'macd_hist' a un dataframe con colonna
    'close' — stesso pattern di mean_reversion_signals.py (funzione
    pura, riusabile, non modifica engine.py)."""
    out = df.copy()
    out["roc"] = (out["close"] - out["close"].shift(ROC_PERIOD)) / out["close"].shift(ROC_PERIOD) * 100.0

    ema_fast = out["close"].ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = out["close"].ewm(span=MACD_SLOW, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
    out["macd_hist"] = macd_line - signal_line
    return out


class BacktestEngineROCMACDFiltro(BacktestEngineFloatingKillSwitch):

    def __init__(self, *args, enabled: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.enabled = enabled  # False = bypass completo, per il sanity check
        self.n_blocked = {"DAX": 0, "FTSE100": 0}

    def _confirms(self, direction: str, roc: float, macd_hist: float) -> bool:
        if pd.isna(roc) or pd.isna(macd_hist):
            return False  # dati insufficienti (warmup) -> tratta come non confermato, blocca
        if direction == "long":
            return macd_hist > 0 and roc > 0
        else:
            return macd_hist < 0 and roc < 0

    def run(self, data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
        # precalcola roc/macd_hist per ogni strumento, una volta sola
        data_ext = {name: add_roc_macd_columns(df) for name, df in data.items()}

        tradable_instruments = [
            name for name in data_ext
            if self.instruments.get(name) is not None and self.instruments[name].tradable
        ]
        if not tradable_instruments:
            raise ValueError("Nessuno strumento tradabile fornito a run().")

        all_timestamps = sorted(set().union(
            *[set(data_ext[i]["timestamp"]) for i in tradable_instruments]))

        for ts in all_timestamps:
            self._reset_day_if_needed(ts)

            for pos in list(self.open_positions):
                inst_df = data_ext[pos.instrument]
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
                    inst_df = data_ext[pos.instrument]
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
                inst_df = data_ext[name]
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

                # --- NUOVO: filtro di conferma ROC/MACD, letto dalla barra di segnale ---
                if self.enabled and not self._confirms(prev_bar["signal"], prev_bar["roc"], prev_bar["macd_hist"]):
                    self.n_blocked[name] = self.n_blocked.get(name, 0) + 1
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
        import engine as eng
        metrics_df = eng.compute_run_metrics(trades_df, self.capital0, self.capital)
        return trades_df, metrics_df


if __name__ == "__main__":
    import sys
    print("Questo file va importato come modulo. Vedi engine_roc_macd_filtro_test.py "
          "per il sanity check e il test di impatto sui 5 periodi ufficiali.")
    sys.exit(0)
