"""
engine_floating_kill_switch.py — Estensione isolata di BacktestEngine
che risolve il gap identificato in chat il 15/07/2026: il kill switch
esistente (kill_switch_pct=0.04 in ChartaParams) controlla SOLO il
capitale già realizzato a chiusura trade, mai la perdita non
realizzata (floating) delle posizioni ancora aperte. Una sequenza di
2 trade stoppati + un 3° trade quasi al proprio stop può superare la
soglia di rischio giornaliero dichiarata senza che nulla intervenga.

COMPORTAMENTO INVARIATO rispetto all'originale: SOLO blocco nuovi
ordini per il resto della giornata, MAI chiusura forzata delle
posizioni aperte (deciso esplicitamente in chat — una chiusura forzata
taglierebbe anche trade che potrebbero recuperare prima del proprio
stop, il motore non ha visibilità oltre la chiusura di barra corrente
per giudicare se convenga). Le posizioni aperte continuano a essere
gestite dai loro stop/target/max_holding normali, esattamente come
oggi — cambia SOLO il momento in cui si attiva il blocco nuovi ordini,
ora anche a bar-by-bar invece che solo a chiusura trade.

LIMITE NOTO (non risolvibile con questi dati): il check avviene una
volta per barra (30min), non istante per istante — un affondo
temporaneo del prezzo che rientra PRIMA della chiusura della barra
successiva non viene visto. Discusso esplicitamente in chat: risolvibile
solo con un feed a risoluzione più fine (tick o comunque sub-30min),
non disponibile in Fase 1 backtest storico. Vedi nota Fase 2 nel
riepilogo sessione precedente (Streaming API Lightstreamer, solo per
operatività futura in tempo reale, non per backtest storico).

Sanity check obbligatorio incluso in fondo al file: con soglia
impostata a un valore irraggiungibile (es. 0.99), questa sottoclasse
deve produrre risultati IDENTICI al motore standard.
"""

from __future__ import annotations

import pandas as pd
import engine as eng


class BacktestEngineFloatingKillSwitch(eng.BacktestEngine):

    def _floating_loss_pct(self, current_bars: dict[str, pd.Series]) -> float:
        """Perdita (realizzato oggi + non realizzato aperto) come frazione
        POSITIVA del capitale a inizio giornata. 0.0 se in utile o pari."""
        floating_pnl = 0.0
        for pos in self.open_positions:
            bar = current_bars.get(pos.instrument)
            if bar is None:
                continue
            close_price = bar["close"]
            if pos.direction == "long":
                floating_pnl += (close_price - pos.entry_price) * pos.size
            else:
                floating_pnl += (pos.entry_price - close_price) * pos.size

        realized_change = self.capital - self._day_start_capital
        total_change = realized_change + floating_pnl
        if self._day_start_capital == 0:
            return 0.0
        pct = total_change / self._day_start_capital
        return abs(pct) if pct < 0 else 0.0

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

            # ── ESTENSIONE: check floating loss ad ogni barra, non solo a chiusura trade ──
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
            # ── fine estensione ──

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


if __name__ == "__main__":
    # Sanity check standalone: con soglia irraggiungibile, deve produrre
    # risultati identici al motore standard. Va eseguito con dati reali
    # prima di consegnare (vedi workflow dedicato).
    import sys
    print("Questo file va importato come modulo dal workflow di test, "
          "non eseguito direttamente. Vedi floating_kill_switch_test.py")
    sys.exit(0)
