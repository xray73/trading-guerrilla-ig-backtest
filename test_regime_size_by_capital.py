"""
test_regime_size_by_capital.py — Isola l'effetto della size-modulation
(dimezzare il rischio in regime correlazione alta) PER LIVELLO DI
CAPITALE, non per periodo calendariale — il test precedente
(test_regime_size_modulation.py) confonde le due dimensioni perche' i
5 periodi ufficiali partono tutti da capital0=1400 fisso, quindi non
puo' mostrare l'effetto "il vincolo di size minima si attenua quando
il capitale cresce" (ipotesi dell'utente, chat 22/07/2026).

Fa girare la STESSA sottoclasse (BacktestEngineRegimeSized) a capitali
di partenza crescenti (1400/3000/6000/10000/20000 EUR) sullo stesso
periodo (2024-2025, il piu' ricco di trade tra i 5 ufficiali, buon
compromesso tra volume campione e rappresentativita'), per isolare
pulitamente l'effetto capitale dall'effetto periodo.

Per ogni livello: delta PnL vs baseline, % di trade "ad alta
correlazione" che finiscono comunque forzati al minimo negoziabile
(l'effetto che annulla la riduzione, quantificato direttamente).
"""
import os
import pandas as pd
import requests

import engine as eng
from engine_floating_kill_switch import BacktestEngineFloatingKillSwitch
from ohlc_data_source import get_ohlc

CF_ACCOUNT_ID = os.environ["CLOUDFLARE_ACCOUNT_ID"]
CF_API_TOKEN = os.environ["CLOUDFLARE_API_TOKEN"]
D1_ID = "b9fbd4d6-7837-4d86-9c0f-ca60c0cf69e3"

CORR_THRESHOLD = 0.70
SIZE_MULTIPLIER_HIGH = 0.5
CAPITAL_LEVELS = [1400, 3000, 6000, 10000, 20000]
PERIOD = ("2024-2025", "2024-01-01", "2025-12-31")


def d1(sql):
    url = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/d1/database/{D1_ID}/query"
    headers = {"Authorization": f"Bearer {CF_API_TOKEN}", "Content-Type": "application/json"}
    resp = requests.post(url, json={"sql": sql}, headers=headers, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if not data.get("success"):
        raise RuntimeError(data.get("errors"))
    return data["result"][0]["results"]


class BacktestEngineRegimeSized(BacktestEngineFloatingKillSwitch):
    """Identica a quella di test_regime_size_modulation.py — vedi quel
    file per la documentazione completa. Qui aggiunto un contatore
    diagnostico n_high_corr_forced_min per quantificare l'effetto
    "forzato al minimo annulla la riduzione"."""

    def __init__(self, capital0, regime_lookup: pd.Series, **kwargs):
        super().__init__(capital0, **kwargs)
        self.regime_lookup = regime_lookup
        self._current_entry_time = None
        self.n_high_corr_trades = 0
        self.n_high_corr_forced_min = 0

    def _lookup_corr(self, ts):
        idx = self.regime_lookup.index.searchsorted(ts, side="right") - 1
        if idx < 0:
            return None
        return self.regime_lookup.iloc[idx]

    def _open_position(self, instrument, direction, bar, atr_at_entry, adx_at_entry):
        self._current_entry_time = bar["timestamp"]
        super()._open_position(instrument, direction, bar, atr_at_entry, adx_at_entry)

    def _position_size(self, entry_price, stop_price, inst):
        risk_amount = self.capital * inst.risk_pct

        corr_val = self._lookup_corr(self._current_entry_time)
        is_high_corr = corr_val is not None and corr_val >= CORR_THRESHOLD
        if is_high_corr:
            risk_amount *= SIZE_MULTIPLIER_HIGH
            self.n_high_corr_trades += 1

        risk_distance = abs(entry_price - stop_price)
        if risk_distance <= 0:
            return 0.0, 0.0, False, False

        size = risk_amount / (risk_distance * inst.point_value)
        forced_min_size = False
        if size < inst.min_tradable_size:
            size = inst.min_tradable_size
            forced_min_size = True
            if is_high_corr:
                self.n_high_corr_forced_min += 1

        margin_required = size * entry_price * inst.point_value * inst.margin_pct
        margin_reduced = False
        if margin_required > self.capital:
            max_size_by_margin = self.capital / (entry_price * inst.point_value * inst.margin_pct)
            if max_size_by_margin < size:
                size = max(max_size_by_margin, 0.0)
                margin_reduced = True

        return size, risk_amount, forced_min_size, margin_reduced


def slice_period(signals, start, end):
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    return signals[(signals["timestamp"] >= start_ts) & (signals["timestamp"] < end_ts)].reset_index(drop=True)


def main():
    print("Scarico OHLC continuo DAX+FTSE100...")
    hist = {name: get_ohlc(name, CF_ACCOUNT_ID, CF_API_TOKEN) for name in ("DAX", "FTSE100")}

    print("Genero segnali V6...")
    signals = {name: eng.generate_signals(hist[name], eng.INSTRUMENTS[name]) for name in hist}

    print("Scarico corr_dax_ftse_7d da market_regime_indicators...")
    rows = d1("SELECT timestamp, corr_dax_ftse_7d FROM market_regime_indicators "
              "WHERE instrument='DAX' AND corr_dax_ftse_7d IS NOT NULL ORDER BY timestamp ASC")
    regime_lookup = pd.Series(
        [r["corr_dax_ftse_7d"] for r in rows],
        index=pd.to_datetime([r["timestamp"] for r in rows], utc=True)
    )
    print(f"  {len(regime_lookup)} punti di regime caricati")

    period_name, start, end = PERIOD
    sliced = {name: slice_period(sig, start, end) for name, sig in signals.items()}

    print(f"\n=== Periodo {period_name}, capitali crescenti ===\n")
    print(f"{'Capitale':>10}{'PnL base':>14}{'PnL mod':>14}{'Delta':>12}{'Delta %':>10}"
          f"{'Trade alta-corr':>16}{'...forzati min':>16}{'% annullati':>13}")

    for cap in CAPITAL_LEVELS:
        baseline_engine = BacktestEngineFloatingKillSwitch(capital0=cap)
        baseline_trades, _ = baseline_engine.run(sliced)
        baseline_pnl = float(baseline_trades["pnl"].sum()) if len(baseline_trades) else 0.0

        regime_engine = BacktestEngineRegimeSized(capital0=cap, regime_lookup=regime_lookup)
        regime_trades, _ = regime_engine.run(sliced)
        regime_pnl = float(regime_trades["pnl"].sum()) if len(regime_trades) else 0.0

        delta = regime_pnl - baseline_pnl
        delta_pct = (delta / cap) * 100 if cap else 0.0
        n_high = regime_engine.n_high_corr_trades
        n_forced = regime_engine.n_high_corr_forced_min
        pct_annullati = (n_forced / n_high * 100) if n_high else 0.0

        print(f"{cap:>10}{baseline_pnl:>14.2f}{regime_pnl:>14.2f}{delta:>12.2f}{delta_pct:>9.2f}%"
              f"{n_high:>16}{n_forced:>16}{pct_annullati:>12.1f}%")


if __name__ == "__main__":
    main()
