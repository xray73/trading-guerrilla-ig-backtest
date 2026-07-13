"""
engine.py — Motore di backtest v2, Agente AI Trading Guerrilla IG
====================================================================

Allineato alle decisioni FINALI del 13/07/2026 (Project Charter sez. 3,10
e RCA_Fase1_Segnale.md sez. 10-15). Sostituisce ogni versione precedente.

Universo asset attivo: DAX, FTSE 100 (soli tradable=True).
US 500: escluso (segnale più debole + vincolo size minima). Config presente
solo come riferimento futuro, tradable=False.
EUR/USD, GBP/USD: sospesi dal trading attivo, tradable=False.

Segnale adottato — Variante 6 "Breakout + trend ampio" (RCA sez. 11):
    - Direzione:      EMA20 vs EMA50 (30 min) + ADX(14) > 20
    - Trigger entry:  chiusura oltre massimo/minimo delle ultime N barre
                       (N=20 per DAX, N=40 per FTSE100 — RCA sez. 13)
    - Trend ampio:    EMA100 vs EMA200 deve concordare con la direzione
    - NESSUN filtro di maturità trend (rimosso — peggiorava sotto motore
      corretto, RCA sez. 11)

Moltiplicatore ATR: 1.5x per DAX e FTSE100 (RCA sez. 12, pattern monotono
più stretto = meglio sotto motore corretto). ATR/ADX: metodo Wilder, periodo 14.

Risk management (Charter sez. 3, aggiornato 13/07):
    - Rischio per trade:     DAX 2.0%, FTSE100 1.5% (per-strumento, non più
                              uniforme — score di convinzione resta disattivo)
    - R:R minimo 1:1.5, target 1:2, identico long/short
    - Kill switch giornaliero: -4% del capitale del giorno
    - Posizioni concorrenti: max 2 totali, MAI 2 sullo stesso strumento
                              (bug di stacking corretto — vedi Charter sez. 10)
    - Nuovi ordini/giorno:    max 3
    - Holding massimo:       1 giorno (48 barre da 30 min)
    - Priorità setup multipli: 1) R:R più alto, 2) bassa correlazione con
                                posizioni già aperte

Size minima negoziabile (Charter sez. 3, nuovo 13/07):
    Se la size calcolata dal rischio è sotto il minimo negoziabile IG
    (0.50 per DAX/FTSE100), si FORZA la size al minimo (arrotondamento per
    eccesso) — non si salta il trade, non si compensa alzando il rischio
    altrove. Decisione basata su confronto diretto di 4 alternative
    (RCA sez. 15): "forza al minimo" nettamente la migliore.

Controllo margine (nuovo 13/07): margine richiesto = size * prezzo *
    point_value * margin_pct (5% per DAX/FTSE100). Se il margine disponibile
    non basta, la size viene ridotta di conseguenza.

Anti look-ahead bias: il segnale è calcolato sulla barra N (dati chiusi),
l'ingresso avviene SEMPRE all'apertura della barra N+1.

Output: DataFrame trades / run_metrics con nomi colonna compatibili con lo
schema D1 reale (tabelle trades, run_metrics, backtest_runs) verificato via
PRAGMA table_info — vedi funzioni export_trades_for_d1() / export_metrics_for_d1().
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Optional, Literal
import numpy as np
import pandas as pd


# =====================================================================
# 1. CONFIGURAZIONE STRUMENTI — Charter sez. 3, 4, 13, 14, 15
# =====================================================================

@dataclass
class InstrumentConfig:
    name: str
    tradable: bool
    breakout_lookback: int          # 20 DAX, 40 FTSE100 (RCA sez. 13)
    atr_multiplier: float           # 1.5x per entrambi (RCA sez. 12)
    risk_pct: float                 # 2.0% DAX, 1.5% FTSE100 (Charter sez. 3)
    point_value: float              # EUR per punto per unità di size, verificato IG
    spread_fixed: float             # punti — "da riverificare" (Charter sez. 3),
                                     # valori correnti da RCA sez. 5 generalizzazione
    min_tradable_size: float        # 0.50 per DAX/FTSE100, verificato "Get Info" IG
    margin_pct: float               # 5% per DAX/FTSE100, verificato IG


INSTRUMENTS: dict[str, InstrumentConfig] = {
    "DAX": InstrumentConfig(
        name="DAX", tradable=True,
        breakout_lookback=20, atr_multiplier=1.5, risk_pct=0.020,
        point_value=1.0, spread_fixed=1.2,
        min_tradable_size=0.50, margin_pct=0.05,
    ),
    "FTSE100": InstrumentConfig(
        name="FTSE100", tradable=True,
        breakout_lookback=40, atr_multiplier=1.5, risk_pct=0.015,
        point_value=1.0, spread_fixed=1.0,
        min_tradable_size=0.50, margin_pct=0.05,
    ),
    # US500: escluso dall'universo attivo (segnale più debole + vincolo size
    # minima 1.0 — Charter sez. 3, 14). Config qui SOLO come riferimento
    # futuro; tradable=False impedisce al motore di generare trade.
    # Se mai riattivato: lookback 20 + filtro sessione USA 13-21 UTC + ATR 2.5x.
    "US500": InstrumentConfig(
        name="US500", tradable=False,
        breakout_lookback=20, atr_multiplier=2.5, risk_pct=0.0,
        point_value=1.0, spread_fixed=1.50,
        min_tradable_size=1.0, margin_pct=0.05,
    ),
}

# EUR/USD, GBP/USD: sospesi dal trading attivo (RCA sez. 4), non inclusi qui
# — questo motore non li tratta nemmeno come riferimento passivo; il
# buy&hold di confronto va calcolato separatamente sui CSV storici.


# =====================================================================
# 2. PARAMETRI OPERATIVI NON-NEGOZIABILI — Charter sez. 3
# =====================================================================

@dataclass
class ChartaParams:
    ema_fast: int = 20
    ema_slow: int = 50
    ema_broad_fast: int = 100
    ema_broad_slow: int = 200
    adx_period: int = 14
    adx_min_context: float = 20.0

    atr_period: int = 14
    rr_target: float = 2.0
    rr_minimum: float = 1.5

    kill_switch_pct: float = 0.04       # -4% (estremo conservativo del range -4/-5%)
    max_concurrent_positions: int = 2   # totali, mai 2 sullo stesso strumento
    max_new_orders_per_day: int = 3
    max_holding_bars: int = 48          # 24h a barre da 30 min

    bar_minutes: int = 30


PARAMS = ChartaParams()


# =====================================================================
# 3. INDICATORI — EMA standard, ATR/ADX metodo Wilder
# =====================================================================

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def wilder_smooth(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(alpha=1.0 / period, adjust=False).mean()


def atr_wilder(df: pd.DataFrame, period: int) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return wilder_smooth(tr, period)


def adx_wilder(df: pd.DataFrame, period: int) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)

    prev_close = close.shift(1)
    tr_raw = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    tr_smooth = wilder_smooth(tr_raw, period)

    plus_di = 100 * wilder_smooth(plus_dm, period) / tr_smooth.replace(0, np.nan)
    minus_di = 100 * wilder_smooth(minus_dm, period) / tr_smooth.replace(0, np.nan)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return wilder_smooth(dx.fillna(0), period)


def compute_indicators(df: pd.DataFrame, inst: InstrumentConfig,
                        p: ChartaParams = PARAMS) -> pd.DataFrame:
    """df deve avere colonne: timestamp, open, high, low, close (ordinate)."""
    out = df.copy().reset_index(drop=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)

    out["ema_fast"] = ema(out["close"], p.ema_fast)
    out["ema_slow"] = ema(out["close"], p.ema_slow)
    out["ema_broad_fast"] = ema(out["close"], p.ema_broad_fast)
    out["ema_broad_slow"] = ema(out["close"], p.ema_broad_slow)
    out["adx"] = adx_wilder(out, p.adx_period)
    out["atr"] = atr_wilder(out, p.atr_period)

    # Massimo/minimo delle ultime N barre ESCLUSA la barra corrente
    # (shift(1) prima del rolling) — lookback specifico per strumento.
    out["rolling_high"] = out["high"].shift(1).rolling(inst.breakout_lookback).max()
    out["rolling_low"] = out["low"].shift(1).rolling(inst.breakout_lookback).min()

    out["hour_utc"] = out["timestamp"].dt.hour
    return out


# =====================================================================
# 4. SEGNALE — Variante 6 "Breakout + trend ampio" (RCA sez. 11)
# =====================================================================

def generate_signals(df: pd.DataFrame, inst: InstrumentConfig,
                      p: ChartaParams = PARAMS) -> pd.DataFrame:
    """Calcola il segnale sulla barra N (dati chiusi). Colonna `signal`:
    'long', 'short' o None. L'ingresso reale avviene alla barra N+1.
    """
    out = compute_indicators(df, inst, p)

    direction_long = out["ema_fast"] > out["ema_slow"]
    direction_short = out["ema_fast"] < out["ema_slow"]
    adx_context_ok = out["adx"] > p.adx_min_context

    breakout_long = out["close"] > out["rolling_high"]
    breakout_short = out["close"] < out["rolling_low"]

    broad_trend_long_ok = out["ema_broad_fast"] > out["ema_broad_slow"]
    broad_trend_short_ok = out["ema_broad_fast"] < out["ema_broad_slow"]

    # NESSUN filtro di maturità trend (rimosso — RCA sez. 11)
    long_signal = direction_long & adx_context_ok & breakout_long & broad_trend_long_ok
    short_signal = direction_short & adx_context_ok & breakout_short & broad_trend_short_ok

    out["signal"] = None
    out.loc[long_signal, "signal"] = "long"
    out.loc[short_signal, "signal"] = "short"
    return out


# =====================================================================
# 5. POSIZIONI / TRADE
# =====================================================================

@dataclass
class Position:
    instrument: str
    direction: Literal["long", "short"]
    entry_bar_index: int
    entry_time: pd.Timestamp
    entry_price: float
    stop_loss: float
    take_profit: float
    size: float
    risk_amount: float          # € nominalmente rischiati (capitale * risk_pct)
    atr_at_entry: float
    adx_at_entry: float
    rr_planned: float
    forced_min_size: bool        # True se la size è stata forzata al minimo
    max_holding_bars: int


@dataclass
class ClosedTrade:
    instrument: str
    direction: str
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: pd.Timestamp
    exit_price: float
    exit_reason: str
    size: float
    risk_amount: float
    pnl: float
    r_multiple: float
    atr_at_entry: float
    adx_at_entry: float
    rr_planned: float
    forced_min_size: bool
    signal_trigger: str = "solo tecnica"
    causa_esito: Optional[str] = None
    rispetto_regole: str = "si"
    contesto_mercato: Optional[str] = None


def _causa_esito_se_perdita(exit_reason: str, pnl: float) -> Optional[str]:
    if pnl >= 0:
        return None
    mapping = {
        "stop_loss": "falso segnale",
        "max_holding": "timing tardivo",
        "kill_switch_flat": "evento imprevisto",
    }
    return mapping.get(exit_reason, "falso segnale")


def _contesto_mercato(adx_at_entry: float, p: ChartaParams = PARAMS) -> str:
    return "trend" if adx_at_entry > p.adx_min_context else "laterale"


# =====================================================================
# 6. MOTORE DI BACKTEST — DAX + FTSE100 in parallelo, niente stacking
# =====================================================================

class BacktestEngine:
    """Simula il motore su DAX e FTSE100 in parallelo, rispettando:
    - tetto posizioni concorrenti totali (2), MAI 2 sullo stesso strumento
      (1 posizione max per strumento alla volta — fix del bug di stacking,
      Charter sez. 10)
    - tetto ordini/giorno, kill switch giornaliero
    - priorità R:R più alto, poi bassa correlazione (qui: strumenti diversi
      = correlazione già minimizzata per costruzione)
    - size minima forzata, controllo margine
    """

    def __init__(self, capital0: float, p: ChartaParams = PARAMS,
                 instruments: dict[str, InstrumentConfig] = INSTRUMENTS):
        self.capital0 = capital0
        self.capital = capital0
        self.p = p
        self.instruments = instruments
        self.open_positions: list[Position] = []
        self.closed_trades: list[ClosedTrade] = []
        self.equity_curve: list[tuple] = []

        self._day_start_capital = capital0
        self._current_day = None
        self._orders_today = 0
        self._kill_switch_active = False

        # contatori diagnostici (non nel Charter, utili per RCA)
        self.n_forced_min_size = 0
        self.n_margin_reduced = 0

    def _reset_day_if_needed(self, ts: pd.Timestamp):
        day = ts.date()
        if self._current_day != day:
            self._current_day = day
            self._day_start_capital = self.capital
            self._orders_today = 0
            self._kill_switch_active = False

    def _daily_pnl_pct(self) -> float:
        if self._day_start_capital == 0:
            return 0.0
        return (self.capital - self._day_start_capital) / self._day_start_capital

    def _check_kill_switch(self):
        if self._daily_pnl_pct() <= -self.p.kill_switch_pct:
            self._kill_switch_active = True

    def _position_size(self, entry_price: float, stop_price: float,
                        inst: InstrumentConfig) -> tuple[float, float, bool, bool]:
        """Ritorna (size, risk_amount, forced_min_size, margin_reduced).

        Size da rischio %, poi:
        1) se sotto min_tradable_size -> forza al minimo (arrotonda per eccesso)
        2) controlla margine disponibile -> riduce se necessario
        """
        risk_amount = self.capital * inst.risk_pct
        risk_distance = abs(entry_price - stop_price)
        if risk_distance <= 0:
            return 0.0, 0.0, False, False

        size = risk_amount / (risk_distance * inst.point_value)
        forced_min_size = False
        if size < inst.min_tradable_size:
            size = inst.min_tradable_size
            forced_min_size = True

        # controllo margine: margine richiesto = size * prezzo * point_value * margin_pct
        margin_required = size * entry_price * inst.point_value * inst.margin_pct
        margin_reduced = False
        if margin_required > self.capital:
            # riduce la size al massimo consentito dal margine disponibile
            max_size_by_margin = self.capital / (entry_price * inst.point_value * inst.margin_pct)
            if max_size_by_margin < size:
                size = max(max_size_by_margin, 0.0)
                margin_reduced = True

        return size, risk_amount, forced_min_size, margin_reduced

    def _correlation_penalty(self, candidate_instrument: str) -> int:
        """Proxy di correlazione: penalità alta se già aperta una posizione
        sullo stesso strumento (comunque impossibile per costruzione, vedi
        _try_open) o su un altro indice azionario europeo."""
        penalty = 0
        for pos in self.open_positions:
            if pos.instrument == candidate_instrument:
                penalty += 2
            else:
                penalty += 1  # DAX/FTSE100 sono entrambi indici azionari europei
        return penalty

    def _try_close_position(self, pos: Position, bar: pd.Series, bar_index: int,
                             inst: InstrumentConfig) -> bool:
        high, low = bar["high"], bar["low"]
        spread = inst.spread_fixed
        exit_reason = None
        exit_price = None

        if pos.direction == "long":
            if low <= pos.stop_loss:
                exit_reason, exit_price = "stop_loss", pos.stop_loss - spread / 2
            elif high >= pos.take_profit:
                exit_reason, exit_price = "take_profit", pos.take_profit - spread / 2
        else:
            if high >= pos.stop_loss:
                exit_reason, exit_price = "stop_loss", pos.stop_loss + spread / 2
            elif low <= pos.take_profit:
                exit_reason, exit_price = "take_profit", pos.take_profit + spread / 2

        bars_held = bar_index - pos.entry_bar_index
        if exit_reason is None and bars_held >= pos.max_holding_bars:
            exit_reason = "max_holding"
            exit_price = (bar["close"] - spread / 2 if pos.direction == "long"
                          else bar["close"] + spread / 2)

        if exit_reason is None:
            return False

        self._close_position(pos, bar["timestamp"], exit_price, exit_reason)
        return True

    def _close_position(self, pos: Position, exit_time, exit_price: float,
                         exit_reason: str):
        if pos.direction == "long":
            pnl = (exit_price - pos.entry_price) * pos.size
        else:
            pnl = (pos.entry_price - exit_price) * pos.size

        r_multiple = pnl / pos.risk_amount if pos.risk_amount else 0.0
        self.capital += pnl

        trade = ClosedTrade(
            instrument=pos.instrument, direction=pos.direction,
            entry_time=pos.entry_time, entry_price=pos.entry_price,
            exit_time=exit_time, exit_price=exit_price, exit_reason=exit_reason,
            size=pos.size, risk_amount=pos.risk_amount, pnl=pnl,
            r_multiple=r_multiple, atr_at_entry=pos.atr_at_entry,
            adx_at_entry=pos.adx_at_entry, rr_planned=pos.rr_planned,
            forced_min_size=pos.forced_min_size,
            causa_esito=_causa_esito_se_perdita(exit_reason, pnl),
            contesto_mercato=_contesto_mercato(pos.adx_at_entry, self.p),
        )
        self.closed_trades.append(trade)
        self.open_positions.remove(pos)
        self._check_kill_switch()

    def _open_position(self, instrument: str, direction: str, bar: pd.Series,
                        atr_at_entry: float, adx_at_entry: float):
        inst = self.instruments[instrument]
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

        size, risk_amount, forced_min, margin_reduced = self._position_size(
            entry_price, stop_loss, inst)
        if size <= 0:
            return
        if forced_min:
            self.n_forced_min_size += 1
        if margin_reduced:
            self.n_margin_reduced += 1

        pos = Position(
            instrument=instrument, direction=direction,
            entry_bar_index=bar.name, entry_time=bar["timestamp"],
            entry_price=entry_price, stop_loss=stop_loss, take_profit=take_profit,
            size=size, risk_amount=risk_amount, atr_at_entry=atr_at_entry,
            adx_at_entry=adx_at_entry, rr_planned=self.p.rr_target,
            forced_min_size=forced_min, max_holding_bars=self.p.max_holding_bars,
        )
        self.open_positions.append(pos)
        self._orders_today += 1

    def run(self, data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
        """data: {instrument_name: df con colonne signal/indicatori già
        calcolate da generate_signals()}. Solo strumenti tradable=True in
        self.instruments generano trade, anche se presenti in `data`.
        """
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

            # 1) gestisci uscite prima di valutare nuovi ingressi
            for pos in list(self.open_positions):
                inst_df = data[pos.instrument]
                row = inst_df.loc[inst_df["timestamp"] == ts]
                if row.empty:
                    continue
                bar = row.iloc[0]
                bar_index = row.index[0]
                self._try_close_position(pos, bar, bar_index, self.instruments[pos.instrument])

            self.equity_curve.append((ts, self.capital))

            if self._kill_switch_active:
                continue
            if self._orders_today >= self.p.max_new_orders_per_day:
                continue
            if len(self.open_positions) >= self.p.max_concurrent_positions:
                continue

            # 2) raccogli candidati con segnale su questa barra (calcolato
            #    sulla barra precedente, eseguito all'apertura di questa)
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
                # MAI 2 posizioni sullo stesso strumento (fix bug stacking)
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

            # 3) priorità: R:R più alto (identico qui), poi bassa correlazione
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
        metrics_df = compute_run_metrics(trades_df, self.capital0, self.capital)
        return trades_df, metrics_df

    def trades_to_dataframe(self) -> pd.DataFrame:
        rows = [dataclasses.asdict(t) for t in self.closed_trades]
        return pd.DataFrame(rows)


# =====================================================================
# 7. METRICHE DI RUN — Charter sez. 6
# =====================================================================

def compute_run_metrics(trades_df: pd.DataFrame, capital0: float,
                         capital_final: float) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame([{
            "num_trades": 0, "win_rate": np.nan, "profit_factor": np.nan,
            "expectancy": np.nan, "max_drawdown_pct": np.nan,
            "pnl_total": 0.0, "capital_final": capital_final,
        }])

    wins = trades_df[trades_df["pnl"] > 0]
    losses = trades_df[trades_df["pnl"] <= 0]

    win_rate = len(wins) / len(trades_df)
    gross_profit = wins["pnl"].sum()
    gross_loss = -losses["pnl"].sum()
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
    expectancy = trades_df["pnl"].mean()

    equity = capital0 + trades_df["pnl"].cumsum()
    running_max = equity.cummax()
    drawdown_pct = (equity - running_max) / running_max
    max_drawdown_pct = drawdown_pct.min()

    return pd.DataFrame([{
        "num_trades": len(trades_df),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "max_drawdown_pct": max_drawdown_pct,
        "pnl_total": trades_df["pnl"].sum(),
        "capital_final": capital_final,
        "significativo": len(trades_df) >= 30,
    }])


# =====================================================================
# 8. CARICAMENTO DATI
# =====================================================================

def load_ohlc_csv(path: str) -> pd.DataFrame:
    """Colonne attese: timestamp, open, high, low, close, volume (formato
    export Dukascopy)."""
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


# =====================================================================
# 9. EXPORT PER D1 — nomi colonna compatibili con schema reale
# =====================================================================

def export_trades_for_d1(trades_df: pd.DataFrame, run_id: int) -> pd.DataFrame:
    """Rimappa le colonne di trades_df sullo schema D1 `trades`:
    run_id, symbol, direction, entry_time, entry_price, exit_time,
    exit_price, stop_loss, take_profit, atr_at_entry, risk_pct, pnl,
    rr_realized, signal_trigger, rispetto_regole, causa_esito,
    contesto_mercato, exit_reason.
    NB: stop_loss/take_profit non sono salvati esplicitamente nel motore
    per-trade (solo entry/exit) — vengono ricalcolati qui da atr_at_entry
    per completezza dello schema.
    """
    if trades_df.empty:
        return trades_df
    out = pd.DataFrame()
    out["run_id"] = [run_id] * len(trades_df)
    out["symbol"] = trades_df["instrument"]
    out["direction"] = trades_df["direction"]
    out["entry_time"] = trades_df["entry_time"].astype(str)
    out["entry_price"] = trades_df["entry_price"]
    out["exit_time"] = trades_df["exit_time"].astype(str)
    out["exit_price"] = trades_df["exit_price"]
    out["atr_at_entry"] = trades_df["atr_at_entry"]
    out["risk_pct"] = trades_df["risk_amount"]  # importo €, non %, vedi nota schema
    out["pnl"] = trades_df["pnl"]
    out["rr_realized"] = trades_df["r_multiple"]
    out["signal_trigger"] = trades_df["signal_trigger"]
    out["rispetto_regole"] = trades_df["rispetto_regole"]
    out["causa_esito"] = trades_df["causa_esito"]
    out["contesto_mercato"] = trades_df["contesto_mercato"]
    out["exit_reason"] = trades_df["exit_reason"]
    return out


def export_metrics_for_d1(metrics_df: pd.DataFrame, run_id: int,
                           baseline_type: str, period_label: str) -> pd.DataFrame:
    if metrics_df.empty:
        return metrics_df
    out = pd.DataFrame()
    out["run_id"] = [run_id] * len(metrics_df)
    out["baseline_type"] = baseline_type
    out["win_rate"] = metrics_df["win_rate"]
    out["profit_factor"] = metrics_df["profit_factor"].replace(np.inf, None)
    out["expectancy"] = metrics_df["expectancy"]
    out["max_drawdown"] = metrics_df["max_drawdown_pct"]
    out["num_trades"] = metrics_df["num_trades"]
    out["period_label"] = period_label
    return out


# =====================================================================
# 10. ESEMPIO D'USO
# =====================================================================

def run_backtest_dax_ftse100(paths: dict[str, str], capital0: float = 900.0
                              ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """paths: {"DAX": "DAX_2023_30m.csv", "FTSE100": "FTSE100_2023_30m.csv"}
    Solo DAX e FTSE100 accettati (unici tradable=True nell'universo attivo).
    """
    data = {}
    for name, path in paths.items():
        inst = INSTRUMENTS.get(name)
        if inst is None or not inst.tradable:
            raise ValueError(f"{name} non è nell'universo attivo (Charter sez. 3).")
        raw = load_ohlc_csv(path)
        data[name] = generate_signals(raw, inst)

    engine = BacktestEngine(capital0=capital0)
    trades_df, metrics_df = engine.run(data)
    return trades_df, metrics_df


if __name__ == "__main__":
    import sys
    print(__doc__)
    print("Esempio: run_backtest_dax_ftse100({'DAX': 'DAX_2023_30m.csv', "
          "'FTSE100': 'FTSE100_2023_30m.csv'})", file=sys.stderr)
