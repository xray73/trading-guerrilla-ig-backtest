"""
engine.py — Motore di backtest, Agente AI Trading Guerrilla IG
================================================================

Riscrittura da zero (2026-07-12) con i parametri validati in Fase 1
(vedi Project_Charter_Agente_Trading_Guerrilla_IG.md sez. 3, 10 e
RCA_Fase1_Segnale.md sez. 3, 5, 8).

Universo asset attivo (trading reale nel motore):
    US 500, DAX, FTSE 100  (segnale validato e generalizzato, RCA sez. 3/5)

EUR/USD, GBP/USD: SOSPESI dal trading attivo (RCA sez. 4). Restano
utilizzabili solo come riferimento passivo buy&hold — questo motore
non genera trade su questi due strumenti; il flag `tradable=False`
nell'InstrumentConfig lo impone esplicitamente.

Segnale validato ("Breakout + trend ampio + maturità trend"):
    - Direzione:      EMA20 vs EMA50 (30 min) + ADX(14) > 20
    - Trigger entry:  chiusura oltre massimo/minimo delle ultime 20 barre
    - Trend ampio:    EMA100 vs EMA200 deve concordare con la direzione
    - Maturità trend: pendenza ADX su 10 barre non in calo marcato (> -2)

Moltiplicatore ATR: 3.0x (aggiornato da 2.0x il 12/07/2026 — RCA sez. 8.1,
robustezza su 18/30 combinazioni indice/periodo, non "ottimalità").
ATR/ADX: metodo Wilder, periodo 14.

Risk management (Project Charter sez. 3):
    - Rischio per trade:            1% fisso del capitale corrente
      (score di convinzione NON attivato — RCA #9, peggiora sempre nei test)
    - R:R minimo 1:1.5, target 1:2, identico long/short
    - Kill switch giornaliero:      -4% / -5% del capitale del giorno
    - Posizioni concorrenti:        max 1-2
    - Nuovi ordini/giorno:          max 2-3
    - Holding massimo:              1 giorno (48 barre da 30 min)
      (eccezione 2 notti per alta convinzione non applicabile: score disattivo)
    - Priorità setup multipli:      1) R:R più alto, 2) bassa correlazione
      con posizioni già aperte

Nota importante anti look-ahead bias (vedi RCA sez. 6.1 — bug già
trovato e corretto in passato su un altro modulo di questo progetto):
il segnale è calcolato sulla barra N (dati chiusi), l'ingresso avviene
SEMPRE all'apertura della barra N+1. Mai usare dati della barra
corrente per decidere l'ingresso sulla barra corrente.

Output: log per-trade e metriche di run pensati per essere compatibili
con lo schema D1 esistente (tabelle trades, run_metrics, backtest_runs).
Le colonne esatte del D1 reale non sono visibili da questo ambiente —
i nomi qui sotto sono un mapping ragionevole (snake_case, stessi campi
descritti nel Charter sez. 8 "struttura log per trade"); verificare e
allineare i nomi colonna con `wrangler d1 execute ... "PRAGMA table_info(trades)"`
prima dell'INSERT reale, poi eventualmente rinominare nel dict di export
in fondo al file.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Optional, Literal
import numpy as np
import pandas as pd


# =====================================================================
# 1. CONFIGURAZIONE — parametri fissati dal Project Charter (sez. 3)
# =====================================================================

@dataclass
class InstrumentConfig:
    name: str
    tradable: bool                 # False per EUR/USD, GBP/USD (sospesi)
    asset_class: Literal["index", "fx"]
    point_value: float             # valore monetario di 1 punto/pip per 1 unità
                                    # di size=1.0 — DA VERIFICARE su "Get Info" IG
                                    # (Charter sez. 4: nessuna tabella globale pubblica)
    spread_fixed: Optional[float] = None       # punti, per indici (US500/DAX/FTSE100)
    spread_by_hour_utc: Optional[dict] = None  # pip, per FX (solo riferimento)


# Spread model (Charter sez. 3, RCA sez. 1 correzione critica spread):
# FX per fascia oraria UTC — usato SOLO per il buy&hold di riferimento,
# non per generare trade (EUR/USD e GBP/USD sono tradable=False).
FX_SPREAD_BY_HOUR = {
    "13-17": 0.80,
    "8-13": 1.15,
    "17-21": 1.50,
    "21-8": 2.25,
}


def fx_spread_for_hour(hour_utc: int) -> float:
    """Ritorna lo spread in pip per un'ora UTC data, secondo il modello a fasce."""
    if 13 <= hour_utc < 17:
        return FX_SPREAD_BY_HOUR["13-17"]
    if 8 <= hour_utc < 13:
        return FX_SPREAD_BY_HOUR["8-13"]
    if 17 <= hour_utc < 21:
        return FX_SPREAD_BY_HOUR["17-21"]
    return FX_SPREAD_BY_HOUR["21-8"]


INSTRUMENTS: dict[str, InstrumentConfig] = {
    "US500": InstrumentConfig(
        name="US500", tradable=True, asset_class="index",
        point_value=1.0, spread_fixed=1.50,
    ),
    "DAX": InstrumentConfig(
        name="DAX", tradable=True, asset_class="index",
        point_value=1.0, spread_fixed=1.2,
    ),
    "FTSE100": InstrumentConfig(
        name="FTSE100", tradable=True, asset_class="index",
        point_value=1.0, spread_fixed=1.0,
    ),
    "EURUSD": InstrumentConfig(
        name="EURUSD", tradable=False, asset_class="fx",
        point_value=1.0, spread_by_hour_utc=FX_SPREAD_BY_HOUR,
    ),
    "GBPUSD": InstrumentConfig(
        name="GBPUSD", tradable=False, asset_class="fx",
        point_value=1.0, spread_by_hour_utc=FX_SPREAD_BY_HOUR,
    ),
}


@dataclass
class ChartaParams:
    """Parametri operativi non-negoziabili, Project Charter sez. 3."""
    ema_fast: int = 20
    ema_slow: int = 50
    ema_broad_fast: int = 100
    ema_broad_slow: int = 200
    adx_period: int = 14
    adx_min_context: float = 20.0
    adx_maturity_window: int = 10
    adx_maturity_min_slope: float = -2.0      # pendenza ADX su 10 barre non < -2
    breakout_lookback: int = 20               # barre per massimo/minimo recente

    atr_period: int = 14
    atr_multiplier: float = 3.0               # aggiornato 12/07/2026 (RCA sez. 8.1)
    rr_target: float = 2.0                    # target 1:2
    rr_minimum: float = 1.5                   # minimo accettabile 1:1.5

    risk_pct_fixed: float = 0.01              # 1% fisso (score disattivato)
    kill_switch_pct: float = 0.04             # soglia -4% (variante conservativa
                                               # del range -4/-5% del Charter)
    max_concurrent_positions: int = 2
    max_new_orders_per_day: int = 3
    max_holding_bars: int = 48                # 48 barre da 30 min = 1 giorno

    bar_minutes: int = 30


PARAMS = ChartaParams()


# =====================================================================
# 2. INDICATORI — EMA standard, ATR/ADX metodo Wilder
# =====================================================================

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def wilder_smooth(series: pd.Series, period: int) -> pd.Series:
    """Smoothing di Wilder (RMA), equivalente a EMA con alpha=1/period."""
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

    tr = atr_wilder(df, period) * period  # ATR è già smoothed; ricostruiamo TR "grezzo" smussato coerente
    # Nota: per coerenza interna calcoliamo un TR smussato dedicato invece di
    # riusare atr_wilder (che ha la sua stessa formula ma la richiamiamo per
    # chiarezza — nessun impatto numerico, stessa formula di smoothing).
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
    adx = wilder_smooth(dx.fillna(0), period)
    return adx


def compute_indicators(df: pd.DataFrame, p: ChartaParams = PARAMS) -> pd.DataFrame:
    """Aggiunge tutte le colonne indicatore necessarie al segnale validato.

    df deve avere colonne: timestamp (tz-aware UTC), open, high, low, close.
    """
    out = df.copy().reset_index(drop=True)
    out["ema_fast"] = ema(out["close"], p.ema_fast)
    out["ema_slow"] = ema(out["close"], p.ema_slow)
    out["ema_broad_fast"] = ema(out["close"], p.ema_broad_fast)
    out["ema_broad_slow"] = ema(out["close"], p.ema_broad_slow)
    out["adx"] = adx_wilder(out, p.adx_period)
    out["adx_slope"] = out["adx"] - out["adx"].shift(p.adx_maturity_window)
    out["atr"] = atr_wilder(out, p.atr_period)

    # Massimo/minimo delle ultime N barre ESCLUSA la barra corrente
    # (shift(1) prima del rolling) per evitare che il breakout usi il
    # proprio stesso high/low come riferimento.
    out["rolling_high"] = out["high"].shift(1).rolling(p.breakout_lookback).max()
    out["rolling_low"] = out["low"].shift(1).rolling(p.breakout_lookback).min()

    if "timestamp" in out.columns:
        out["hour_utc"] = pd.to_datetime(out["timestamp"]).dt.hour
    return out


# =====================================================================
# 3. SEGNALE — "Breakout + trend ampio + maturità trend" (RCA sez. 3)
# =====================================================================

def generate_signals(df: pd.DataFrame, p: ChartaParams = PARAMS) -> pd.DataFrame:
    """Calcola il segnale sulla barra N (dati chiusi). Colonna `signal`:
    'long', 'short' o None. L'ingresso reale avviene alla barra N+1 (vedi
    motore di simulazione — mai eseguire sulla stessa barra del segnale).
    """
    out = compute_indicators(df, p)

    direction_long = out["ema_fast"] > out["ema_slow"]
    direction_short = out["ema_fast"] < out["ema_slow"]
    adx_context_ok = out["adx"] > p.adx_min_context

    breakout_long = out["close"] > out["rolling_high"]
    breakout_short = out["close"] < out["rolling_low"]

    broad_trend_long_ok = out["ema_broad_fast"] > out["ema_broad_slow"]
    broad_trend_short_ok = out["ema_broad_fast"] < out["ema_broad_slow"]

    maturity_ok = out["adx_slope"] > p.adx_maturity_min_slope

    long_signal = (
        direction_long & adx_context_ok & breakout_long
        & broad_trend_long_ok & maturity_ok
    )
    short_signal = (
        direction_short & adx_context_ok & breakout_short
        & broad_trend_short_ok & maturity_ok
    )

    out["signal"] = None
    out.loc[long_signal, "signal"] = "long"
    out.loc[short_signal, "signal"] = "short"
    return out


# =====================================================================
# 4. POSIZIONI / TRADE
# =====================================================================

@dataclass
class Position:
    instrument: str
    direction: Literal["long", "short"]
    entry_bar_index: int
    entry_time: pd.Timestamp
    entry_price: float           # prezzo eseguito, spread già incluso
    stop_loss: float
    take_profit: float
    size: float                  # unità/contratti
    risk_amount: float           # € rischiati (capitale * risk_pct)
    atr_at_entry: float
    adx_at_entry: float
    rr_planned: float
    max_holding_bars: int


@dataclass
class ClosedTrade:
    instrument: str
    direction: str
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: pd.Timestamp
    exit_price: float
    exit_reason: Literal["stop_loss", "take_profit", "max_holding", "kill_switch_flat"]
    size: float
    risk_amount: float
    pnl: float
    r_multiple: float
    atr_at_entry: float
    adx_at_entry: float
    rr_planned: float
    # campi RCA (Charter sez. 8 — struttura log per trade)
    segnale_trigger: str = "solo_tecnica"          # questo motore non usa notizie
    causa_esito: Optional[str] = None              # valorizzato solo se perdita
    rispetto_regole: bool = True
    contesto_mercato: Optional[str] = None          # es. "trend" / "laterale"


def _causa_esito_se_perdita(exit_reason: str, pnl: float) -> Optional[str]:
    if pnl >= 0:
        return None
    mapping = {
        "stop_loss": "falso_segnale",
        "max_holding": "timing_tardivo",
        "kill_switch_flat": "evento_imprevisto",
    }
    return mapping.get(exit_reason, "falso_segnale")


def _contesto_mercato(adx_at_entry: float) -> str:
    return "trend" if adx_at_entry > PARAMS.adx_min_context else "laterale"


# =====================================================================
# 5. MOTORE DI BACKTEST — singolo o multi-strumento con priorità setup
# =====================================================================

class BacktestEngine:
    """Simula il motore su uno o più strumenti in parallelo, rispettando
    tetto posizioni concorrenti, tetto ordini/giorno e kill switch
    giornaliero, con priorità R:R poi bassa correlazione (Charter sez. 3).
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

    # -- helpers -------------------------------------------------------

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

    def _instrument_spread(self, inst_cfg: InstrumentConfig, hour_utc: int) -> float:
        if inst_cfg.spread_fixed is not None:
            return inst_cfg.spread_fixed
        return fx_spread_for_hour(hour_utc)

    def _position_size(self, entry_price: float, stop_price: float,
                        point_value: float) -> tuple[float, float]:
        """Ritorna (size, risk_amount) per rischio 1% fisso del capitale
        corrente (score di convinzione disattivato — RCA #9)."""
        risk_amount = self.capital * self.p.risk_pct_fixed
        risk_distance = abs(entry_price - stop_price)
        if risk_distance <= 0 or point_value <= 0:
            return 0.0, 0.0
        size = risk_amount / (risk_distance * point_value)
        return size, risk_amount

    def _correlation_penalty(self, candidate_instrument: str) -> int:
        """Proxy semplice di correlazione: stesso strumento o entrambi
        indici azionari = alta correlazione (penalità alta). In assenza
        di una matrice di correlazione calcolata sui rendimenti reali,
        questo è un'approssimazione dichiarata — da sostituire con una
        vera matrice di correlazione rolling se necessario."""
        penalty = 0
        for pos in self.open_positions:
            if pos.instrument == candidate_instrument:
                penalty += 2
            elif (self.instruments[pos.instrument].asset_class
                  == self.instruments[candidate_instrument].asset_class):
                penalty += 1
        return penalty

    # -- gestione posizioni aperte --------------------------------------

    def _try_close_position(self, pos: Position, bar: pd.Series, bar_index: int,
                             inst_cfg: InstrumentConfig, hour_utc: int) -> bool:
        """Verifica stop/target/scadenza holding sulla barra corrente.
        Ritorna True se la posizione è stata chiusa."""
        high, low = bar["high"], bar["low"]
        spread = self._instrument_spread(inst_cfg, hour_utc)
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
            exit_price = bar["close"] - spread / 2 if pos.direction == "long" else bar["close"] + spread / 2

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
            causa_esito=_causa_esito_se_perdita(exit_reason, pnl),
            contesto_mercato=_contesto_mercato(pos.adx_at_entry),
            rispetto_regole=True,
        )
        self.closed_trades.append(trade)
        self.open_positions.remove(pos)
        self._check_kill_switch()

    def _open_position(self, instrument: str, direction: str, bar: pd.Series,
                        atr_at_entry: float, adx_at_entry: float, hour_utc: int):
        inst_cfg = self.instruments[instrument]
        spread = self._instrument_spread(inst_cfg, hour_utc)
        raw_price = bar["open"]
        entry_price = raw_price + spread / 2 if direction == "long" else raw_price - spread / 2

        stop_distance = atr_at_entry * self.p.atr_multiplier
        if direction == "long":
            stop_loss = entry_price - stop_distance
            take_profit = entry_price + stop_distance * self.p.rr_target
        else:
            stop_loss = entry_price + stop_distance
            take_profit = entry_price - stop_distance * self.p.rr_target

        size, risk_amount = self._position_size(entry_price, stop_loss, inst_cfg.point_value)
        if size <= 0:
            return

        pos = Position(
            instrument=instrument, direction=direction,
            entry_bar_index=bar.name, entry_time=bar["timestamp"],
            entry_price=entry_price, stop_loss=stop_loss, take_profit=take_profit,
            size=size, risk_amount=risk_amount, atr_at_entry=atr_at_entry,
            adx_at_entry=adx_at_entry, rr_planned=self.p.rr_target,
            max_holding_bars=self.p.max_holding_bars,
        )
        self.open_positions.append(pos)
        self._orders_today += 1

    # -- loop principale -------------------------------------------------

    def run(self, data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
        """data: {instrument_name: df con colonne signal/indicatori già
        calcolate da generate_signals(), allineate su timeline comune}.
        Solo gli strumenti con tradable=True generano trade; gli altri
        vengono ignorati anche se presenti in `data` (uso per buy&hold
        di riferimento va fatto separatamente, fuori da questo motore).
        """
        tradable_instruments = [
            name for name in data if self.instruments.get(name, InstrumentConfig(
                name, False, "index", 1.0)).tradable
        ]
        if not tradable_instruments:
            raise ValueError("Nessuno strumento tradabile fornito a run().")

        # timeline comune (union degli indici, ordinata)
        all_timestamps = sorted(set().union(
            *[set(data[i]["timestamp"]) for i in tradable_instruments]))

        for ts in all_timestamps:
            self._reset_day_if_needed(ts)
            hour_utc = pd.to_datetime(ts).hour

            # 1) gestisci uscite prima di valutare nuovi ingressi
            for pos in list(self.open_positions):
                inst_df = data[pos.instrument]
                row = inst_df.loc[inst_df["timestamp"] == ts]
                if row.empty:
                    continue
                bar = row.iloc[0]
                bar_index = row.index[0]
                self._try_close_position(pos, bar, bar_index, self.instruments[pos.instrument], hour_utc)

            self.equity_curve.append((ts, self.capital))

            if self._kill_switch_active:
                continue
            if self._orders_today >= self.p.max_new_orders_per_day:
                continue
            if len(self.open_positions) >= self.p.max_concurrent_positions:
                continue

            # 2) raccogli candidati con segnale su questa barra (già chiusa
            #    alla barra precedente: signal calcolato su N, eseguito su N+1)
            candidates = []
            for name in tradable_instruments:
                inst_df = data[name]
                idx = inst_df.index[inst_df["timestamp"] == ts]
                if len(idx) == 0:
                    continue
                i = idx[0]
                if i == 0:
                    continue
                prev_bar = inst_df.iloc[i - 1]   # segnale generato sulla barra precedente
                cur_bar = inst_df.iloc[i]        # esecuzione all'apertura di questa barra
                if prev_bar["signal"] not in ("long", "short"):
                    continue
                already_open = any(p.instrument == name for p in self.open_positions)
                if already_open:
                    continue
                rr = self.p.rr_target  # identico per tutti i setup di questo segnale
                candidates.append({
                    "instrument": name, "direction": prev_bar["signal"],
                    "bar": cur_bar, "atr": prev_bar["atr"], "adx": prev_bar["adx"],
                    "rr": rr,
                })

            if not candidates:
                continue

            # 3) priorità: R:R più alto, poi bassa correlazione con posizioni aperte
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
                                     c["atr"], c["adx"], hour_utc)
                slots_free -= 1

        trades_df = self.trades_to_dataframe()
        metrics_df = compute_run_metrics(trades_df, self.capital0, self.capital)
        return trades_df, metrics_df

    def trades_to_dataframe(self) -> pd.DataFrame:
        rows = [dataclasses.asdict(t) for t in self.closed_trades]
        return pd.DataFrame(rows)


# =====================================================================
# 6. METRICHE DI RUN — Charter sez. 6 "Protocollo di valutazione"
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
        "significativo": len(trades_df) >= 30,  # soglia minima Charter sez. 6
    }])


# =====================================================================
# 7. CARICAMENTO DATI (placeholder da adattare a D1/Colab)
# =====================================================================

def load_ohlc_csv(path: str) -> pd.DataFrame:
    """Carica OHLC 30min da CSV (uso tipico: export da Colab dopo download
    Dukascopy). Colonne attese: timestamp, open, high, low, close.
    Per caricare da Cloudflare D1 invece che da CSV, sostituire questa
    funzione con una query alla tabella `ohlc_prices` (via wrangler o
    binding D1) che restituisca lo stesso schema di colonne.
    """
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


# =====================================================================
# 8. ESEMPIO D'USO
# =====================================================================

def run_backtest_for_instruments(paths: dict[str, str], capital0: float = 900.0
                                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """paths: {"US500": "us500_2024_2025.csv", "DAX": "...", "FTSE100": "..."}
    Solo strumenti tradable=True vanno passati qui (US500/DAX/FTSE100).
    """
    data = {}
    for name, path in paths.items():
        if not INSTRUMENTS[name].tradable:
            raise ValueError(f"{name} è sospeso dal trading attivo (vedi Charter sez. 3, RCA sez. 4).")
        raw = load_ohlc_csv(path)
        data[name] = generate_signals(raw)

    engine = BacktestEngine(capital0=capital0)
    trades_df, metrics_df = engine.run(data)
    return trades_df, metrics_df


if __name__ == "__main__":
    import sys
    print(__doc__)
    print("Esempio: modificare `paths` in run_backtest_for_instruments() con i CSV reali "
          "(export da Colab) ed eseguire. Nessun dato incluso in questo file.", file=sys.stderr)
