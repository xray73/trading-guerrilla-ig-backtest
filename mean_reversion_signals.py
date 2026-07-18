"""
mean_reversion_signals.py — Ramo esplorativo: segnale mean-reversion
pensato per girare IN PARALLELO a Variante 6 (non sostituirla), attivo
nei momenti in cui V6 tende a tacere (mercato laterale).

SPECIFICA CONFERMATA IN CHAT IL 17/07/2026:
  1. Filtro di regime: ADX(14) < 20, letto in modo CONTINUO barra per
     barra (stessa logica di V6, dal lato opposto — non una lettura
     "una tantum" all'apertura come nell'ORB, chiuso). Se ADX>=20 in
     quel momento, nessun segnale mean-reversion su quella barra,
     indipendentemente da cosa fosse successo prima nello stesso giorno.
  2. Nessuna regola whipsaw ad hoc: i limiti già esistenti del motore
     (max_new_orders_per_day, max_concurrent_positions) bastano da
     soli a contenere segnali ripetuti nello stesso giorno — nessuna
     logica duplicata.
  3. USCITE: stop/target ATR-based fissi (stessa formula di V6/ORB:
     stop = ATR*atr_multiplier, target = stop*rr_target), NON un
     ritorno dinamico alla banda/media — il motore riusato invariato
     non supporta target che si spostano nel tempo. Bollinger/RSI
     sono usati SOLO come trigger di ingresso.
  4. RSI: entrata IMMEDIATA al superamento soglia (30/70), non attesa
     di conferma/rientro.

DUE VARIANTI, per confronto:
  mode="bollinger": media mobile 20 barre +/- 2 deviazioni standard.
      Long se chiusura sotto banda inferiore, short se sopra banda
      superiore (sempre con ADX<20 in quel momento).
  mode="rsi": RSI(14). Long se RSI<30, short se RSI>70 (ADX<20).

Riusa atr_wilder/adx_wilder da engine.py (import diretto, nessuna
duplicazione). Implementa qui SOLO Bollinger e RSI, non presenti in
engine.py (V6 non li usa).

Capitale/rischio: lasciati come PARAMETRO esterno (passati al motore
al momento del run, non fissati qui) — la decisione se condividere il
capitale con V6 o usare un'allocazione separata resta aperta e si
decide fuori da questo modulo.

Output: stesso schema di eng.generate_signals() — colonna 'signal'
('long'/'short'/None) + atr/adx, consumabile da
BacktestEngineFloatingKillSwitch (o qualunque sua sottoclasse) SENZA
NESSUNA MODIFICA al motore.

STATO: prima implementazione, non ancora testata. Vedi
mean_reversion_feasibility_test.py per il primo controllo.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

import engine as eng

ADX_THRESHOLD = 20.0
BB_PERIOD = 20
BB_STD = 2.0
RSI_PERIOD = 14
RSI_OVERSOLD = 30.0
RSI_OVERBOUGHT = 70.0


def _rsi_wilder(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """RSI con smoothing alla Wilder (stessa famiglia di formula già
    usata per ATR/ADX in engine.py, per coerenza metodologica)."""
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _bollinger_bands(df: pd.DataFrame, period: int = 20, n_std: float = 2.0):
    mid = df["close"].rolling(period).mean()
    std = df["close"].rolling(period).std()
    upper = mid + n_std * std
    lower = mid - n_std * std
    return upper, mid, lower


def generate_mean_reversion_signals(df: pd.DataFrame, inst: eng.InstrumentConfig,
                                     mode: str = "bollinger",
                                     p: eng.ChartaParams = eng.PARAMS) -> pd.DataFrame:
    """df: colonne timestamp, open, high, low, close. mode: 'bollinger'
    o 'rsi'. Ritorna df con colonne atr, adx, signal — stesso schema
    consumabile dal motore esistente."""
    if mode not in ("bollinger", "rsi"):
        raise ValueError(f"mode deve essere 'bollinger' o 'rsi', ricevuto '{mode}'")

    out = df.copy().reset_index(drop=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)

    # riuso diretto delle formule Wilder già validate in engine.py
    out["atr"] = eng.atr_wilder(out, p.atr_period)
    out["adx"] = eng.adx_wilder(out, p.adx_period)

    out["signal"] = None

    if mode == "bollinger":
        upper, mid, lower = _bollinger_bands(out, BB_PERIOD, BB_STD)
        regime_ok = out["adx"] < ADX_THRESHOLD
        long_trigger = (out["close"] < lower) & regime_ok
        short_trigger = (out["close"] > upper) & regime_ok
        out.loc[long_trigger, "signal"] = "long"
        out.loc[short_trigger, "signal"] = "short"

    else:  # rsi
        rsi = _rsi_wilder(out, RSI_PERIOD)
        regime_ok = out["adx"] < ADX_THRESHOLD
        long_trigger = (rsi < RSI_OVERSOLD) & regime_ok
        short_trigger = (rsi > RSI_OVERBOUGHT) & regime_ok
        out.loc[long_trigger, "signal"] = "long"
        out.loc[short_trigger, "signal"] = "short"
        out["rsi"] = rsi  # utile per ispezione/debug, non richiesto dal motore

    return out


if __name__ == "__main__":
    import sys
    print("Questo file va importato come modulo. Vedi mean_reversion_feasibility_test.py per il primo test.")
    sys.exit(0)
