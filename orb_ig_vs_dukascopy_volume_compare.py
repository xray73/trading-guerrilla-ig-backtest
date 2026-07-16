"""
orb_ig_vs_dukascopy_volume_compare.py — Confronta il volume su FTSE100
tra IG (dati storici reali del broker) e Dukascopy, su un campione di
~10 giorni di sessione (per restare sotto la quota settimanale IG di
10.000 punti dato). Risponde alla domanda: il volume FTSE100 è
utilizzabile su ALMENO una delle due fonti, per la specifica ORB+VWAP?

Non fa nessun ordine. Solo lettura prezzi storici (consuma quota IG,
si resetta ogni 7 giorni) e lettura Dukascopy (gratuita, nessun limite
noto).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd

import dukascopy_python
from dukascopy_python.instruments import INSTRUMENT_IDX_EUROPE_E_FUTSEE_100

from ig_client import IGSession, load_credentials_from_env

DAYS_BACK = 10
INSTRUMENT = "FTSE100"


def fetch_dukascopy_1min(start: datetime, end: datetime) -> pd.DataFrame:
    df = dukascopy_python.fetch(
        INSTRUMENT_IDX_EUROPE_E_FUTSEE_100, dukascopy_python.INTERVAL_MIN_1,
        dukascopy_python.OFFER_SIDE_BID, start, end,
    ).reset_index()
    ts_col = df.columns[0]
    df = df.rename(columns={ts_col: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def parse_ig_prices(raw: dict) -> pd.DataFrame:
    rows = []
    for p in raw.get("prices", []):
        rows.append({
            "timestamp": pd.to_datetime(p["snapshotTime"], utc=True, format="mixed"),
            "open": p["openPrice"]["bid"],
            "high": p["highPrice"]["bid"],
            "low": p["lowPrice"]["bid"],
            "close": p["closePrice"]["bid"],
            "volume": p.get("lastTradedVolume"),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def summarize_volume(df: pd.DataFrame, label: str):
    if df.empty or "volume" not in df.columns:
        print(f"  {label}: nessun dato/volume disponibile.")
        return
    vol = df["volume"].fillna(0)
    n = len(df)
    n_zero = (vol == 0).sum()
    print(f"  {label}: {n} barre, volume medio={vol.mean():.3f}, mediana={vol.median():.3f}, "
          f"barre a zero={n_zero} ({100*n_zero/n:.1f}%)")


def main():
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=DAYS_BACK)

    print(f"=== Confronto volume {INSTRUMENT} — IG vs Dukascopy, ultimi {DAYS_BACK} giorni ===\n")

    print("--- Dukascopy ---")
    duka_df = fetch_dukascopy_1min(start, end)
    summarize_volume(duka_df, "Dukascopy 1min")

    print("\n--- IG (storico reale, consuma quota settimanale) ---")
    creds = load_credentials_from_env()
    with IGSession(creds) as session:
        raw = session.get_historical_prices(
            INSTRUMENT, resolution="MINUTE",
            start=start.strftime("%Y-%m-%dT%H:%M:%S"),
            end=end.strftime("%Y-%m-%dT%H:%M:%S"),
        )
        allowance = raw.get("allowance", {})
        print(f"  Quota IG: usata {allowance.get('totalAllowance', '?') and (int(allowance.get('totalAllowance', 0)) - int(allowance.get('remainingAllowance', 0)))} "
              f"/ {allowance.get('totalAllowance', '?')} punti questa settimana "
              f"(reset tra {allowance.get('allowanceExpiry', '?')} secondi)")

        ig_df = parse_ig_prices(raw)
        summarize_volume(ig_df, "IG 1min")

    if not duka_df.empty and not ig_df.empty:
        print("\n--- Pattern orario a confronto (volume medio per ora UTC) ---")
        duka_df["hour"] = duka_df["timestamp"].dt.hour
        ig_df["hour"] = ig_df["timestamp"].dt.hour
        duka_hourly = duka_df.groupby("hour")["volume"].mean()
        ig_hourly = ig_df.groupby("hour")["volume"].mean()
        print(f"{'ora':>4} {'Dukascopy':>12} {'IG':>12}")
        for h in sorted(set(duka_hourly.index) | set(ig_hourly.index)):
            d = duka_hourly.get(h, float('nan'))
            i = ig_hourly.get(h, float('nan'))
            print(f"{h:>4} {d:>12.3f} {i:>12.3f}")

    print("\n=== Fine confronto. Nessun ordine inviato. ===")


if __name__ == "__main__":
    main()
