"""
Module 4 — Oil Spill Drift Prediction.
Primary: OpenDrift/OpenOil (physics-based Lagrangian particle tracking).
Optional: LSTM as ML comparison exhibit.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path


# ── OpenOil (physics) — primary ────────────────────────────────────────────────

def run_opendrift_simulation(
    spill_lon: float,
    spill_lat: float,
    spill_time: datetime,
    duration_hours: int = 48,
    num_particles: int = 500,
    current_source: str = "copernicus",   # or "opendap" / custom reader
    output_csv: str = "checkpoints/drift_output.csv",
) -> pd.DataFrame:
    """
    Run an OpenOil Lagrangian simulation from a detected spill point.
    Returns DataFrame: lon, lat, hour columns — the predicted drift trajectory.

    Requires: opendrift installed + Copernicus Marine / CMEMS credentials.
    """
    try:
        from opendrift.models.openoil import OpenOil
    except ImportError:
        print("[drift] opendrift not installed — returning synthetic demo drift.")
        return _synthetic_drift(spill_lon, spill_lat, duration_hours)

    o = OpenOil(loglevel=20)

    if current_source == "copernicus":
        try:
            from opendrift.readers import reader_netCDF_CF_generic
            import copernicusmarine as cm
            ds = cm.open_dataset(
                dataset_id="cmems_mod_glo_phy_anfc_0.083deg_PT1H-m",
                variables=["uo", "vo"],
                start_datetime=spill_time.strftime("%Y-%m-%dT%H:%M:%S"),
                end_datetime=(spill_time + timedelta(hours=duration_hours)).strftime("%Y-%m-%dT%H:%M:%S"),
                minimum_longitude=spill_lon - 5,
                maximum_longitude=spill_lon + 5,
                minimum_latitude=spill_lat - 5,
                maximum_latitude=spill_lat + 5,
                minimum_depth=0,
                maximum_depth=1,
            )
            reader = reader_netCDF_CF_generic.Reader(ds)
            o.add_reader(reader)
        except Exception as e:
            print(f"[drift] Copernicus reader failed: {e} — using fallback winds only.")

    o.add_readers_from_list(["https://opendap.oceanbiology.org/erddap/griddap/ucsdHFRagg"])

    o.seed_elements(
        lon=spill_lon, lat=spill_lat,
        number=num_particles,
        radius=500,
        time=spill_time,
        oiltype="CRUDE OIL",
    )

    o.run(
        duration=timedelta(hours=duration_hours),
        time_step=timedelta(hours=1),
        time_step_output=timedelta(hours=1),
    )

    # extract centroid path
    lons, lats = [], []
    for t in range(duration_hours + 1):
        lons.append(float(np.nanmedian(o.history["lon"][:, t])))
        lats.append(float(np.nanmedian(o.history["lat"][:, t])))

    df = pd.DataFrame({"lon": lons, "lat": lats, "hour": range(len(lons))})
    df.to_csv(output_csv, index=False)
    print(f"Drift simulation saved → {output_csv}")
    return df


def _synthetic_drift(lon: float, lat: float, hours: int = 48) -> pd.DataFrame:
    """
    Demo fallback when OpenDrift/Copernicus is unavailable.
    Simple linear drift: ~0.5 kt northeast.
    """
    rows = []
    for h in range(hours + 1):
        rows.append({
            "lon": lon + h * 0.02,
            "lat": lat + h * 0.015,
            "hour": h,
        })
    return pd.DataFrame(rows)


# ── LSTM (comparison exhibit, not load-bearing) ────────────────────────────────

class DriftLSTM(torch.nn.Module if True else object):
    pass


def train_drift_lstm(
    historical_spills: pd.DataFrame,
    save_path: str = "checkpoints/drift_lstm.pt",
):
    """
    Minimal LSTM trained on (currents, wind, spill_position) → next_position.
    Used as ML comparison vs OpenOil physics.
    Expected columns in historical_spills: lon, lat, hour, u_current, v_current, u_wind, v_wind
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("PyTorch not available for LSTM drift model.")
        return None

    class _DriftLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(input_size=6, hidden_size=64, num_layers=2, batch_first=True)
            self.fc   = nn.Linear(64, 2)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])

    features = ["lon", "lat", "hour", "u_current", "v_current", "u_wind", "v_wind"]
    available = [c for c in features if c in historical_spills.columns]
    if len(available) < 4:
        print("[drift LSTM] insufficient columns — skipping.")
        return None

    X = historical_spills[available].values.astype(np.float32)
    # simple sequence prediction: predict t+1 from window of 6 steps
    seq_len = 6
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i+seq_len, :])
        ys.append(X[i+seq_len, :2])
    if not Xs:
        return None

    Xt = torch.tensor(np.array(Xs))
    yt = torch.tensor(np.array(ys))

    model = _DriftLSTM()
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(50):
        model.train()
        opt.zero_grad()
        pred = model(Xt)
        loss = loss_fn(pred, yt)
        loss.backward()
        opt.step()
        if epoch % 10 == 0:
            print(f"  LSTM drift epoch {epoch}: loss={loss.item():.6f}")

    torch.save(model.state_dict(), save_path)
    print(f"LSTM drift model saved → {save_path}")
    return model
