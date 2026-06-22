from pathlib import Path
import sys

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xvortices import load_cylind, project_to_cylind, storm_relative


def _decode(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore").strip()
    return str(value).strip()


def main():
    root = Path(__file__).resolve().parents[1]
    ibtracs_path = Path(r"L:\TC_Energy\IBTrACS.ALL.v04r01.nc")
    grid_path = root / "Data" / "Haima2004.nc"

    tracks = xr.open_dataset(ibtracs_path)
    names = np.array([_decode(v) for v in tracks["name"].values])
    seasons = tracks["season"].values
    candidates = np.where((np.char.upper(names) == "HAIMA") & (seasons == 2004))[0]
    if candidates.size == 0:
        raise RuntimeError("Could not find 2004 HAIMA in IBTrACS")

    storm_index = int(candidates[0])
    track = tracks.isel(storm=storm_index)
    valid = np.isfinite(track["lat"].values) & np.isfinite(track["lon"].values)
    track_times = track["time"].values[valid]
    track_lon = xr.DataArray(track["lon"].values[valid], dims=("time",), coords={"time": track_times})
    track_lat = xr.DataArray(track["lat"].values[valid], dims=("time",), coords={"time": track_times})

    ds = xr.open_dataset(grid_path)[["u", "v", "h"]].isel(lev=slice(0, 10))
    olon = track_lon.interp(time=ds.time)
    olat = track_lat.interp(time=ds.time)

    fields, lons, lats, etas = load_cylind(
        ds, olon=olon, olat=olat, azimNum=72, radiNum=31, radMax=6
    )
    u, v, h = fields
    uc = xr.DataArray(np.gradient(olon.values), dims=("time",), coords={"time": ds.time})
    vc = xr.DataArray(np.gradient(olat.values), dims=("time",), coords={"time": ds.time})
    uaz, vra = project_to_cylind(u, v, etas)
    uaz_rel, vra_rel = storm_relative(uc, vc, uaz, vra)

    if not np.isfinite(uaz_rel.values).any() or not np.isfinite(vra_rel.values).any():
        raise RuntimeError("IBTrACS validation produced no finite transformed winds")

    print(f"storm_index={storm_index}")
    print(f"sid={_decode(track['sid'].values)}")
    print(f"name={names[storm_index]}")
    print(f"output_shape={uaz_rel.shape}")
    print(f"finite_fraction={np.isfinite(uaz_rel.values).mean():.3f}")
    print(f"height_mean={float(h.mean()):.6f}")


if __name__ == "__main__":
    main()
