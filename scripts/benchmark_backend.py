import time
import sys
from pathlib import Path
from typing import Union

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xvortices import load_cylind


CenterValue = Union[float, xr.DataArray]


def old_xarray_load_cylind(
    da: xr.DataArray,
    olon: CenterValue,
    olat: CenterValue,
    azim_num: int,
    radi_num: int,
    rad_max: float,
) -> xr.DataArray:
    azim_values = np.linspace(0, 360 - 360 / azim_num, azim_num)
    radi_values = np.linspace(0, rad_max, radi_num)
    azim = xr.DataArray(azim_values, dims="azim", coords={"azim": azim_values})
    radi = xr.DataArray(radi_values, dims="radi", coords={"radi": radi_values})

    olon_r = np.deg2rad(olon)
    olat_r = np.deg2rad(olat)
    azim_r = np.deg2rad(azim)
    radi_r = np.deg2rad(radi)
    lats_r = np.arcsin(
        np.sin(olat_r) * np.cos(radi_r)
        + np.cos(olat_r) * np.sin(radi_r) * np.cos(azim_r)
    )
    dlam_r = np.arcsin(np.sin(radi_r) * np.sin(azim_r)) / np.cos(lats_r)
    lons = np.rad2deg(olon_r - dlam_r)
    lats = np.rad2deg(lats_r)
    return da.interp(coords={"lon": lons, "lat": lats}).drop_vars(
        ["lat", "lon"], errors="ignore"
    )


def main() -> None:
    lat = np.linspace(0, 50, 401)
    lon = np.linspace(80, 160, 641)
    times = np.arange(48)
    lev = np.arange(4)
    rng = np.random.default_rng(0)
    da = xr.DataArray(
        rng.normal(size=(times.size, lev.size, lat.size, lon.size)),
        dims=("time", "lev", "lat", "lon"),
        coords={"time": times, "lev": lev, "lat": lat, "lon": lon},
        name="u",
    )
    olon = xr.DataArray(
        np.linspace(105, 140, times.size), dims=("time",), coords={"time": times}
    )
    olat = xr.DataArray(
        np.linspace(10, 35, times.size), dims=("time",), coords={"time": times}
    )

    azim_num, radi_num, rad_max = 72, 31, 6

    old_xarray_load_cylind(da, olon, olat, azim_num, radi_num, rad_max)
    load_cylind(da, olon, olat, azimNum=azim_num, radiNum=radi_num, radMax=rad_max)

    start = time.perf_counter()
    expected = old_xarray_load_cylind(da, olon, olat, azim_num, radi_num, rad_max)
    old_time = time.perf_counter() - start

    start = time.perf_counter()
    actual = load_cylind(
        da, olon, olat, azimNum=azim_num, radiNum=radi_num, radMax=rad_max
    )[0]
    new_time = time.perf_counter() - start

    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
    print(f"old_xarray_seconds={old_time:.6f}")
    print(f"fortran_backend_seconds={new_time:.6f}")
    print(f"speedup={old_time / new_time:.2f}x")


if __name__ == "__main__":
    main()
