import time

import numpy as np
import pytest
import xarray as xr

from xvortices import load_cylind, project_to_cylind, storm_relative
from xvortices import _core


def reference_cylind_coords(olon, olat, azim, radi):
    olon_r = np.deg2rad(olon)[:, None, None]
    olat_r = np.deg2rad(olat)[:, None, None]
    azim_r = np.deg2rad(azim)[None, None, :]
    radi_r = np.deg2rad(radi)[None, :, None]

    lats_r = np.arcsin(
        np.sin(olat_r) * np.cos(radi_r)
        + np.cos(olat_r) * np.sin(radi_r) * np.cos(azim_r)
    )
    dlam_r = np.arcsin(np.sin(radi_r) * np.sin(azim_r)) / np.cos(lats_r)
    lons_r = olon_r - dlam_r
    etas_r = np.arccos(
        np.sin(olat_r) * np.sin(dlam_r) * np.sin(azim_r)
        - np.cos(dlam_r) * np.cos(azim_r)
    )
    etas_r = np.where(azim[None, None, :] < 180, -etas_r + np.pi, etas_r + np.pi)
    return np.rad2deg(lons_r), np.rad2deg(lats_r), etas_r


def test_core_cylind_coords_matches_reference():
    olon = np.array([120.0, 121.5, 125.0])
    olat = np.array([18.0, 20.5, 24.0])
    azim = np.linspace(0, 355, 72)
    radi = np.linspace(0, 6, 31)

    actual = _core.cylind_coords(olon, olat, azim, radi)
    expected = reference_cylind_coords(olon, olat, azim, radi)

    for got, want in zip(actual, expected):
        np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)


def test_load_cylind_preserves_xarray_dims_and_values():
    lat = np.linspace(10, 30, 41)
    lon = np.linspace(100, 140, 81)
    data = xr.DataArray(
        np.add.outer(lat, lon),
        dims=("lat", "lon"),
        coords={"lat": lat, "lon": lon},
        name="z",
    )

    out, lons, lats, etas = load_cylind(
        data, olon=120.0, olat=20.0, azimNum=16, radiNum=9, radMax=4
    )

    assert out.dims == ("radi", "azim")
    assert lons.dims == ("radi", "azim")
    assert lats.dims == ("radi", "azim")
    assert etas.dims == ("radi", "azim")
    np.testing.assert_allclose(out.values, lons.values + lats.values, atol=1e-10)


def test_load_cylind_matches_xarray_with_missing_values():
    lat = np.linspace(0, 4, 5)
    lon = np.linspace(100, 104, 5)
    values = np.add.outer(lat, lon)
    values[2, 2] = np.nan
    data = xr.DataArray(
        values,
        dims=("lat", "lon"),
        coords={"lat": lat, "lon": lon},
        name="z",
    )
    azim_num, radi_num, rad_max = 8, 3, 1
    azim_values = np.linspace(0, 360 - 360 / azim_num, azim_num)
    radi_values = np.linspace(0, rad_max, radi_num)
    azim = xr.DataArray(azim_values, dims="azim", coords={"azim": azim_values})
    radi = xr.DataArray(radi_values, dims="radi", coords={"radi": radi_values})

    olon = 102.0
    olat = 2.0
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
    expected = data.interp(coords={"lon": lons, "lat": lats}).drop_vars(
        ["lat", "lon"], errors="ignore"
    )

    actual = load_cylind(
        data, olon=olon, olat=olat, azimNum=azim_num, radiNum=radi_num, radMax=rad_max
    )[0]

    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10, equal_nan=True)
    assert np.isnan(actual.sel(radi=0.0)).all()


def test_project_to_cylind_matches_reference():
    radi = np.linspace(0, 2, 5)
    azim = np.linspace(0, 315, 8)
    dims = ("radi", "azim")
    coords = {"radi": radi, "azim": azim}
    u = xr.DataArray(np.arange(40.0).reshape(5, 8), dims=dims, coords=coords)
    v = xr.DataArray((np.arange(40.0) + 1).reshape(5, 8), dims=dims, coords=coords)
    etas = xr.DataArray(np.linspace(0, np.pi, 40).reshape(5, 8), dims=dims, coords=coords)

    uaz, vra = project_to_cylind(u, v, etas)

    np.testing.assert_allclose(uaz, -u * np.cos(etas) - v * np.sin(etas))
    np.testing.assert_allclose(vra, -u * np.sin(etas) + v * np.cos(etas))
    assert uaz.name == "ut"
    assert vra.name == "vr"


def test_storm_relative_matches_reference_and_keeps_dim_order():
    radi = np.linspace(0, 2, 5)
    azim = np.linspace(0, 315, 8)
    dims = ("time", "radi", "azim")
    coords = {"time": [0, 1], "radi": radi, "azim": azim}
    uaz = xr.DataArray(np.ones((2, 5, 8)) * 10, dims=dims, coords=coords)
    vra = xr.DataArray(np.ones((2, 5, 8)) * 2, dims=dims, coords=coords)
    uc = xr.DataArray([3.0, 4.0], dims=("time",), coords={"time": [0, 1]})
    vc = xr.DataArray([1.0, 2.0], dims=("time",), coords={"time": [0, 1]})

    uaz_rel, vra_rel = storm_relative(uc, vc, uaz, vra)

    cd = np.arctan2(vc, uc)
    cs = np.hypot(uc, vc)
    ang = cd - np.deg2rad(uaz.azim) - np.pi / 2.0
    expected_uaz = uaz - np.sin(ang) * cs
    expected_vra = vra - np.cos(ang) * cs

    assert uaz_rel.dims == dims
    assert vra_rel.dims == dims
    np.testing.assert_allclose(uaz_rel, expected_uaz)
    np.testing.assert_allclose(vra_rel, expected_vra)


def test_regular_load_cylind_path_is_at_least_10x_faster_than_xarray_reference():
    lat = np.linspace(0, 50, 241)
    lon = np.linspace(80, 160, 361)
    times = np.arange(24)
    lev = np.arange(3)
    rng = np.random.default_rng(0)
    da = xr.DataArray(
        rng.normal(size=(times.size, lev.size, lat.size, lon.size)),
        dims=("time", "lev", "lat", "lon"),
        coords={"time": times, "lev": lev, "lat": lat, "lon": lon},
    )
    olon = xr.DataArray(np.linspace(105, 140, times.size), dims=("time",), coords={"time": times})
    olat = xr.DataArray(np.linspace(10, 35, times.size), dims=("time",), coords={"time": times})
    azim_num, radi_num, rad_max = 72, 31, 6

    def xarray_reference():
        azim = xr.DataArray(
            np.linspace(0, 360 - 360 / azim_num, azim_num),
            dims="azim",
            coords={"azim": np.linspace(0, 360 - 360 / azim_num, azim_num)},
        )
        radi = xr.DataArray(
            np.linspace(0, rad_max, radi_num),
            dims="radi",
            coords={"radi": np.linspace(0, rad_max, radi_num)},
        )
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

    def backend():
        return load_cylind(
            da, olon, olat, azimNum=azim_num, radiNum=radi_num, radMax=rad_max
        )[0]

    np.testing.assert_allclose(backend(), xarray_reference(), rtol=1e-10, atol=1e-10)

    xarray_reference()
    backend()
    start = time.perf_counter()
    xarray_reference()
    reference_time = time.perf_counter() - start
    start = time.perf_counter()
    backend()
    backend_time = time.perf_counter() - start

    assert reference_time / backend_time >= 10.0
