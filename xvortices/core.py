# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union, overload

import numpy as np
import xarray as xr

from . import _core

CenterValue = Union[float, int, Sequence[float], np.ndarray, xr.DataArray]
FieldInput = Union[xr.DataArray, xr.Dataset, Sequence[xr.DataArray]]
FieldOutput = Union[xr.DataArray, List[xr.DataArray]]


def _as_center_dataarray(value: CenterValue, name: str) -> xr.DataArray:
    if isinstance(value, xr.DataArray):
        return value.astype(np.float64)

    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        return xr.DataArray(arr)

    return xr.DataArray(arr, dims=(name,), coords={name: np.arange(arr.size)})


def _drop_interp_coords(obj: xr.DataArray, lonname: str, latname: str) -> xr.DataArray:
    return obj.drop_vars([latname, lonname], errors="ignore")


def _regular_spacing(coord: xr.DataArray) -> Optional[Tuple[float, float]]:
    values = np.asarray(coord.values, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        return None
    diffs = np.diff(values)
    step = diffs[0]
    if step == 0 or not np.allclose(diffs, step, rtol=1e-10, atol=1e-12):
        return None
    return values[0], step


def _interp_regular_dataarray(
    da: xr.DataArray,
    lons: xr.DataArray,
    lats: xr.DataArray,
    lonname: str,
    latname: str,
) -> Optional[xr.DataArray]:
    if lonname not in da.coords or latname not in da.coords:
        return None
    if da.coords[lonname].dims != (lonname,) or da.coords[latname].dims != (latname,):
        return None
    lon_spacing = _regular_spacing(da.coords[lonname])
    lat_spacing = _regular_spacing(da.coords[latname])
    if lon_spacing is None or lat_spacing is None:
        return None
    if lons.dims != lats.dims or len(lons.dims) < 2:
        return None
    if lons.dims[-2:] != ("radi", "azim"):
        return None

    center_dims = lons.dims[:-2]
    other_dims = tuple(dim for dim in da.dims if dim not in (latname, lonname))
    if not set(center_dims).issubset(other_dims):
        return None

    spatial = da.transpose(*other_dims, latname, lonname)
    other_shape = spatial.shape[:-2]
    nouter = int(np.prod(other_shape, dtype=np.int64)) if other_shape else 1
    nlat = spatial.sizes[latname]
    nlon = spatial.sizes[lonname]
    data = np.ascontiguousarray(spatial.values.reshape(nouter, nlat, nlon), dtype=np.float64)

    center_shape = tuple(lons.sizes[dim] for dim in center_dims)
    ncenter = int(np.prod(center_shape, dtype=np.int64)) if center_shape else 1
    ntarget = lons.sizes["radi"] * lons.sizes["azim"]
    target_lon = np.ascontiguousarray(lons.values.reshape(ncenter, ntarget), dtype=np.float64)
    target_lat = np.ascontiguousarray(lats.values.reshape(ncenter, ntarget), dtype=np.float64)

    if center_dims:
        axis_by_dim = {dim: i for i, dim in enumerate(other_dims)}
        center_axes = [axis_by_dim[dim] for dim in center_dims]
        center_map = np.empty(nouter, dtype=np.int32)
        for flat_index, multi_index in enumerate(np.ndindex(other_shape)):
            center_index = tuple(multi_index[axis] for axis in center_axes)
            center_map[flat_index] = np.ravel_multi_index(center_index, center_shape)
    else:
        center_map = np.zeros(nouter, dtype=np.int32)

    out = _core.interp_regular(
        data,
        float(lon_spacing[0]),
        float(lon_spacing[1]),
        float(lat_spacing[0]),
        float(lat_spacing[1]),
        center_map,
        target_lon,
        target_lat,
    )

    out_dims = other_dims + ("radi", "azim")
    out_shape = other_shape + (lons.sizes["radi"], lons.sizes["azim"])
    out_coords = {
        dim: da.coords[dim]
        for dim in other_dims
        if dim in da.coords
    }
    out_coords["radi"] = lons.coords["radi"]
    out_coords["azim"] = lons.coords["azim"]
    return xr.DataArray(out.reshape(out_shape), dims=out_dims, coords=out_coords, name=da.name, attrs=da.attrs)


def _interp_dataarray(
    da: xr.DataArray,
    lons: xr.DataArray,
    lats: xr.DataArray,
    lonname: str,
    latname: str,
) -> xr.DataArray:
    out = _interp_regular_dataarray(da, lons, lats, lonname, latname)
    if out is not None:
        return out
    return _drop_interp_coords(da.interp(coords={lonname: lons, latname: lats}), lonname, latname)


def _compute_cylind_coords(
    olon: CenterValue,
    olat: CenterValue,
    azim: xr.DataArray,
    radi: xr.DataArray,
) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    olon_da, olat_da = xr.broadcast(
        _as_center_dataarray(olon, "center"),
        _as_center_dataarray(olat, "center"),
    )
    center_dims = olon_da.dims
    center_shape = olon_da.shape
    center_coords = {
        dim: olon_da.coords[dim]
        for dim in center_dims
        if dim in olon_da.coords
    }

    lons, lats, etas = _core.cylind_coords(
        np.ascontiguousarray(olon_da.values.reshape(-1), dtype=np.float64),
        np.ascontiguousarray(olat_da.values.reshape(-1), dtype=np.float64),
        np.ascontiguousarray(azim.values, dtype=np.float64),
        np.ascontiguousarray(radi.values, dtype=np.float64),
    )

    out_shape = center_shape + (radi.size, azim.size)
    out_dims = center_dims + ("radi", "azim")
    out_coords = dict(center_coords)
    out_coords["radi"] = radi.values
    out_coords["azim"] = azim.values

    if not center_shape:
        lons = lons.reshape(radi.size, azim.size)
        lats = lats.reshape(radi.size, azim.size)
        etas = etas.reshape(radi.size, azim.size)
        out_shape = (radi.size, azim.size)
        out_dims = ("radi", "azim")
        out_coords = {"radi": radi.values, "azim": azim.values}
    else:
        lons = lons.reshape(out_shape)
        lats = lats.reshape(out_shape)
        etas = etas.reshape(out_shape)

    return (
        xr.DataArray(lons, dims=out_dims, coords=out_coords, name="lon"),
        xr.DataArray(lats, dims=out_dims, coords=out_coords, name="lat"),
        xr.DataArray(etas, dims=out_dims, coords=out_coords, name="eta"),
    )


@overload
def load_cylind(
    ds: xr.DataArray,
    olon: CenterValue,
    olat: CenterValue,
    azimNum: int = 36,
    radiNum: int = 11,
    radMax: float = 10,
    lonname: str = "lon",
    latname: str = "lat",
) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    ...


@overload
def load_cylind(
    ds: Union[xr.Dataset, Sequence[xr.DataArray]],
    olon: CenterValue,
    olat: CenterValue,
    azimNum: int = 36,
    radiNum: int = 11,
    radMax: float = 10,
    lonname: str = "lon",
    latname: str = "lat",
) -> Tuple[List[xr.DataArray], xr.DataArray, xr.DataArray, xr.DataArray]:
    ...


def load_cylind(
    ds: FieldInput,
    olon: CenterValue,
    olat: CenterValue,
    azimNum: int = 36,
    radiNum: int = 11,
    radMax: float = 10,
    lonname: str = "lon",
    latname: str = "lat",
) -> Tuple[FieldOutput, xr.DataArray, xr.DataArray, xr.DataArray]:
    """Load binary data

    Load scalar data from a lat/lon grid to a cylindrical grid translating
    with a vortex.

    Parameters
    ----------
    ds: xarray.DataArray or a xarray.Dataset or (list of) DataArray
        A given lat/lon grid variable or dataset to be interpolated
    olon: (list of) float, numpy.array, or xarray.DataArray
        Central longitude of the cylindrical coordinate, in degree
    olat: (list of) float, numpy.array, or xarray.DataArray
        Central latitude of the cylindrical coordinate, in degree
    azimNum: int
        Number of azimuthal grid points
    radiNum: int
        Number of radial grid points
    radMax: float
        Maximum radius in degree
    lonname: str
        Name of longitude in ds
    latname: str
        Name of latitude in ds

    Return
    ----------
    vs_interp: xarray.DataArray or list of xarray.DataArray
        Interpolated variables
    lons: xarray.DataArray
        Longitudes for cylindrical coordinates (degree)
    lats: xarray.DataArray
        latitudes for cylindrical coordinates (degree)
    etas_r: xarray.DataArray
        Local angle between radial direction and local north (radian)
    """
    azim_values = np.linspace(0, 360 - 360 / azimNum, azimNum)
    radi_values = np.linspace(0, radMax, radiNum)
    azim = xr.DataArray(azim_values, dims="azim", coords={"azim": azim_values})
    radi = xr.DataArray(radi_values, dims="radi", coords={"radi": radi_values})

    lons, lats, etas_r = _compute_cylind_coords(olon, olat, azim, radi)

    if isinstance(ds, (list, tuple)):
        vs_interp = [
            _interp_dataarray(v, lons, lats, lonname, latname)
            for v in ds
        ]
    elif isinstance(ds, xr.Dataset):
        vs_interp = [
            _interp_dataarray(ds[v], lons, lats, lonname, latname)
            for v in ds.data_vars
        ]
    else:
        vs_interp = _interp_dataarray(ds, lons, lats, lonname, latname)

    return vs_interp, lons, lats, etas_r


def project_to_cylind(
    u: xr.DataArray,
    v: xr.DataArray,
    etas: xr.DataArray,
) -> Tuple[xr.DataArray, xr.DataArray]:
    """Re-project a vector

    Re-project zonal/meridional (u/v) velocity components onto
    azimuthal/radial (uaz/vra) components.

    Parameters
    ----------
    u: xarray.DataArray
        Zonal velocity component
    v: xarray.DataArray
        Meridional velocity component
    etas: xarray.DataArray
        Local angle between radial direction and local north
        
    Return
    ----------
    uaz: xarray.DataArray
        Azimuthal component of velocity
    vra: xarray.DataArray
        radial component of velocity
    """
    u_b, v_b, etas_b = xr.broadcast(u, v, etas)
    uaz, vra = _core.project(
        np.ascontiguousarray(u_b.values, dtype=np.float64),
        np.ascontiguousarray(v_b.values, dtype=np.float64),
        np.ascontiguousarray(etas_b.values, dtype=np.float64),
    )
    uaz = xr.DataArray(uaz, dims=u_b.dims, coords=u_b.coords, attrs=u.attrs)
    vra = xr.DataArray(vra, dims=u_b.dims, coords=u_b.coords, attrs=v.attrs)

    return uaz.rename('ut'), vra.rename('vr')


def storm_relative(
    uc: xr.DataArray,
    vc: xr.DataArray,
    uaz: xr.DataArray,
    vra: xr.DataArray,
) -> Tuple[xr.DataArray, xr.DataArray]:
    """Removing storm motion

    Calculate storm-relative velocity given the translating
    velocity of the center.

    Parameters
    ----------
    uc: xarray.DataArray
        Zonal velocity of the center.
    vc: xarray.DataArray
        Meridional velocity of the center.
    uaz: xarray.DataArray
        Azimuth velocity component
    vra: xarray.DataArray
        radial velocity component
        
    Return
    ----------
    uaz_rel: xarray.DataArray
        Azimuthal component of storm-relative velocity
    vra_rel: xarray.DataArray
        radial component of storm-relative velocity
    """
    if "azim" not in uaz.coords:
        raise ValueError("uaz must include an 'azim' coordinate")

    azim = xr.DataArray(uaz["azim"].values, dims=("azim",), coords={"azim": uaz["azim"].values})
    uc_b, vc_b, uaz_b, vra_b, azim_b = xr.broadcast(uc, vc, uaz, vra, azim)
    uaz_rel, vra_rel = _core.storm_relative(
        np.ascontiguousarray(uc_b.values, dtype=np.float64),
        np.ascontiguousarray(vc_b.values, dtype=np.float64),
        np.ascontiguousarray(azim_b.values, dtype=np.float64),
        np.ascontiguousarray(uaz_b.values, dtype=np.float64),
        np.ascontiguousarray(vra_b.values, dtype=np.float64),
    )
    uaz_rel = xr.DataArray(uaz_rel, dims=uaz_b.dims, coords=uaz_b.coords, attrs=uaz.attrs)
    vra_rel = xr.DataArray(vra_rel, dims=uaz_b.dims, coords=uaz_b.coords, attrs=vra.attrs)

    return uaz_rel, vra_rel
