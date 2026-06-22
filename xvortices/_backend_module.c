#define PY_SSIZE_T_CLEAN

#include <limits.h>
#include <Python.h>
#include <numpy/arrayobject.h>

void xvortices_cylind_coords_c(
    void *olon, void *olat, void *azim, void *radi,
    int ncenter, int nazim, int nradi,
    void *lons, void *lats, void *etas);
void xvortices_project_c(int n, void *u, void *v, void *etas, void *uaz, void *vra);
void xvortices_storm_relative_c(
    int n, void *uc, void *vc, void *azim, void *uaz, void *vra,
    void *uaz_rel, void *vra_rel);
void xvortices_interp_regular_c(
    int nouter, int nlat, int nlon, int ncenter, int ntarget,
    void *data, double lon0, double dlon, double lat0, double dlat,
    void *center_map, void *target_lon, void *target_lat, void *out);

static PyArrayObject *as_float64_array(PyObject *obj) {
    return (PyArrayObject *)PyArray_FROM_OTF(
        obj, NPY_FLOAT64, NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS);
}

static PyObject *py_cylind_coords(PyObject *self, PyObject *args) {
    PyObject *olon_obj = NULL;
    PyObject *olat_obj = NULL;
    PyObject *azim_obj = NULL;
    PyObject *radi_obj = NULL;
    PyArrayObject *olon = NULL;
    PyArrayObject *olat = NULL;
    PyArrayObject *azim = NULL;
    PyArrayObject *radi = NULL;
    PyArrayObject *lons = NULL;
    PyArrayObject *lats = NULL;
    PyArrayObject *etas = NULL;
    npy_intp dims[3];
    int ncenter, nazim, nradi;
    (void)self;

    if (!PyArg_ParseTuple(args, "OOOO", &olon_obj, &olat_obj, &azim_obj, &radi_obj)) {
        return NULL;
    }

    olon = as_float64_array(olon_obj);
    olat = as_float64_array(olat_obj);
    azim = as_float64_array(azim_obj);
    radi = as_float64_array(radi_obj);
    if (olon == NULL || olat == NULL || azim == NULL || radi == NULL) {
        goto fail;
    }
    if (PyArray_NDIM(olon) != 1 || PyArray_NDIM(olat) != 1 ||
        PyArray_NDIM(azim) != 1 || PyArray_NDIM(radi) != 1) {
        PyErr_SetString(PyExc_ValueError, "cylind_coords expects 1D float64 inputs");
        goto fail;
    }
    if (PyArray_SIZE(olon) != PyArray_SIZE(olat)) {
        PyErr_SetString(PyExc_ValueError, "olon and olat must have the same size");
        goto fail;
    }
    if (PyArray_SIZE(olon) > INT_MAX || PyArray_SIZE(azim) > INT_MAX || PyArray_SIZE(radi) > INT_MAX) {
        PyErr_SetString(PyExc_OverflowError, "input is too large for the Fortran backend");
        goto fail;
    }

    ncenter = (int)PyArray_SIZE(olon);
    nazim = (int)PyArray_SIZE(azim);
    nradi = (int)PyArray_SIZE(radi);
    dims[0] = ncenter;
    dims[1] = nradi;
    dims[2] = nazim;

    lons = (PyArrayObject *)PyArray_EMPTY(3, dims, NPY_FLOAT64, 0);
    lats = (PyArrayObject *)PyArray_EMPTY(3, dims, NPY_FLOAT64, 0);
    etas = (PyArrayObject *)PyArray_EMPTY(3, dims, NPY_FLOAT64, 0);
    if (lons == NULL || lats == NULL || etas == NULL) {
        goto fail;
    }

    Py_BEGIN_ALLOW_THREADS
    xvortices_cylind_coords_c(
        PyArray_DATA(olon), PyArray_DATA(olat), PyArray_DATA(azim), PyArray_DATA(radi),
        ncenter, nazim, nradi,
        PyArray_DATA(lons), PyArray_DATA(lats), PyArray_DATA(etas));
    Py_END_ALLOW_THREADS

    Py_DECREF(olon);
    Py_DECREF(olat);
    Py_DECREF(azim);
    Py_DECREF(radi);
    return Py_BuildValue("NNN", lons, lats, etas);

fail:
    Py_XDECREF(olon);
    Py_XDECREF(olat);
    Py_XDECREF(azim);
    Py_XDECREF(radi);
    Py_XDECREF(lons);
    Py_XDECREF(lats);
    Py_XDECREF(etas);
    return NULL;
}

static int parse_same_size_arrays(
    PyObject *a_obj, PyObject *b_obj, PyObject *c_obj,
    PyArrayObject **a, PyArrayObject **b, PyArrayObject **c) {
    *a = as_float64_array(a_obj);
    *b = as_float64_array(b_obj);
    *c = as_float64_array(c_obj);
    if (*a == NULL || *b == NULL || *c == NULL) {
        return -1;
    }
    if (PyArray_SIZE(*a) != PyArray_SIZE(*b) || PyArray_SIZE(*a) != PyArray_SIZE(*c)) {
        PyErr_SetString(PyExc_ValueError, "input arrays must have the same size");
        return -1;
    }
    return 0;
}

static PyObject *py_project(PyObject *self, PyObject *args) {
    PyObject *u_obj = NULL;
    PyObject *v_obj = NULL;
    PyObject *etas_obj = NULL;
    PyArrayObject *u = NULL;
    PyArrayObject *v = NULL;
    PyArrayObject *etas = NULL;
    PyArrayObject *uaz = NULL;
    PyArrayObject *vra = NULL;
    int n;
    (void)self;

    if (!PyArg_ParseTuple(args, "OOO", &u_obj, &v_obj, &etas_obj)) {
        return NULL;
    }
    if (parse_same_size_arrays(u_obj, v_obj, etas_obj, &u, &v, &etas) != 0) {
        goto fail;
    }
    if (PyArray_SIZE(u) > INT_MAX) {
        PyErr_SetString(PyExc_OverflowError, "input is too large for the Fortran backend");
        goto fail;
    }
    uaz = (PyArrayObject *)PyArray_EMPTY(PyArray_NDIM(u), PyArray_DIMS(u), NPY_FLOAT64, 0);
    vra = (PyArrayObject *)PyArray_EMPTY(PyArray_NDIM(u), PyArray_DIMS(u), NPY_FLOAT64, 0);
    if (uaz == NULL || vra == NULL) {
        goto fail;
    }
    n = (int)PyArray_SIZE(u);

    Py_BEGIN_ALLOW_THREADS
    xvortices_project_c(n, PyArray_DATA(u), PyArray_DATA(v), PyArray_DATA(etas),
                        PyArray_DATA(uaz), PyArray_DATA(vra));
    Py_END_ALLOW_THREADS

    Py_DECREF(u);
    Py_DECREF(v);
    Py_DECREF(etas);
    return Py_BuildValue("NN", uaz, vra);

fail:
    Py_XDECREF(u);
    Py_XDECREF(v);
    Py_XDECREF(etas);
    Py_XDECREF(uaz);
    Py_XDECREF(vra);
    return NULL;
}

static PyObject *py_storm_relative(PyObject *self, PyObject *args) {
    PyObject *uc_obj = NULL;
    PyObject *vc_obj = NULL;
    PyObject *azim_obj = NULL;
    PyObject *uaz_obj = NULL;
    PyObject *vra_obj = NULL;
    PyArrayObject *uc = NULL;
    PyArrayObject *vc = NULL;
    PyArrayObject *azim = NULL;
    PyArrayObject *uaz = NULL;
    PyArrayObject *vra = NULL;
    PyArrayObject *uaz_rel = NULL;
    PyArrayObject *vra_rel = NULL;
    int n;
    (void)self;

    if (!PyArg_ParseTuple(args, "OOOOO", &uc_obj, &vc_obj, &azim_obj, &uaz_obj, &vra_obj)) {
        return NULL;
    }

    uc = as_float64_array(uc_obj);
    vc = as_float64_array(vc_obj);
    azim = as_float64_array(azim_obj);
    uaz = as_float64_array(uaz_obj);
    vra = as_float64_array(vra_obj);
    if (uc == NULL || vc == NULL || azim == NULL || uaz == NULL || vra == NULL) {
        goto fail;
    }
    if (PyArray_SIZE(uc) != PyArray_SIZE(vc) || PyArray_SIZE(uc) != PyArray_SIZE(azim) ||
        PyArray_SIZE(uc) != PyArray_SIZE(uaz) || PyArray_SIZE(uc) != PyArray_SIZE(vra)) {
        PyErr_SetString(PyExc_ValueError, "input arrays must have the same size");
        goto fail;
    }
    if (PyArray_SIZE(uaz) > INT_MAX) {
        PyErr_SetString(PyExc_OverflowError, "input is too large for the Fortran backend");
        goto fail;
    }

    uaz_rel = (PyArrayObject *)PyArray_EMPTY(PyArray_NDIM(uaz), PyArray_DIMS(uaz), NPY_FLOAT64, 0);
    vra_rel = (PyArrayObject *)PyArray_EMPTY(PyArray_NDIM(uaz), PyArray_DIMS(uaz), NPY_FLOAT64, 0);
    if (uaz_rel == NULL || vra_rel == NULL) {
        goto fail;
    }
    n = (int)PyArray_SIZE(uaz);

    Py_BEGIN_ALLOW_THREADS
    xvortices_storm_relative_c(
        n, PyArray_DATA(uc), PyArray_DATA(vc), PyArray_DATA(azim),
        PyArray_DATA(uaz), PyArray_DATA(vra),
        PyArray_DATA(uaz_rel), PyArray_DATA(vra_rel));
    Py_END_ALLOW_THREADS

    Py_DECREF(uc);
    Py_DECREF(vc);
    Py_DECREF(azim);
    Py_DECREF(uaz);
    Py_DECREF(vra);
    return Py_BuildValue("NN", uaz_rel, vra_rel);

fail:
    Py_XDECREF(uc);
    Py_XDECREF(vc);
    Py_XDECREF(azim);
    Py_XDECREF(uaz);
    Py_XDECREF(vra);
    Py_XDECREF(uaz_rel);
    Py_XDECREF(vra_rel);
    return NULL;
}

static PyObject *py_interp_regular(PyObject *self, PyObject *args) {
    PyObject *data_obj = NULL;
    PyObject *center_map_obj = NULL;
    PyObject *target_lon_obj = NULL;
    PyObject *target_lat_obj = NULL;
    PyArrayObject *data = NULL;
    PyArrayObject *center_map = NULL;
    PyArrayObject *target_lon = NULL;
    PyArrayObject *target_lat = NULL;
    PyArrayObject *out = NULL;
    double lon0, dlon, lat0, dlat;
    int nouter, nlat, nlon, ncenter, ntarget;
    npy_intp dims[2];
    (void)self;

    if (!PyArg_ParseTuple(
            args, "OddddOOO",
            &data_obj, &lon0, &dlon, &lat0, &dlat,
            &center_map_obj, &target_lon_obj, &target_lat_obj)) {
        return NULL;
    }

    data = as_float64_array(data_obj);
    center_map = (PyArrayObject *)PyArray_FROM_OTF(
        center_map_obj, NPY_INT32, NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS);
    target_lon = as_float64_array(target_lon_obj);
    target_lat = as_float64_array(target_lat_obj);
    if (data == NULL || center_map == NULL || target_lon == NULL || target_lat == NULL) {
        goto fail;
    }
    if (PyArray_NDIM(data) != 3 || PyArray_NDIM(center_map) != 1 ||
        PyArray_NDIM(target_lon) != 2 || PyArray_NDIM(target_lat) != 2) {
        PyErr_SetString(PyExc_ValueError, "interp_regular expects data(nouter,nlat,nlon), center_map(nouter), and target arrays(ncenter,ntarget)");
        goto fail;
    }
    if (PyArray_SIZE(center_map) != PyArray_DIM(data, 0) ||
        PyArray_DIM(target_lon, 0) != PyArray_DIM(target_lat, 0) ||
        PyArray_DIM(target_lon, 1) != PyArray_DIM(target_lat, 1)) {
        PyErr_SetString(PyExc_ValueError, "interp_regular input dimensions are inconsistent");
        goto fail;
    }
    if (PyArray_DIM(data, 0) > INT_MAX || PyArray_DIM(data, 1) > INT_MAX ||
        PyArray_DIM(data, 2) > INT_MAX || PyArray_DIM(target_lon, 0) > INT_MAX ||
        PyArray_DIM(target_lon, 1) > INT_MAX) {
        PyErr_SetString(PyExc_OverflowError, "input is too large for the Fortran backend");
        goto fail;
    }

    nouter = (int)PyArray_DIM(data, 0);
    nlat = (int)PyArray_DIM(data, 1);
    nlon = (int)PyArray_DIM(data, 2);
    ncenter = (int)PyArray_DIM(target_lon, 0);
    ntarget = (int)PyArray_DIM(target_lon, 1);
    dims[0] = nouter;
    dims[1] = ntarget;
    out = (PyArrayObject *)PyArray_EMPTY(2, dims, NPY_FLOAT64, 0);
    if (out == NULL) {
        goto fail;
    }

    Py_BEGIN_ALLOW_THREADS
    xvortices_interp_regular_c(
        nouter, nlat, nlon, ncenter, ntarget,
        PyArray_DATA(data), lon0, dlon, lat0, dlat,
        PyArray_DATA(center_map), PyArray_DATA(target_lon), PyArray_DATA(target_lat),
        PyArray_DATA(out));
    Py_END_ALLOW_THREADS

    Py_DECREF(data);
    Py_DECREF(center_map);
    Py_DECREF(target_lon);
    Py_DECREF(target_lat);
    return (PyObject *)out;

fail:
    Py_XDECREF(data);
    Py_XDECREF(center_map);
    Py_XDECREF(target_lon);
    Py_XDECREF(target_lat);
    Py_XDECREF(out);
    return NULL;
}

static PyMethodDef backend_methods[] = {
    {"cylind_coords", py_cylind_coords, METH_VARARGS, "Compute cylindrical sampling longitude, latitude, and eta arrays."},
    {"project", py_project, METH_VARARGS, "Project zonal/meridional vectors to azimuthal/radial vectors."},
    {"storm_relative", py_storm_relative, METH_VARARGS, "Remove storm translation from cylindrical vector components."},
    {"interp_regular", py_interp_regular, METH_VARARGS, "Bilinearly interpolate a regular lat/lon grid to cylindrical targets."},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef backend_module = {
    PyModuleDef_HEAD_INIT,
    "_backend",
    "Fortran backend for xvortices.",
    -1,
    backend_methods,
};

PyMODINIT_FUNC PyInit__backend(void) {
    PyObject *module = PyModule_Create(&backend_module);
    if (module == NULL) {
        return NULL;
    }
    import_array();
    if (PyErr_Occurred()) {
        Py_DECREF(module);
        return NULL;
    }
    return module;
}
