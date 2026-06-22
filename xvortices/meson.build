project(
  'xvortices-backend',
  'c',
  'fortran',
  version: '0.2.0',
  default_options: ['warning_level=2', 'buildtype=release', 'b_lto=true'],
)

py_mod = import('python')
py = py_mod.find_installation(pure: false)
py_dep = py.dependency()
build_py = find_program('build_python', required: false)
if not build_py.found()
  build_py = py
endif

incdir_numpy_cmd = run_command(
  build_py,
  ['-c', '''
import numpy
import os
paths = [
    numpy.get_include(),
    os.path.join(numpy.__path__[0], "core", "include"),
    os.path.join(numpy.__path__[0], "_core", "include"),
]
for path in paths:
    if os.path.exists(os.path.join(path, "numpy", "arrayobject.h")):
        print(path)
        break
else:
    print(numpy.get_include())
'''],
  check: true,
)
incdir_numpy = incdir_numpy_cmd.stdout().strip()

host_system = host_machine.system()
host_cpu = host_machine.cpu_family()
fortran_compiler = meson.get_compiler('fortran')
c_compiler = meson.get_compiler('c')

openmp_dependencies = []
openmp_fortran_args = []
openmp_c_args = []
openmp_link_args = []

if fortran_compiler.get_id() == 'gcc' and c_compiler.get_id() == 'gcc'
  openmp_dep = dependency('openmp', required: false)
  if openmp_dep.found()
    openmp_dependencies += openmp_dep
  endif
  openmp_fortran_args += ['-fopenmp']
  openmp_c_args += ['-fopenmp']
  openmp_link_args += ['-fopenmp']
elif host_system != 'windows'
  openmp_dep = dependency('openmp', required: false)
  if openmp_dep.found()
    openmp_dependencies += openmp_dep
  endif
endif

fortran_args = [
  '-Ofast',
  '-fPIC',
  '-funroll-loops',
  '-finline-functions',
  '-ftree-vectorize',
  '-ffinite-math-only',
  '-fno-math-errno',
  '-ffree-line-length-none',
]
c_args = [
  '-DNPY_NO_DEPRECATED_API=NPY_1_19_API_VERSION',
  '-Ofast',
  '-fno-math-errno',
]

if host_system == 'darwin' and host_cpu == 'aarch64'
  fortran_args += ['-march=armv8-a', '-mtune=apple-m1']
  c_args += ['-march=armv8-a', '-mtune=apple-m1']
else
  fortran_args += ['-march=x86-64', '-mtune=generic']
  c_args += ['-march=x86-64', '-mtune=generic']
endif

fortran_args += openmp_fortran_args
c_args += openmp_c_args

inc_np = include_directories(incdir_numpy)
dependencies = [py_dep] + openmp_dependencies

py.extension_module(
  '_backend',
  ['_backend_module.c', 'src/core.f90'],
  include_directories: inc_np,
  dependencies: dependencies,
  fortran_args: fortran_args,
  c_args: c_args,
  link_args: openmp_link_args,
  install: true,
  install_dir: get_option('python.purelibdir') / 'xvortices',
  build_by_default: true,
)
