import io
import os
import shutil
import subprocess
import sys
from pathlib import Path
import re

import codecs
from os import path

from setuptools import Extension, setup, find_packages
from setuptools.command.build_ext import build_ext


def _tool_from_active_env(name):
    prefixes = [Path(sys.prefix), Path(sys.base_prefix)]
    candidates = []
    for active in prefixes:
        candidates.extend(
            [
                active / "Library" / "bin" / f"{name}.exe",
                active / "Library" / "mingw-w64" / "bin" / f"{name}.exe",
                active / "bin" / name,
                active / "bin" / f"{name}.exe",
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return shutil.which(name)


def _configure_compilers():
    if os.environ.get("SKIP_FORTRAN") == "1":
        return
    defaults = {
        "FC": _tool_from_active_env("gfortran"),
        "F77": _tool_from_active_env("gfortran"),
        "F90": _tool_from_active_env("gfortran"),
        "CC": _tool_from_active_env("gcc"),
    }
    for key, value in defaults.items():
        if value and not os.environ.get(key):
            os.environ[key] = value


class MesonBuildExt(build_ext):
    """Build the C/Fortran extension through Meson before setuptools staging."""

    @staticmethod
    def _meson_command(*args):
        return [sys.executable, "-m", "mesonbuild.mesonmain", *args]

    @staticmethod
    def _ninja_command(*args):
        ninja = shutil.which("ninja")
        if ninja:
            return [ninja, *args]
        return [sys.executable, "-m", "ninja", *args]

    @staticmethod
    def _write_native_file(module_dir):
        native_file = module_dir / "meson-python-native.ini"
        python_executable = Path(os.path.abspath(sys.executable)).as_posix()
        native_file.write_text(
            "[binaries]\n"
            f"python = '{python_executable}'\n"
            f"build_python = '{python_executable}'\n",
            encoding="utf-8",
        )
        return native_file

    def run(self):
        if os.environ.get("SKIP_FORTRAN") != "1":
            _configure_compilers()
            built = self._build_backend()
            for ext in self.extensions:
                target = Path(self.get_ext_fullpath(ext.name)).resolve()
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(built, target)
        else:
            super().run()

    def _build_backend(self):
        module_dir = Path("xvortices").resolve()
        build_dir = module_dir / "build"
        if build_dir.exists():
            shutil.rmtree(build_dir)
        build_dir.mkdir(exist_ok=True)
        native_file = self._write_native_file(module_dir)

        setup_cmd = self._meson_command(
            "setup",
            str(build_dir),
            str(module_dir),
            "--wipe",
            f"--native-file={native_file}",
        )
        compile_cmd = self._meson_command("compile", "-C", str(build_dir))

        env = os.environ.copy()
        path_parts = []
        for key in ("FC", "CC"):
            if env.get(key):
                path_parts.append(str(Path(env[key]).parent))
        if path_parts:
            env["PATH"] = os.pathsep.join(path_parts + [env.get("PATH", "")])

        subprocess.check_call(setup_cmd, env=env)
        subprocess.check_call(compile_cmd, env=env)

        suffixes = (".pyd", ".so", ".dylib")
        built = [
            path
            for path in build_dir.iterdir()
            if path.name.startswith("_backend") and path.suffix in suffixes
        ]
        if not built:
            built = [
                path
                for path in build_dir.rglob("_backend*")
                if path.suffix in suffixes
            ]
        if not built:
            raise RuntimeError("Meson completed but did not produce xvortices._backend")

        for stale in module_dir.glob("_backend*.pyd"):
            stale.unlink()
        for stale in module_dir.glob("_backend*.so"):
            stale.unlink()
        for stale in module_dir.glob("_backend*.dylib"):
            stale.unlink()

        source_ext = module_dir / built[0].name
        shutil.copy2(built[0], source_ext)
        return source_ext


_configure_compilers()

with io.open("xvortices/__init__.py", "rt", encoding="utf8") as f:
    version = re.search(r'__version__ = "(.*?)"', f.read()).group(1)

here = path.abspath(path.dirname(__file__))

with codecs.open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='xvortices',

    version=version,

    description='viewing vortices in a translating cylindrical coordinate',
    long_description=long_description,
    long_description_content_type='text/markdown',

    url='https://github.com/QianyeSu/xvortices',

    author='Qianye Su',
    author_email='suqianye2000@gmail.com',

    license='MIT',

    classifiers=[
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Programming Language :: Python :: 3.14',
        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
    ],

    keywords='vortex vortices xarray dask numpy',

    packages=find_packages(exclude=['docs', 'tests', "notebooks", "pics"]),

    install_requires=[
        "numpy",
        "xarray",
        "dask",
    ],
    ext_modules=[Extension("xvortices._backend", sources=[])],
    cmdclass={"build_ext": MesonBuildExt},
)
