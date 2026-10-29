from pathlib import Path
import re
import os
import shutil

from setuptools import setup
from setuptools.command.sdist import sdist
from pybind11.setup_helpers import Pybind11Extension, build_ext

HERE = Path(__file__).resolve().parent
# A checkout uses the repository headers; a source archive carries the same header closure.
DEFAULT_ROOT = HERE / "vendor" if (HERE / "vendor/FML/Global/Global.cpp").is_file() else HERE.parents[1]
ROOT = Path(os.environ.get("FML_ANALYSIS_HEADER_ROOT", DEFAULT_ROOT)).resolve()


def sources():
    pending = [Path("FML/CountsInCells/SpatialStatistics.h"), Path("FML/Global/Global.cpp")]
    found = set()
    while pending:
        name = pending.pop()
        if name in found:
            continue
        found.add(name)
        for delim, include in re.findall(r'^\s*#\s*include\s*([<"])([^>"\n]+)[>"]', (ROOT / name).read_text(), re.M):
            candidate = Path(include) if include.startswith("FML/") else name.parent / include
            candidate = Path(os.path.normpath(candidate))
            if (include.startswith("FML/") or delim == '"') and (ROOT / candidate).is_file():
                pending.append(candidate)
    return sorted(found)


class NativeBuild(build_ext):
    def build_extensions(self):
        # Register checkout dependencies after manifest generation: external headers are
        # explicitly packaged by SourceArchive, not copied through setuptools' manifest.
        for extension in self.extensions:
            extension.depends = [str(ROOT / name) for name in sources()]
        super().build_extensions()


class SourceArchive(sdist):
    def make_release_tree(self, base_dir, files):
        super().make_release_tree(base_dir, files)
        for name in sources():
            destination = Path(base_dir) / "vendor" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, destination)


setup(
    packages=["fml_analysis"],
    ext_modules=[Pybind11Extension(
        "fml_analysis._native", ["src/bindings.cpp", "src/runtime.cpp"],
        include_dirs=[str(ROOT)],
        define_macros=[("NO_AUTO_FML_SETUP", "1")], cxx_std=17,
    )],
    cmdclass={"build_ext": NativeBuild, "sdist": SourceArchive},
)
