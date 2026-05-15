import importlib.util
import sys
from pathlib import Path

def _load_native():
    # Priority 1: maturin editable install (site-packages/simulator/simulator.cpXXX-win_amd64.pyd)
    for sp in sys.path:
        pkg_dir = Path(sp) / "simulator"
        for pyd in sorted(pkg_dir.glob("simulator*.pyd")):
            if pyd != Path(__file__).parent / "simulator.pyd":
                spec = importlib.util.spec_from_file_location("simulator", pyd)
                if spec:
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    return mod
    # Priority 2: local simulator.pyd (last-built fallback committed to repo)
    try:
        from . import simulator as _m
        return _m
    except ImportError:
        pass
    raise ImportError(
        "Could not find compiled simulator extension. "
        "Run: cd simulator && maturin develop --release"
    )

_native = _load_native()
PyBattle = _native.PyBattle

__all__ = ["PyBattle"]
