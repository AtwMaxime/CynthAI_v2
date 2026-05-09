try:
    from .simulator import PyBattle
except ImportError:
    from simulator import PyBattle  # development editable install

__all__ = ["PyBattle"]
