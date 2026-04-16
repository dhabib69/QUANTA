"""
quanta_nike_screener — backward-compatibility shim.
All logic has moved to quanta_thor_screener.py.
"""
from quanta_thor_screener import (
    ThorSignal as NikeSignal,
    ThorScreener as NikeScreener,
    _thor_check as _nike_check,
    _SymbolBuffer,
    COOLDOWN_BARS,
)

__all__ = [
    "NikeSignal", "NikeScreener", "_nike_check", "_SymbolBuffer", "COOLDOWN_BARS",
]
