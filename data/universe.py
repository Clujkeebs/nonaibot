"""
Dynamic asset universe — static seed lists + runtime filtering.
Symbols are filtered for tradability before being handed to strategies.
"""
from __future__ import annotations

from typing import Dict, List

import config


class Universe:
    """Categorised, deduplicated, filterable asset universe."""

    THEMES: Dict[str, List[str]] = {
        "ai_tech":       config.AI_TECH,
        "growth_etf":    config.GROWTH_ETFS,
        "crypto_equity": config.CRYPTO_EQUITIES,
        "core_macro":    config.CORE_MACRO,
    }

    CRYPTO: List[str] = config.CRYPTO_SYMBOLS

    # Priority multipliers used by the portfolio allocator
    THEME_PRIORITY: Dict[str, float] = {
        "ai_tech":       1.5,
        "crypto_equity": 1.3,   # bumped — crypto-equities track best-performing asset class
        "growth_etf":    1.1,
        "core_macro":    0.8,
    }

    def __init__(self) -> None:
        # Build deduped master list
        seen: set[str] = set()
        self._equities: List[str] = []
        for symbols in self.THEMES.values():
            for s in symbols:
                if s not in seen:
                    seen.add(s)
                    self._equities.append(s)

        # Precompute crypto set for O(1) is_crypto lookups
        self._crypto_set: set[str] = set()
        for s in self.CRYPTO:
            self._crypto_set.add(s)
            self._crypto_set.add(s.replace("/", ""))  # also add no-slash version

    # ── Public accessors ──────────────────────────────────────────────────────

    @property
    def equities(self) -> List[str]:
        return list(self._equities)

    @property
    def crypto(self) -> List[str]:
        return list(self.CRYPTO)

    def symbols_for_theme(self, theme: str) -> List[str]:
        return list(self.THEMES.get(theme, []))

    def theme_for_symbol(self, symbol: str) -> str:
        clean = symbol.replace("/", "")
        return config.SECTOR_MAP.get(symbol, config.SECTOR_MAP.get(clean, "unknown"))

    def priority_for_symbol(self, symbol: str) -> float:
        return self.THEME_PRIORITY.get(self.theme_for_symbol(symbol), 1.0)

    def all_symbols(self) -> List[str]:
        """Equities + crypto (for unified loops)."""
        return self._equities + self.CRYPTO

    def is_crypto(self, symbol: str) -> bool:
        """O(1) crypto check using precomputed set."""
        return symbol in self._crypto_set
