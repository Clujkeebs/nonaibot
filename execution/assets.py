"""
AssetCache — caches Alpaca asset metadata (fractionability, tradability).

WHY: The Assets API response doesn't change often, but checking it per-order
would burn rate-limit tokens. We fetch once on startup and refresh daily.

The key field: fractionable=True means we can send a notional (dollar amount)
order instead of a whole-share qty order. Per Alpaca rules:
  - Fractional/notional equity orders: market orders with time_in_force=DAY only
  - GTC and OCO orders NOT supported for fractional positions
  - Minimum notional: $1
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional, Set

from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass, AssetStatus

from core.broker import BrokerClient

logger = logging.getLogger(__name__)

_REFRESH_INTERVAL = 3600 * 6  # refresh every 6 hours


class AssetCache:
    def __init__(self, broker: BrokerClient) -> None:
        self._broker = broker
        self._assets: Dict[str, Dict[str, Any]] = {}
        self._last_refresh: float = 0.0
        self._fractionable: Set[str] = set()
        self._tradable: Set[str] = set()
        self._refresh()

    def _refresh(self) -> None:
        """Fetch all active, tradable US equity assets from Alpaca."""
        try:
            req = GetAssetsRequest(
                asset_class=AssetClass.US_EQUITY,
                status=AssetStatus.ACTIVE,
            )
            assets = self._broker.get_all_assets()
            self._assets.clear()
            self._fractionable.clear()
            self._tradable.clear()

            for asset in assets:
                sym = asset.symbol
                tradable = bool(getattr(asset, "tradable", False))
                fractionable = bool(getattr(asset, "fractionable", False))
                self._assets[sym] = {
                    "tradable": tradable,
                    "fractionable": fractionable,
                    "exchange": getattr(asset, "exchange", ""),
                }
                if tradable:
                    self._tradable.add(sym)
                if fractionable:
                    self._fractionable.add(sym)

            self._last_refresh = time.time()
            logger.info(
                "AssetCache refreshed: %d assets, %d tradable, %d fractionable",
                len(self._assets), len(self._tradable), len(self._fractionable)
            )
        except Exception as e:
            logger.warning("AssetCache._refresh() failed: %s", e)

    def _maybe_refresh(self) -> None:
        if time.time() - self._last_refresh > _REFRESH_INTERVAL:
            self._refresh()

    def is_fractionable(self, symbol: str) -> bool:
        """Return True if this symbol supports fractional/notional orders."""
        self._maybe_refresh()
        # Crypto is always fractionable (handled separately)
        if "/" in symbol:
            return True
        return symbol in self._fractionable

    def is_tradable(self, symbol: str) -> bool:
        """Return True if this symbol is currently tradable on Alpaca."""
        self._maybe_refresh()
        if "/" in symbol:
            return True  # crypto tradability is checked separately
        return symbol in self._tradable

    def get(self, symbol: str) -> Optional[Dict[str, Any]]:
        self._maybe_refresh()
        return self._assets.get(symbol)
