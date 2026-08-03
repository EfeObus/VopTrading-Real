"""Bitvavo exchange subclass."""

import logging

from voptrade.exchange import Exchange
from voptrade.exchange.exchange_types import FtHas


logger = logging.getLogger(__name__)


class Bitvavo(Exchange):
    """Bitvavo exchange class.

    Contains adjustments needed for Voptrade to work with this exchange.

    Please note that this exchange is not included in the list of exchanges
    officially supported by the Voptrade development team. So some features
    may still not work as expected.
    """

    _ft_has: FtHas = {
        "ohlcv_candle_limit": 1440,
    }
