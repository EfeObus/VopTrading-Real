import logging

# from voptrade.enums import MarginMode, TradingMode
from voptrade.exchange import Exchange
from voptrade.exchange.exchange_types import FtHas


logger = logging.getLogger(__name__)


class Modetrade(Exchange):
    """
    MOdetrade exchange class. Contains adjustments needed for Voptrade to work
    with this exchange.

    Please note that this exchange is not included in the list of exchanges
    officially supported by the Voptrade development team. So some features
    may still not work as expected.
    """

    _ft_has: FtHas = {
        "always_require_api_keys": True,  # Requires API keys to fetch candles
    }

    # _supported_trading_mode_margin_pairs: list[tuple[TradingMode, MarginMode]] = [
    #     (TradingMode.FUTURES, MarginMode.ISOLATED),
    # ]
