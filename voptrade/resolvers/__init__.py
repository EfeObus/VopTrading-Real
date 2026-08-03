# flake8: noqa: F401
# isort: off
from voptrade.resolvers.iresolver import IResolver
from voptrade.resolvers.exchange_resolver import ExchangeResolver

# isort: on
# Don't import HyperoptResolver to avoid loading the whole Optimize tree
# from voptrade.resolvers.hyperopt_resolver import HyperOptResolver
from voptrade.resolvers.pairlist_resolver import PairListResolver
from voptrade.resolvers.protection_resolver import ProtectionResolver
from voptrade.resolvers.strategy_resolver import StrategyResolver
