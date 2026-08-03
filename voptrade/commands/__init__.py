# flake8: noqa: F401
"""
Commands module.
Contains all start-commands, subcommands and CLI Interface creation.

Note: Be careful with file-scoped imports in these subfiles.
    as they are parsed on startup, nothing containing optional modules should be loaded.
"""

from voptrade.commands.analyze_commands import start_analysis_entries_exits
from voptrade.commands.arguments import Arguments
from voptrade.commands.build_config_commands import start_new_config, start_show_config
from voptrade.commands.data_commands import (
    start_convert_data,
    start_convert_trades,
    start_download_data,
    start_list_data,
    start_list_trades_data,
)
from voptrade.commands.db_commands import start_convert_db
from voptrade.commands.deploy_commands import (
    start_create_userdir,
    start_install_ui,
    start_new_strategy,
)
from voptrade.commands.hyperopt_commands import start_hyperopt_list, start_hyperopt_show
from voptrade.commands.list_commands import (
    start_list_exchanges,
    start_list_vopAI_models,
    start_list_hyperopt_loss_functions,
    start_list_markets,
    start_list_strategies,
    start_list_timeframes,
    start_show_trades,
)
from voptrade.commands.optimize_commands import (
    start_backtesting,
    start_backtesting_show,
    start_edge,
    start_hyperopt,
    start_lookahead_analysis,
    start_recursive_analysis,
)
from voptrade.commands.pairlist_commands import start_test_pairlist
from voptrade.commands.plot_commands import start_plot_dataframe, start_plot_profit
from voptrade.commands.strategy_utils_commands import start_strategy_update
from voptrade.commands.trade_commands import start_trading
from voptrade.commands.webserver_commands import start_webserver
