from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import PropertyMock

import pytest

from voptrade.commands.optimize_commands import setup_optimize_configuration
from voptrade.configuration.timerange import TimeRange
from voptrade.data import history
from voptrade.data.dataprovider import DataProvider
from voptrade.enums import RunMode
from voptrade.exceptions import OperationalException
from voptrade.vopai.data_kitchen import VopaiDataKitchen
from voptrade.optimize.backtesting import Backtesting
from tests.conftest import (
    CURRENT_TEST_STRATEGY,
    get_args,
    get_patched_exchange,
    log_has_re,
    patch_exchange,
    patched_configuration_load_config_file,
)
from tests.vopai.conftest import get_patched_vopai_strategy


def test_vopai_backtest_start_backtest_list(vopai_conf, mocker, testdatadir, caplog):
    patch_exchange(mocker)

    now = datetime.now(UTC)
    mocker.patch(
        "voptrade.plugins.pairlistmanager.PairListManager.whitelist",
        PropertyMock(return_value=["HULUMULU/USDT", "XRP/USDT"]),
    )
    mocker.patch("voptrade.optimize.backtesting.history.load_data")
    mocker.patch("voptrade.optimize.backtesting.history.get_timerange", return_value=(now, now))

    patched_configuration_load_config_file(mocker, vopai_conf)

    args = [
        "backtesting",
        "--config",
        "config.json",
        "--datadir",
        str(testdatadir),
        "--strategy-path",
        str(Path(__file__).parents[1] / "strategy/strats"),
        "--timeframe",
        "1m",
        "--strategy-list",
        CURRENT_TEST_STRATEGY,
    ]
    args = get_args(args)
    bt_config = setup_optimize_configuration(args, RunMode.BACKTEST)
    Backtesting(bt_config)
    assert log_has_re(
        "Using --strategy-list with VopAI REQUIRES all strategies to have identical", caplog
    )
    Backtesting.cleanup()


@pytest.mark.parametrize(
    "timeframe, expected_startup_candle_count",
    [
        ("5m", 876),
        ("15m", 492),
        ("1d", 302),
    ],
)
def test_vopai_backtest_load_data(
    vopai_conf, mocker, caplog, timeframe, expected_startup_candle_count
):
    patch_exchange(mocker)

    now = datetime.now(UTC)
    mocker.patch(
        "voptrade.plugins.pairlistmanager.PairListManager.whitelist",
        PropertyMock(return_value=["HULUMULU/USDT", "XRP/USDT"]),
    )
    mocker.patch("voptrade.optimize.backtesting.history.load_data")
    mocker.patch("voptrade.optimize.backtesting.history.get_timerange", return_value=(now, now))
    vopai_conf["timeframe"] = timeframe
    vopai_conf.get("vopai", {}).get("feature_parameters", {}).update({"include_timeframes": []})
    backtesting = Backtesting(deepcopy(vopai_conf))
    backtesting.load_bt_data()

    assert log_has_re(
        f"Increasing startup_candle_count for vopai on {timeframe} "
        f"to {expected_startup_candle_count}",
        caplog,
    )
    assert history.load_data.call_args[1]["startup_candles"] == expected_startup_candle_count

    Backtesting.cleanup()


def test_vopai_backtest_live_models_model_not_found(vopai_conf, mocker, testdatadir, caplog):
    patch_exchange(mocker)

    now = datetime.now(UTC)
    mocker.patch(
        "voptrade.plugins.pairlistmanager.PairListManager.whitelist",
        PropertyMock(return_value=["HULUMULU/USDT", "XRP/USDT"]),
    )
    mocker.patch("voptrade.optimize.backtesting.history.load_data")
    mocker.patch("voptrade.optimize.backtesting.history.get_timerange", return_value=(now, now))
    vopai_conf["timerange"] = ""
    vopai_conf.get("vopai", {}).update({"backtest_using_historic_predictions": False})

    patched_configuration_load_config_file(mocker, vopai_conf)

    args = [
        "backtesting",
        "--config",
        "config.json",
        "--datadir",
        str(testdatadir),
        "--strategy-path",
        str(Path(__file__).parents[1] / "strategy/strats"),
        "--timeframe",
        "5m",
        "--vopai-backtest-live-models",
    ]
    args = get_args(args)
    bt_config = setup_optimize_configuration(args, RunMode.BACKTEST)

    with pytest.raises(
        OperationalException, match=r".* Historic predictions data is required to run backtest .*"
    ):
        Backtesting(bt_config)

    Backtesting.cleanup()


def test_vopai_backtest_consistent_timerange(mocker, vopai_conf):
    vopai_conf["runmode"] = "backtest"
    mocker.patch(
        "voptrade.plugins.pairlistmanager.PairListManager.whitelist",
        PropertyMock(return_value=["XRP/USDT:USDT"]),
    )

    gbs = mocker.patch("voptrade.optimize.backtesting.generate_backtest_stats")

    vopai_conf["trading_mode"] = "futures"
    vopai_conf.get("exchange", {}).update({"pair_whitelist": ["XRP/USDT:USDT"]})
    vopai_conf.get("vopai", {}).get("feature_parameters", {}).update(
        {"include_timeframes": ["5m", "1h"], "include_corr_pairlist": []}
    )
    vopai_conf["timerange"] = "20211120-20211121"

    strategy = get_patched_vopai_strategy(mocker, vopai_conf)
    exchange = get_patched_exchange(mocker, vopai_conf)

    strategy.dp = DataProvider(vopai_conf, exchange)
    strategy.vopai_info = vopai_conf.get("vopai", {})
    vopai = strategy.vopai
    vopai.dk = VopaiDataKitchen(vopai_conf)

    timerange = TimeRange.parse_timerange("20211115-20211122")
    vopai.dd.load_all_pair_histories(timerange, vopai.dk)

    backtesting = Backtesting(deepcopy(vopai_conf))
    backtesting.start()

    assert gbs.call_args[1]["min_date"] == datetime(2021, 11, 20, 0, 0, tzinfo=UTC)
    assert gbs.call_args[1]["max_date"] == datetime(2021, 11, 21, 0, 0, tzinfo=UTC)
    Backtesting.cleanup()
