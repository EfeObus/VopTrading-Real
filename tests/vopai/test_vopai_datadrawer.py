import shutil
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from voptrade.configuration import TimeRange
from voptrade.data.dataprovider import DataProvider
from voptrade.exceptions import OperationalException
from voptrade.vopai.data_kitchen import VopaiDataKitchen
from tests.conftest import get_patched_exchange
from tests.vopai.conftest import get_patched_vopai_strategy


def test_update_historic_data(mocker, vopai_conf):
    vopai_conf["runmode"] = "backtest"
    strategy = get_patched_vopai_strategy(mocker, vopai_conf)
    exchange = get_patched_exchange(mocker, vopai_conf)
    strategy.dp = DataProvider(vopai_conf, exchange)
    vopai = strategy.vopai
    vopai.live = True
    vopai.dk = VopaiDataKitchen(vopai_conf)
    vopai.dk.live = True
    timerange = TimeRange.parse_timerange("20180110-20180114")

    vopai.dd.load_all_pair_histories(timerange, vopai.dk)
    historic_candles = len(vopai.dd.historic_data["ADA/BTC"]["5m"])
    dp_candles = len(strategy.dp.get_pair_dataframe("ADA/BTC", "5m"))
    candle_difference = dp_candles - historic_candles
    vopai.dk.pair = "ADA/BTC"
    vopai.dd.update_historic_data(strategy, vopai.dk)

    updated_historic_candles = len(vopai.dd.historic_data["ADA/BTC"]["5m"])

    assert updated_historic_candles - historic_candles == candle_difference
    shutil.rmtree(Path(vopai.dk.full_path))


def test_load_all_pairs_histories(mocker, vopai_conf):
    strategy = get_patched_vopai_strategy(mocker, vopai_conf)
    exchange = get_patched_exchange(mocker, vopai_conf)
    strategy.dp = DataProvider(vopai_conf, exchange)
    vopai = strategy.vopai
    vopai.live = True
    vopai.dk = VopaiDataKitchen(vopai_conf)
    vopai.dk.live = True
    timerange = TimeRange.parse_timerange("20180110-20180114")
    vopai.dd.load_all_pair_histories(timerange, vopai.dk)

    assert len(vopai.dd.historic_data.keys()) == len(
        vopai_conf.get("exchange", {}).get("pair_whitelist")
    )
    assert len(vopai.dd.historic_data["ADA/BTC"]) == len(
        vopai_conf.get("vopai", {}).get("feature_parameters", {}).get("include_timeframes")
    )
    shutil.rmtree(Path(vopai.dk.full_path))


def test_get_base_and_corr_dataframes(mocker, vopai_conf):
    strategy = get_patched_vopai_strategy(mocker, vopai_conf)
    exchange = get_patched_exchange(mocker, vopai_conf)
    strategy.dp = DataProvider(vopai_conf, exchange)
    vopai = strategy.vopai
    vopai.live = True
    vopai.dk = VopaiDataKitchen(vopai_conf)
    vopai.dk.live = True
    timerange = TimeRange.parse_timerange("20180110-20180114")
    vopai.dd.load_all_pair_histories(timerange, vopai.dk)
    sub_timerange = TimeRange.parse_timerange("20180111-20180114")
    corr_df, base_df = vopai.dd.get_base_and_corr_dataframes(sub_timerange, "LTC/BTC", vopai.dk)

    num_tfs = len(
        vopai_conf.get("vopai", {}).get("feature_parameters", {}).get("include_timeframes")
    )

    assert len(base_df.keys()) == num_tfs

    assert len(corr_df.keys()) == len(
        vopai_conf.get("vopai", {}).get("feature_parameters", {}).get("include_corr_pairlist")
    )

    assert len(corr_df["ADA/BTC"].keys()) == num_tfs
    shutil.rmtree(Path(vopai.dk.full_path))


def test_use_strategy_to_populate_indicators(mocker, vopai_conf):
    strategy = get_patched_vopai_strategy(mocker, vopai_conf)
    exchange = get_patched_exchange(mocker, vopai_conf)
    strategy.dp = DataProvider(vopai_conf, exchange)
    strategy.vopai_info = vopai_conf.get("vopai", {})
    vopai = strategy.vopai
    vopai.live = True
    vopai.dk = VopaiDataKitchen(vopai_conf)
    vopai.dk.live = True
    timerange = TimeRange.parse_timerange("20180110-20180114")
    vopai.dd.load_all_pair_histories(timerange, vopai.dk)
    sub_timerange = TimeRange.parse_timerange("20180111-20180114")
    corr_df, base_df = vopai.dd.get_base_and_corr_dataframes(sub_timerange, "LTC/BTC", vopai.dk)

    df = vopai.dk.use_strategy_to_populate_indicators(strategy, corr_df, base_df, "LTC/BTC")

    assert len(df.columns) == 33
    shutil.rmtree(Path(vopai.dk.full_path))


def test_get_timerange_from_live_historic_predictions(mocker, vopai_conf):
    strategy = get_patched_vopai_strategy(mocker, vopai_conf)
    exchange = get_patched_exchange(mocker, vopai_conf)
    strategy.dp = DataProvider(vopai_conf, exchange)
    vopai = strategy.vopai
    vopai.live = False
    vopai.dk = VopaiDataKitchen(vopai_conf)
    vopai.dk.live = False
    timerange = TimeRange.parse_timerange("20180126-20180130")
    vopai.dd.load_all_pair_histories(timerange, vopai.dk)
    sub_timerange = TimeRange.parse_timerange("20180128-20180130")
    _, base_df = vopai.dd.get_base_and_corr_dataframes(sub_timerange, "ADA/BTC", vopai.dk)
    base_df["5m"]["date_pred"] = base_df["5m"]["date"]
    vopai.dd.historic_predictions = {}
    vopai.dd.historic_predictions["ADA/USDT"] = base_df["5m"]
    vopai.dd.save_historic_predictions_to_disk()
    vopai.dd.save_global_metadata_to_disk({"start_dry_live_date": 1516406400})

    timerange = vopai.dd.get_timerange_from_live_historic_predictions()
    assert timerange.startts == 1516406400
    assert timerange.stopts == 1517356500


def test_get_timerange_from_backtesting_live_df_pred_not_found(mocker, vopai_conf):
    strategy = get_patched_vopai_strategy(mocker, vopai_conf)
    exchange = get_patched_exchange(mocker, vopai_conf)
    strategy.dp = DataProvider(vopai_conf, exchange)
    vopai = strategy.vopai
    with pytest.raises(OperationalException, match=r"Historic predictions not found.*"):
        vopai.dd.get_timerange_from_live_historic_predictions()


def test_set_initial_return_values(mocker, vopai_conf):
    """
    Simple test of the set initial return values that ensures
    we are concatenating and ffilling values properly.
    """

    strategy = get_patched_vopai_strategy(mocker, vopai_conf)
    exchange = get_patched_exchange(mocker, vopai_conf)
    strategy.dp = DataProvider(vopai_conf, exchange)
    vopai = strategy.vopai
    vopai.live = False
    vopai.dk = VopaiDataKitchen(vopai_conf)
    # Setup
    pair = "BTC/USD"
    end_x = "2023-08-31"
    start_x_plus_1 = "2023-08-30"
    end_x_plus_5 = "2023-09-03"

    historic_data = {"date_pred": pd.date_range(end=end_x, periods=5), "value": range(1, 6)}
    new_data = {
        "date": pd.date_range(start=start_x_plus_1, end=end_x_plus_5),
        "value": range(6, 11),
    }

    vopai.dd.historic_predictions[pair] = pd.DataFrame(historic_data)

    new_pred_df = pd.DataFrame(new_data)
    dataframe = pd.DataFrame(new_data)

    # Action
    with patch("logging.Logger.warning") as mock_logger_warning:
        vopai.dd.set_initial_return_values(pair, new_pred_df, dataframe)

    # Assertions
    hist_pred_df = vopai.dd.historic_predictions[pair]
    model_return_df = vopai.dd.model_return_values[pair]

    assert hist_pred_df["date_pred"].iloc[-1] == pd.Timestamp(end_x_plus_5)
    assert "date_pred" in hist_pred_df.columns
    assert hist_pred_df.shape[0] == 8

    # compare values in model_return_df with hist_pred_df
    assert (
        model_return_df["value"].values == hist_pred_df.tail(len(dataframe))["value"].values
    ).all()
    assert model_return_df.shape[0] == len(dataframe)

    # Ensure logger error is not called
    mock_logger_warning.assert_not_called()


def test_set_initial_return_values_warning(mocker, vopai_conf):
    """
    Simple test of set_initial_return_values that hits the warning
    associated with leaving a VopAI bot offline so long that the
    exchange candles have no common date with the historic predictions
    """

    strategy = get_patched_vopai_strategy(mocker, vopai_conf)
    exchange = get_patched_exchange(mocker, vopai_conf)
    strategy.dp = DataProvider(vopai_conf, exchange)
    vopai = strategy.vopai
    vopai.live = False
    vopai.dk = VopaiDataKitchen(vopai_conf)
    # Setup
    pair = "BTC/USD"
    end_x = "2023-08-31"
    start_x_plus_1 = "2023-09-01"
    end_x_plus_5 = "2023-09-05"

    historic_data = {"date_pred": pd.date_range(end=end_x, periods=5), "value": range(1, 6)}
    new_data = {
        "date": pd.date_range(start=start_x_plus_1, end=end_x_plus_5),
        "value": range(6, 11),
    }

    vopai.dd.historic_predictions[pair] = pd.DataFrame(historic_data)

    new_pred_df = pd.DataFrame(new_data)
    dataframe = pd.DataFrame(new_data)

    # Action
    with patch("logging.Logger.warning") as mock_logger_warning:
        vopai.dd.set_initial_return_values(pair, new_pred_df, dataframe)

    # Assertions
    hist_pred_df = vopai.dd.historic_predictions[pair]
    model_return_df = vopai.dd.model_return_values[pair]

    assert hist_pred_df["date_pred"].iloc[-1] == pd.Timestamp(end_x_plus_5)
    assert "date_pred" in hist_pred_df.columns
    assert hist_pred_df.shape[0] == 10

    # compare values in model_return_df with hist_pred_df
    assert (
        model_return_df["value"].values == hist_pred_df.tail(len(dataframe))["value"].values
    ).all()
    assert model_return_df.shape[0] == len(dataframe)

    # Ensure logger error is not called
    mock_logger_warning.assert_called()
