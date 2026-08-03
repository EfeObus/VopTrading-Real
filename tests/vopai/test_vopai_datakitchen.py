import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from voptrade.configuration import TimeRange
from voptrade.data.dataprovider import DataProvider
from voptrade.exceptions import OperationalException
from voptrade.vopai.data_kitchen import VopaiDataKitchen
from tests.conftest import get_patched_exchange, is_mac
from tests.vopai.conftest import (
    get_patched_data_kitchen,
    get_patched_vopai_strategy,
    make_unfiltered_dataframe,
)


@pytest.mark.parametrize(
    "timerange, train_period_days, expected_result",
    [
        ("20220101-20220201", 30, "20211202-20220201"),
        ("20220301-20220401", 15, "20220214-20220401"),
    ],
)
def test_create_fulltimerange(
    timerange, train_period_days, expected_result, vopai_conf, mocker, caplog
):
    dk = get_patched_data_kitchen(mocker, vopai_conf)
    assert dk.create_fulltimerange(timerange, train_period_days) == expected_result
    shutil.rmtree(Path(dk.full_path))


def test_create_fulltimerange_incorrect_backtest_period(mocker, vopai_conf):
    dk = get_patched_data_kitchen(mocker, vopai_conf)
    with pytest.raises(OperationalException, match=r"backtest_period_days must be an integer"):
        dk.create_fulltimerange("20220101-20220201", 0.5)
    with pytest.raises(OperationalException, match=r"backtest_period_days must be positive"):
        dk.create_fulltimerange("20220101-20220201", -1)
    shutil.rmtree(Path(dk.full_path))


@pytest.mark.parametrize(
    "timerange, train_period_days, backtest_period_days, expected_result",
    [
        ("20220101-20220201", 30, 7, 9),
        ("20220101-20220201", 30, 0.5, 120),
        ("20220101-20220201", 10, 1, 80),
    ],
)
def test_split_timerange(
    mocker, vopai_conf, timerange, train_period_days, backtest_period_days, expected_result
):
    vopai_conf.update({"timerange": "20220101-20220401"})
    dk = get_patched_data_kitchen(mocker, vopai_conf)
    tr_list, bt_list = dk.split_timerange(timerange, train_period_days, backtest_period_days)
    assert len(tr_list) == len(bt_list) == expected_result

    with pytest.raises(
        OperationalException, match=r"train_period_days must be an integer greater than 0."
    ):
        dk.split_timerange("20220101-20220201", -1, 0.5)
    shutil.rmtree(Path(dk.full_path))


def test_check_if_model_expired(mocker, vopai_conf):
    dk = get_patched_data_kitchen(mocker, vopai_conf)
    now = datetime.now(tz=UTC).timestamp()
    assert dk.check_if_model_expired(now) is False
    now = (datetime.now(tz=UTC) - timedelta(hours=2)).timestamp()
    assert dk.check_if_model_expired(now) is True
    shutil.rmtree(Path(dk.full_path))


def test_filter_features(mocker, vopai_conf):
    vopai, unfiltered_dataframe = make_unfiltered_dataframe(mocker, vopai_conf)
    vopai.dk.find_features(unfiltered_dataframe)

    filtered_df, _labels = vopai.dk.filter_features(
        unfiltered_dataframe,
        vopai.dk.training_features_list,
        vopai.dk.label_list,
        training_filter=True,
    )

    assert len(filtered_df.columns) == 14


def test_make_train_test_datasets(mocker, vopai_conf):
    vopai, unfiltered_dataframe = make_unfiltered_dataframe(mocker, vopai_conf)
    vopai.dk.find_features(unfiltered_dataframe)

    features_filtered, labels_filtered = vopai.dk.filter_features(
        unfiltered_dataframe,
        vopai.dk.training_features_list,
        vopai.dk.label_list,
        training_filter=True,
    )

    data_dictionary = vopai.dk.make_train_test_datasets(features_filtered, labels_filtered)

    assert data_dictionary
    assert len(data_dictionary) == 7
    assert len(data_dictionary["train_features"].index) == 1916


@pytest.mark.parametrize("model", ["LightGBMRegressor"])
def test_get_full_model_path(mocker, vopai_conf, model):
    vopai_conf.update({"vopaimodel": model})
    vopai_conf.update({"timerange": "20180110-20180130"})
    vopai_conf.update({"strategy": "vopai_test_strat"})

    if is_mac():
        pytest.skip("Mac is confused during this test for unknown reasons")

    strategy = get_patched_vopai_strategy(mocker, vopai_conf)
    exchange = get_patched_exchange(mocker, vopai_conf)
    strategy.dp = DataProvider(vopai_conf, exchange)
    strategy.vopai_info = vopai_conf.get("vopai", {})
    vopai = strategy.vopai
    vopai.live = True
    vopai.dk = VopaiDataKitchen(vopai_conf)
    vopai.dk.live = True
    timerange = TimeRange.parse_timerange("20180110-20180130")
    vopai.dd.load_all_pair_histories(timerange, vopai.dk)

    vopai.dd.pair_dict = MagicMock()

    data_load_timerange = TimeRange.parse_timerange("20180110-20180130")
    new_timerange = TimeRange.parse_timerange("20180120-20180130")
    vopai.dk.set_paths("ADA/BTC", None)
    vopai.extract_data_and_train_model(
        new_timerange, "ADA/BTC", strategy, vopai.dk, data_load_timerange
    )

    model_path = vopai.dk.get_full_models_path(vopai_conf)
    assert model_path.is_dir() is True


def test_get_pair_data_for_features_with_prealoaded_data(mocker, vopai_conf):
    strategy = get_patched_vopai_strategy(mocker, vopai_conf)
    exchange = get_patched_exchange(mocker, vopai_conf)
    strategy.dp = DataProvider(vopai_conf, exchange)
    strategy.vopai_info = vopai_conf.get("vopai", {})
    vopai = strategy.vopai
    vopai.dk = VopaiDataKitchen(vopai_conf)
    timerange = TimeRange.parse_timerange("20180110-20180130")
    vopai.dd.load_all_pair_histories(timerange, vopai.dk)

    _, base_df = vopai.dd.get_base_and_corr_dataframes(timerange, "LTC/BTC", vopai.dk)
    df = vopai.dk.get_pair_data_for_features(
        "LTC/BTC", "5m", strategy, {}, base_dataframes=base_df
    )

    assert df is base_df["5m"]
    assert not df.empty


def test_get_pair_data_for_features_without_preloaded_data(mocker, vopai_conf):
    vopai_conf.update({"timerange": "20180115-20180130"})
    vopai_conf["runmode"] = "backtest"

    strategy = get_patched_vopai_strategy(mocker, vopai_conf)
    exchange = get_patched_exchange(mocker, vopai_conf)
    strategy.dp = DataProvider(vopai_conf, exchange)
    strategy.vopai_info = vopai_conf.get("vopai", {})
    vopai = strategy.vopai
    vopai.dk = VopaiDataKitchen(vopai_conf)
    timerange = TimeRange.parse_timerange("20180110-20180130")
    vopai.dd.load_all_pair_histories(timerange, vopai.dk)

    base_df = {"5m": pd.DataFrame()}
    df = vopai.dk.get_pair_data_for_features(
        "LTC/BTC", "5m", strategy, {}, base_dataframes=base_df
    )

    assert df is not base_df["5m"]
    assert not df.empty
    assert df.iloc[0]["date"].strftime("%Y-%m-%d %H:%M:%S") == "2018-01-11 23:00:00"
    assert df.iloc[-1]["date"].strftime("%Y-%m-%d %H:%M:%S") == "2018-01-30 00:00:00"


def test_populate_features(mocker, vopai_conf):
    strategy = get_patched_vopai_strategy(mocker, vopai_conf)
    exchange = get_patched_exchange(mocker, vopai_conf)
    strategy.dp = DataProvider(vopai_conf, exchange)
    strategy.vopai_info = vopai_conf.get("vopai", {})
    vopai = strategy.vopai
    vopai.dk = VopaiDataKitchen(vopai_conf)
    timerange = TimeRange.parse_timerange("20180115-20180130")
    vopai.dd.load_all_pair_histories(timerange, vopai.dk)

    corr_df, base_df = vopai.dd.get_base_and_corr_dataframes(timerange, "LTC/BTC", vopai.dk)
    mocker.patch.object(strategy, "feature_engineering_expand_all", return_value=base_df["5m"])
    df = vopai.dk.populate_features(
        base_df["5m"], "LTC/BTC", strategy, base_dataframes=base_df, corr_dataframes=corr_df
    )

    strategy.feature_engineering_expand_all.assert_called_once()
    pd.testing.assert_frame_equal(
        base_df["5m"], strategy.feature_engineering_expand_all.call_args[0][0]
    )

    assert df.iloc[0]["date"].strftime("%Y-%m-%d %H:%M:%S") == "2018-01-15 00:00:00"
