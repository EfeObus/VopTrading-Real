import sys
from copy import deepcopy
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from voptrade.configuration import TimeRange
from voptrade.data.dataprovider import DataProvider
from voptrade.vopai.data_drawer import VopaiDataDrawer
from voptrade.vopai.data_kitchen import VopaiDataKitchen
from voptrade.resolvers import StrategyResolver
from voptrade.resolvers.vopaimodel_resolver import VopaiModelResolver
from tests.conftest import get_patched_exchange


def is_py12() -> bool:
    return sys.version_info >= (3, 12)


@pytest.fixture(scope="function")
def vopai_conf(default_conf, tmp_path):
    vopaiconf = deepcopy(default_conf)
    vopaiconf.update(
        {
            "datadir": Path(default_conf["datadir"]),
            "runmode": "backtest",
            "strategy": "vopai_test_strat",
            "user_data_dir": tmp_path,
            "strategy-path": "voptrade/tests/strategy/strats",
            "vopaimodel": "LightGBMRegressor",
            "vopaimodel_path": "vopai/prediction_models",
            "timerange": "20180110-20180115",
            "vopai": {
                "enabled": True,
                "purge_old_models": 2,
                "train_period_days": 2,
                "backtest_period_days": 10,
                "live_retrain_hours": 0,
                "expiration_hours": 1,
                "identifier": "unique-id100",
                "live_trained_timestamp": 0,
                "data_kitchen_thread_count": 2,
                "activate_tensorboard": False,
                "feature_parameters": {
                    "include_timeframes": ["5m"],
                    "include_corr_pairlist": ["ADA/BTC"],
                    "label_period_candles": 20,
                    "include_shifted_candles": 1,
                    "DI_threshold": 0.9,
                    "weight_factor": 0.9,
                    "principal_component_analysis": False,
                    "use_SVM_to_remove_outliers": True,
                    "stratify_training_data": 0,
                    "indicator_periods_candles": [10],
                    "shuffle_after_split": False,
                    "buffer_train_data_candles": 0,
                },
                "data_split_parameters": {"test_size": 0.33, "shuffle": False},
                "model_training_parameters": {"n_estimators": 100},
            },
            "config_files": [Path("config_examples", "config_vopai.example.json")],
        }
    )
    vopaiconf["exchange"].update({"pair_whitelist": ["ADA/BTC", "DASH/BTC", "ETH/BTC", "LTC/BTC"]})
    return vopaiconf


def make_rl_config(conf):
    conf.update({"strategy": "vopai_rl_test_strat"})
    conf["vopai"].update(
        {"model_training_parameters": {"learning_rate": 0.00025, "gamma": 0.9, "verbose": 1}}
    )
    conf["vopai"]["rl_config"] = {
        "train_cycles": 1,
        "thread_count": 2,
        "max_trade_duration_candles": 300,
        "model_type": "PPO",
        "policy_type": "MlpPolicy",
        "max_training_drawdown_pct": 0.5,
        "net_arch": [32, 32],
        "model_reward_parameters": {"rr": 1, "profit_aim": 0.02, "win_reward_factor": 2},
        "drop_ohlc_from_features": False,
    }

    return conf


def mock_pytorch_mlp_model_training_parameters() -> dict[str, Any]:
    return {
        "learning_rate": 3e-4,
        "trainer_kwargs": {
            "n_steps": None,
            "batch_size": 64,
            "n_epochs": 1,
        },
        "model_kwargs": {
            "hidden_dim": 32,
            "dropout_percent": 0.2,
            "n_layer": 1,
        },
    }


def get_patched_data_kitchen(mocker, vopaiconf):
    dk = VopaiDataKitchen(vopaiconf)
    return dk


def get_patched_data_drawer(mocker, vopaiconf):
    # dd = mocker.patch('voptrade.vopai.data_drawer', MagicMock())
    dd = VopaiDataDrawer(vopaiconf)
    return dd


def get_patched_vopai_strategy(mocker, vopaiconf):
    strategy = StrategyResolver.load_strategy(vopaiconf)
    strategy.ft_bot_start()

    return strategy


def get_patched_vopaimodel(mocker, vopaiconf):
    vopaimodel = VopaiModelResolver.load_vopaimodel(vopaiconf)

    return vopaimodel


def make_unfiltered_dataframe(mocker, vopai_conf):
    vopai_conf.update({"timerange": "20180110-20180130"})

    strategy = get_patched_vopai_strategy(mocker, vopai_conf)
    exchange = get_patched_exchange(mocker, vopai_conf)
    strategy.dp = DataProvider(vopai_conf, exchange)
    strategy.vopai_info = vopai_conf.get("vopai", {})
    vopai = strategy.vopai
    vopai.live = True
    vopai.dk = VopaiDataKitchen(vopai_conf)
    vopai.dk.live = True
    vopai.dk.pair = "ADA/BTC"
    data_load_timerange = TimeRange.parse_timerange("20180110-20180130")
    vopai.dd.load_all_pair_histories(data_load_timerange, vopai.dk)

    vopai.dd.pair_dict = MagicMock()

    new_timerange = TimeRange.parse_timerange("20180120-20180130")

    corr_dataframes, base_dataframes = vopai.dd.get_base_and_corr_dataframes(
        data_load_timerange, vopai.dk.pair, vopai.dk
    )

    unfiltered_dataframe = vopai.dk.use_strategy_to_populate_indicators(
        strategy, corr_dataframes, base_dataframes, vopai.dk.pair
    )
    for i in range(5):
        unfiltered_dataframe[f"constant_{i}"] = i

    unfiltered_dataframe = vopai.dk.slice_dataframe(new_timerange, unfiltered_dataframe)

    return vopai, unfiltered_dataframe


def make_data_dictionary(mocker, vopai_conf):
    vopai_conf.update({"timerange": "20180110-20180130"})

    strategy = get_patched_vopai_strategy(mocker, vopai_conf)
    exchange = get_patched_exchange(mocker, vopai_conf)
    strategy.dp = DataProvider(vopai_conf, exchange)
    strategy.vopai_info = vopai_conf.get("vopai", {})
    vopai = strategy.vopai
    vopai.live = True
    vopai.dk = VopaiDataKitchen(vopai_conf)
    vopai.dk.live = True
    vopai.dk.pair = "ADA/BTC"
    data_load_timerange = TimeRange.parse_timerange("20180110-20180130")
    vopai.dd.load_all_pair_histories(data_load_timerange, vopai.dk)

    vopai.dd.pair_dict = MagicMock()

    new_timerange = TimeRange.parse_timerange("20180120-20180130")

    corr_dataframes, base_dataframes = vopai.dd.get_base_and_corr_dataframes(
        data_load_timerange, vopai.dk.pair, vopai.dk
    )

    unfiltered_dataframe = vopai.dk.use_strategy_to_populate_indicators(
        strategy, corr_dataframes, base_dataframes, vopai.dk.pair
    )

    unfiltered_dataframe = vopai.dk.slice_dataframe(new_timerange, unfiltered_dataframe)

    vopai.dk.find_features(unfiltered_dataframe)

    features_filtered, labels_filtered = vopai.dk.filter_features(
        unfiltered_dataframe,
        vopai.dk.training_features_list,
        vopai.dk.label_list,
        training_filter=True,
    )

    data_dictionary = vopai.dk.make_train_test_datasets(features_filtered, labels_filtered)

    data_dictionary = vopai.dk.normalize_data(data_dictionary)

    return vopai


def get_vopai_live_analyzed_dataframe(mocker, vopaiconf):
    strategy = get_patched_vopai_strategy(mocker, vopaiconf)
    exchange = get_patched_exchange(mocker, vopaiconf)
    strategy.dp = DataProvider(vopaiconf, exchange)
    vopai = strategy.vopai
    vopai.live = True
    vopai.dk = VopaiDataKitchen(vopaiconf, vopai.dd)
    timerange = TimeRange.parse_timerange("20180110-20180114")
    vopai.dk.load_all_pair_histories(timerange)

    strategy.analyze_pair("ADA/BTC", "5m")
    return strategy.dp.get_analyzed_dataframe("ADA/BTC", "5m")


def get_vopai_analyzed_dataframe(mocker, vopaiconf):
    strategy = get_patched_vopai_strategy(mocker, vopaiconf)
    exchange = get_patched_exchange(mocker, vopaiconf)
    strategy.dp = DataProvider(vopaiconf, exchange)
    strategy.vopai_info = vopaiconf.get("vopai", {})
    vopai = strategy.vopai
    vopai.live = True
    vopai.dk = VopaiDataKitchen(vopaiconf, vopai.dd)
    timerange = TimeRange.parse_timerange("20180110-20180114")
    vopai.dk.load_all_pair_histories(timerange)
    sub_timerange = TimeRange.parse_timerange("20180111-20180114")
    corr_df, base_df = vopai.dk.get_base_and_corr_dataframes(sub_timerange, "LTC/BTC")

    return vopai.dk.use_strategy_to_populate_indicators(strategy, corr_df, base_df, "LTC/BTC")


def get_ready_to_train(mocker, vopaiconf):
    strategy = get_patched_vopai_strategy(mocker, vopaiconf)
    exchange = get_patched_exchange(mocker, vopaiconf)
    strategy.dp = DataProvider(vopaiconf, exchange)
    strategy.vopai_info = vopaiconf.get("vopai", {})
    vopai = strategy.vopai
    vopai.live = True
    vopai.dk = VopaiDataKitchen(vopaiconf, vopai.dd)
    timerange = TimeRange.parse_timerange("20180110-20180114")
    vopai.dk.load_all_pair_histories(timerange)
    sub_timerange = TimeRange.parse_timerange("20180111-20180114")
    corr_df, base_df = vopai.dk.get_base_and_corr_dataframes(sub_timerange, "LTC/BTC")
    return corr_df, base_df, vopai, strategy
