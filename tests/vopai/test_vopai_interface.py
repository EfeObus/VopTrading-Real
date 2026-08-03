import logging
import shutil
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from voptrade.configuration import TimeRange
from voptrade.data.dataprovider import DataProvider
from voptrade.enums import RunMode
from voptrade.exceptions import OperationalException
from voptrade.vopai.data_kitchen import VopaiDataKitchen
from voptrade.vopai.utils import download_all_data_for_training, get_required_data_timerange
from voptrade.optimize.backtesting import Backtesting
from voptrade.persistence import Trade
from voptrade.plugins.pairlistmanager import PairListManager
from tests.conftest import (
    EXMS,
    create_mock_trades,
    get_patched_exchange,
    is_arm,
    is_mac,
    log_has_re,
)
from tests.vopai.conftest import (
    get_patched_vopai_strategy,
    make_rl_config,
    mock_pytorch_mlp_model_training_parameters,
)


def can_run_model(model: str) -> None:
    is_pytorch_model = "Reinforcement" in model or "PyTorch" in model

    if is_pytorch_model and is_mac():
        pytest.skip("Reinforcement learning / PyTorch module not available on intel based Mac OS.")


@pytest.mark.parametrize(
    "model, pca, dbscan, float32, can_short, shuffle, buffer, noise",
    [
        ("LightGBMRegressor", True, False, True, True, False, 0, 0),
        ("XGBoostRegressor", False, True, False, True, False, 10, 0.05),
        ("XGBoostRFRegressor", False, False, False, True, False, 0, 0),
        ("PyTorchMLPRegressor", False, False, False, False, False, 0, 0),
        ("PyTorchTransformerRegressor", False, False, False, False, False, 0, 0),
        ("ReinforcementLearner", False, True, False, True, False, 0, 0),
        ("ReinforcementLearner_multiproc", False, False, False, True, False, 0, 0),
        ("ReinforcementLearner_test_3ac", False, False, False, False, False, 0, 0),
        ("ReinforcementLearner_test_3ac", False, False, False, True, False, 0, 0),
        ("ReinforcementLearner_test_4ac", False, False, False, True, False, 0, 0),
    ],
)
def test_extract_data_and_train_model_Standard(
    mocker, vopai_conf, model, pca, dbscan, float32, can_short, shuffle, buffer, noise
):
    can_run_model(model)

    test_tb = True
    if is_mac():
        test_tb = False

    model_save_ext = "joblib"
    vopai_conf.update({"vopaimodel": model})
    vopai_conf.update({"timerange": "20180110-20180130"})
    vopai_conf.update({"strategy": "vopai_test_strat"})
    vopai_conf["vopai"]["feature_parameters"].update({"principal_component_analysis": pca})
    vopai_conf["vopai"]["feature_parameters"].update({"use_DBSCAN_to_remove_outliers": dbscan})
    vopai_conf.update({"reduce_df_footprint": float32})
    vopai_conf["vopai"]["feature_parameters"].update({"shuffle_after_split": shuffle})
    vopai_conf["vopai"]["feature_parameters"].update({"buffer_train_data_candles": buffer})
    vopai_conf["vopai"]["feature_parameters"].update({"noise_standard_deviation": noise})

    if "ReinforcementLearner" in model:
        model_save_ext = "zip"
        vopai_conf = make_rl_config(vopai_conf)
        # test the RL guardrails
        vopai_conf["vopai"]["feature_parameters"].update({"use_SVM_to_remove_outliers": True})
        vopai_conf["vopai"]["feature_parameters"].update({"DI_threshold": 2})
        vopai_conf["vopai"]["data_split_parameters"].update({"shuffle": True})

    if "test_3ac" in model or "test_4ac" in model:
        vopai_conf["vopaimodel_path"] = str(Path(__file__).parents[1] / "vopai" / "test_models")
        vopai_conf["vopai"]["rl_config"]["drop_ohlc_from_features"] = True

    if "PyTorch" in model:
        model_save_ext = "zip"
        pytorch_mlp_mtp = mock_pytorch_mlp_model_training_parameters()
        vopai_conf["vopai"]["model_training_parameters"].update(pytorch_mlp_mtp)
        if "Transformer" in model:
            # transformer model takes a window, unlike the MLP regressor
            vopai_conf.update({"conv_width": 10})

    strategy = get_patched_vopai_strategy(mocker, vopai_conf)
    exchange = get_patched_exchange(mocker, vopai_conf)
    strategy.dp = DataProvider(vopai_conf, exchange)
    strategy.vopai_info = vopai_conf.get("vopai", {})
    vopai = strategy.vopai
    vopai.live = True
    vopai.activate_tensorboard = test_tb
    vopai.can_short = can_short
    vopai.dk = VopaiDataKitchen(vopai_conf)
    vopai.dk.live = True
    vopai.dk.set_paths("ADA/BTC", 10000)
    timerange = TimeRange.parse_timerange("20180110-20180130")
    vopai.dd.load_all_pair_histories(timerange, vopai.dk)

    vopai.dd.pair_dict = MagicMock()

    data_load_timerange = TimeRange.parse_timerange("20180125-20180130")
    new_timerange = TimeRange.parse_timerange("20180127-20180130")
    vopai.dk.set_paths("ADA/BTC", None)

    vopai.train_timer("start", "ADA/BTC")
    vopai.extract_data_and_train_model(
        new_timerange, "ADA/BTC", strategy, vopai.dk, data_load_timerange
    )
    vopai.train_timer("stop", "ADA/BTC")
    vopai.dd.save_metric_tracker_to_disk()
    vopai.dd.save_drawer_to_disk()

    assert Path(vopai.dk.full_path / "metric_tracker.json").is_file()
    assert Path(vopai.dk.full_path / "pair_dictionary.json").is_file()
    assert Path(
        vopai.dk.data_path / f"{vopai.dk.model_filename}_model.{model_save_ext}"
    ).is_file()
    assert Path(vopai.dk.data_path / f"{vopai.dk.model_filename}_metadata.json").is_file()
    assert Path(vopai.dk.data_path / f"{vopai.dk.model_filename}_trained_df.pkl").is_file()

    shutil.rmtree(Path(vopai.dk.full_path))


@pytest.mark.parametrize(
    "model, strat",
    [
        ("LightGBMRegressorMultiTarget", "vopai_test_multimodel_strat"),
        ("XGBoostRegressorMultiTarget", "vopai_test_multimodel_strat"),
        ("LightGBMClassifierMultiTarget", "vopai_test_multimodel_classifier_strat"),
    ],
)
@pytest.mark.filterwarnings(r"ignore:.*__sklearn_tags__.*:DeprecationWarning")
def test_extract_data_and_train_model_MultiTargets(mocker, vopai_conf, model, strat):
    can_run_model(model)

    vopai_conf.update({"timerange": "20180110-20180130"})
    vopai_conf.update({"strategy": strat})
    vopai_conf.update({"vopaimodel": model})
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

    assert len(vopai.dk.label_list) == 2
    assert Path(vopai.dk.data_path / f"{vopai.dk.model_filename}_model.joblib").is_file()
    assert Path(vopai.dk.data_path / f"{vopai.dk.model_filename}_metadata.json").is_file()
    assert Path(vopai.dk.data_path / f"{vopai.dk.model_filename}_trained_df.pkl").is_file()
    assert len(vopai.dk.data["training_features_list"]) == 14

    shutil.rmtree(Path(vopai.dk.full_path))


@pytest.mark.parametrize(
    "model",
    [
        "LightGBMClassifier",
        "XGBoostClassifier",
        "XGBoostRFClassifier",
        "SKLearnRandomForestClassifier",
        "PyTorchMLPClassifier",
    ],
)
def test_extract_data_and_train_model_Classifiers(mocker, vopai_conf, model):
    can_run_model(model)

    vopai_conf.update({"vopaimodel": model})
    vopai_conf.update({"strategy": "vopai_test_classifier"})
    vopai_conf.update({"timerange": "20180110-20180130"})
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

    if "PyTorch" in model:
        pytorch_mlp_mtp = mock_pytorch_mlp_model_training_parameters()
        vopai_conf["vopai"]["model_training_parameters"].update(pytorch_mlp_mtp)

    if vopai.dd.model_type == "joblib":
        model_file_extension = ".joblib"
    elif vopai.dd.model_type == "pytorch":
        model_file_extension = ".zip"
    else:
        raise Exception(
            f"Unsupported model type: {vopai.dd.model_type}, can't assign model_file_extension"
        )

    assert Path(
        vopai.dk.data_path / f"{vopai.dk.model_filename}_model{model_file_extension}"
    ).exists()
    assert Path(vopai.dk.data_path / f"{vopai.dk.model_filename}_metadata.json").exists()
    assert Path(vopai.dk.data_path / f"{vopai.dk.model_filename}_trained_df.pkl").exists()

    shutil.rmtree(Path(vopai.dk.full_path))


@pytest.mark.parametrize(
    "model, num_files, strat",
    [
        ("LightGBMRegressor", 2, "vopai_test_strat"),
        ("XGBoostRegressor", 2, "vopai_test_strat"),
        ("PyTorchMLPRegressor", 2, "vopai_test_strat"),
        ("PyTorchTransformerRegressor", 2, "vopai_test_strat"),
        ("ReinforcementLearner", 3, "vopai_rl_test_strat"),
        ("XGBoostClassifier", 2, "vopai_test_classifier"),
        ("LightGBMClassifier", 2, "vopai_test_classifier"),
        ("PyTorchMLPClassifier", 2, "vopai_test_classifier"),
    ],
)
def test_start_backtesting(mocker, vopai_conf, model, num_files, strat, caplog):
    can_run_model(model)
    test_tb = True
    if is_mac() and not is_arm():
        test_tb = False

    vopai_conf.get("vopai", {}).update({"save_backtest_models": True})
    vopai_conf["runmode"] = RunMode.BACKTEST

    Trade.use_db = False

    vopai_conf.update({"vopaimodel": model})
    vopai_conf.update({"timerange": "20180120-20180130"})
    vopai_conf.update({"strategy": strat})

    if "ReinforcementLearner" in model:
        vopai_conf = make_rl_config(vopai_conf)

    if "test_4ac" in model:
        vopai_conf["vopaimodel_path"] = str(Path(__file__).parents[1] / "vopai" / "test_models")

    if "PyTorch" in model:
        pytorch_mlp_mtp = mock_pytorch_mlp_model_training_parameters()
        vopai_conf["vopai"]["model_training_parameters"].update(pytorch_mlp_mtp)
        if "Transformer" in model:
            # transformer model takes a window, unlike the MLP regressor
            vopai_conf.update({"conv_width": 10})

    vopai_conf.get("vopai", {}).get("feature_parameters", {}).update(
        {"indicator_periods_candles": [2]}
    )

    strategy = get_patched_vopai_strategy(mocker, vopai_conf)
    exchange = get_patched_exchange(mocker, vopai_conf)
    strategy.dp = DataProvider(vopai_conf, exchange)
    strategy.vopai_info = vopai_conf.get("vopai", {})
    vopai = strategy.vopai
    vopai.live = False
    vopai.activate_tensorboard = test_tb
    vopai.dk = VopaiDataKitchen(vopai_conf)
    timerange = TimeRange.parse_timerange("20180110-20180130")
    vopai.dd.load_all_pair_histories(timerange, vopai.dk)
    sub_timerange = TimeRange.parse_timerange("20180110-20180130")
    _, base_df = vopai.dd.get_base_and_corr_dataframes(sub_timerange, "LTC/BTC", vopai.dk)
    df = base_df[vopai_conf["timeframe"]]

    metadata = {"pair": "LTC/BTC"}
    vopai.dk.set_paths("LTC/BTC", None)
    vopai.start_backtesting(df, metadata, vopai.dk, strategy)
    model_folders = [x for x in vopai.dd.full_path.iterdir() if x.is_dir()]

    assert len(model_folders) == num_files
    Trade.use_db = True
    Backtesting.cleanup()
    shutil.rmtree(Path(vopai.dk.full_path))


def test_start_backtesting_appends_null_predictions_when_training_fails(mocker, vopai_conf):
    vopai_conf.update({"vopaimodel": "LightGBMRegressor"})
    vopai_conf.update({"timerange": "20180120-20180130"})
    vopai_conf.update({"strategy": "vopai_test_strat"})
    vopai_conf.get("vopai", {}).get("feature_parameters", {}).update(
        {"indicator_periods_candles": [2]}
    )

    strategy = get_patched_vopai_strategy(mocker, vopai_conf)
    exchange = get_patched_exchange(mocker, vopai_conf)
    strategy.dp = DataProvider(vopai_conf, exchange)
    strategy.vopai_info = vopai_conf.get("vopai", {})
    vopai = strategy.vopai
    vopai.live = False
    vopai.dk = VopaiDataKitchen(vopai_conf)
    timerange = TimeRange.parse_timerange("20180110-20180130")
    vopai.dd.load_all_pair_histories(timerange, vopai.dk)
    sub_timerange = TimeRange.parse_timerange("20180110-20180130")
    _, base_df = vopai.dd.get_base_and_corr_dataframes(sub_timerange, "LTC/BTC", vopai.dk)
    df = base_df[vopai_conf["timeframe"]]

    mocker.patch.object(
        vopai,
        "train",
        side_effect=OperationalException("all training data dropped due to NaNs"),
    )
    predict_mock = mocker.patch.object(vopai, "predict")

    metadata = {"pair": "LTC/BTC"}
    vopai.dk.set_paths("LTC/BTC", None)
    dk = vopai.start_backtesting(df, metadata, vopai.dk, strategy)

    predict_mock.assert_not_called()
    assert "do_predict" in dk.return_dataframe
    assert dk.return_dataframe["do_predict"].eq(0).all()

    # Second run reuses the cached predictions - ensure 2nd backtest run or
    # API rendering does not trigger training or prediction
    strategy2 = get_patched_vopai_strategy(mocker, vopai_conf)
    strategy2.dp = DataProvider(vopai_conf, exchange)
    strategy2.vopai_info = vopai_conf.get("vopai", {})
    vopai2 = strategy2.vopai
    vopai2.live = False
    vopai2.dk = VopaiDataKitchen(vopai_conf)
    vopai2.dd.load_all_pair_histories(timerange, vopai2.dk)
    train_mock = mocker.patch.object(vopai2, "train")
    predict_mock2 = mocker.patch.object(vopai2, "predict")
    vopai2.dk.set_paths("LTC/BTC", None)
    dk2 = vopai2.start_backtesting(df, metadata, vopai2.dk, strategy2)

    train_mock.assert_not_called()
    predict_mock2.assert_not_called()
    assert len(dk2.return_dataframe) == len(df)

    shutil.rmtree(Path(vopai.dk.full_path))


def test_start_backtesting_subdaily_backtest_period(mocker, vopai_conf):
    vopai_conf.update({"timerange": "20180120-20180124"})
    vopai_conf["runmode"] = "backtest"
    vopai_conf.get("vopai", {}).update(
        {
            "backtest_period_days": 0.5,
            "save_backtest_models": True,
        }
    )
    vopai_conf.get("vopai", {}).get("feature_parameters", {}).update(
        {"indicator_periods_candles": [2]}
    )
    strategy = get_patched_vopai_strategy(mocker, vopai_conf)
    exchange = get_patched_exchange(mocker, vopai_conf)
    strategy.dp = DataProvider(vopai_conf, exchange)
    strategy.vopai_info = vopai_conf.get("vopai", {})
    vopai = strategy.vopai
    vopai.live = False
    vopai.dk = VopaiDataKitchen(vopai_conf)
    timerange = TimeRange.parse_timerange("20180110-20180130")
    vopai.dd.load_all_pair_histories(timerange, vopai.dk)
    sub_timerange = TimeRange.parse_timerange("20180110-20180130")
    _, base_df = vopai.dd.get_base_and_corr_dataframes(sub_timerange, "LTC/BTC", vopai.dk)
    df = base_df[vopai_conf["timeframe"]]

    metadata = {"pair": "LTC/BTC"}
    vopai.start_backtesting(df, metadata, vopai.dk, strategy)
    model_folders = [x for x in vopai.dd.full_path.iterdir() if x.is_dir()]

    assert len(model_folders) == 9

    shutil.rmtree(Path(vopai.dk.full_path))


def test_start_backtesting_from_existing_folder(mocker, vopai_conf, caplog):
    vopai_conf.update({"timerange": "20180120-20180130"})
    vopai_conf["runmode"] = "backtest"
    vopai_conf.get("vopai", {}).update({"save_backtest_models": True})
    vopai_conf.get("vopai", {}).get("feature_parameters", {}).update(
        {"indicator_periods_candles": [2]}
    )
    strategy = get_patched_vopai_strategy(mocker, vopai_conf)
    exchange = get_patched_exchange(mocker, vopai_conf)
    strategy.dp = DataProvider(vopai_conf, exchange)
    strategy.vopai_info = vopai_conf.get("vopai", {})
    vopai = strategy.vopai
    vopai.live = False
    vopai.dk = VopaiDataKitchen(vopai_conf)
    timerange = TimeRange.parse_timerange("20180110-20180130")
    vopai.dd.load_all_pair_histories(timerange, vopai.dk)
    sub_timerange = TimeRange.parse_timerange("20180101-20180130")
    _, base_df = vopai.dd.get_base_and_corr_dataframes(sub_timerange, "LTC/BTC", vopai.dk)
    df = base_df[vopai_conf["timeframe"]]

    pair = "ADA/BTC"
    metadata = {"pair": pair}
    vopai.dk.pair = pair
    vopai.start_backtesting(df, metadata, vopai.dk, strategy)
    model_folders = [x for x in vopai.dd.full_path.iterdir() if x.is_dir()]

    assert len(model_folders) == 2

    # without deleting the existing folder structure, re-run

    vopai_conf.update({"timerange": "20180120-20180130"})
    strategy = get_patched_vopai_strategy(mocker, vopai_conf)
    exchange = get_patched_exchange(mocker, vopai_conf)
    strategy.dp = DataProvider(vopai_conf, exchange)
    strategy.vopai_info = vopai_conf.get("vopai", {})
    vopai = strategy.vopai
    vopai.live = False
    vopai.dk = VopaiDataKitchen(vopai_conf)
    timerange = TimeRange.parse_timerange("20180110-20180130")
    vopai.dd.load_all_pair_histories(timerange, vopai.dk)
    sub_timerange = TimeRange.parse_timerange("20180110-20180130")
    _, base_df = vopai.dd.get_base_and_corr_dataframes(sub_timerange, "LTC/BTC", vopai.dk)
    df = base_df[vopai_conf["timeframe"]]

    pair = "ADA/BTC"
    metadata = {"pair": pair}
    vopai.dk.pair = pair
    vopai.start_backtesting(df, metadata, vopai.dk, strategy)

    assert log_has_re(
        "Found backtesting prediction file ",
        caplog,
    )

    pair = "ETH/BTC"
    metadata = {"pair": pair}
    vopai.dk.pair = pair
    vopai.start_backtesting(df, metadata, vopai.dk, strategy)

    path = vopai.dd.full_path / vopai.dk.backtest_predictions_folder
    prediction_files = [x for x in path.iterdir() if x.is_file()]
    assert len(prediction_files) == 2

    shutil.rmtree(Path(vopai.dk.full_path))


def test_backtesting_fit_live_predictions(mocker, vopai_conf, caplog):
    vopai_conf["runmode"] = "backtest"
    vopai_conf.get("vopai", {}).update({"fit_live_predictions_candles": 10})
    strategy = get_patched_vopai_strategy(mocker, vopai_conf)
    exchange = get_patched_exchange(mocker, vopai_conf)
    strategy.dp = DataProvider(vopai_conf, exchange)
    strategy.vopai_info = vopai_conf.get("vopai", {})
    vopai = strategy.vopai
    vopai.live = False
    vopai.dk = VopaiDataKitchen(vopai_conf)
    timerange = TimeRange.parse_timerange("20180128-20180130")
    vopai.dd.load_all_pair_histories(timerange, vopai.dk)
    sub_timerange = TimeRange.parse_timerange("20180129-20180130")
    corr_df, base_df = vopai.dd.get_base_and_corr_dataframes(sub_timerange, "LTC/BTC", vopai.dk)
    df = vopai.dk.use_strategy_to_populate_indicators(strategy, corr_df, base_df, "LTC/BTC")
    df = strategy.set_vopai_targets(df.copy(), metadata={"pair": "LTC/BTC"})
    df = vopai.dk.remove_special_chars_from_feature_names(df)
    vopai.dk.get_unique_classes_from_labels(df)
    vopai.dk.pair = "ADA/BTC"
    vopai.dk.full_df = df.fillna(0)

    assert "&-s_close_mean" not in vopai.dk.full_df.columns
    assert "&-s_close_std" not in vopai.dk.full_df.columns
    vopai.backtesting_fit_live_predictions(vopai.dk)
    assert "&-s_close_mean" in vopai.dk.full_df.columns
    assert "&-s_close_std" in vopai.dk.full_df.columns
    shutil.rmtree(Path(vopai.dk.full_path))


def test_plot_feature_importance(mocker, vopai_conf):
    from voptrade.vopai.utils import plot_feature_importance

    vopai_conf.update({"timerange": "20180110-20180130"})
    vopai_conf.get("vopai", {}).get("feature_parameters", {}).update(
        {"princpial_component_analysis": "true"}
    )

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

    vopai.dd.pair_dict = {
        "ADA/BTC": {
            "model_filename": "fake_name",
            "trained_timestamp": 1,
            "data_path": "",
            "extras": {},
        }
    }

    data_load_timerange = TimeRange.parse_timerange("20180110-20180130")
    new_timerange = TimeRange.parse_timerange("20180120-20180130")
    vopai.dk.set_paths("ADA/BTC", None)

    vopai.extract_data_and_train_model(
        new_timerange, "ADA/BTC", strategy, vopai.dk, data_load_timerange
    )

    model = vopai.dd.load_data("ADA/BTC", vopai.dk)

    plot_feature_importance(model, "ADA/BTC", vopai.dk)

    assert Path(vopai.dk.data_path / f"{vopai.dk.model_filename}.html")

    shutil.rmtree(Path(vopai.dk.full_path))


@pytest.mark.parametrize(
    "timeframes,corr_pairs",
    [
        (["5m"], ["ADA/BTC", "DASH/BTC"]),
        (["5m"], ["ADA/BTC", "DASH/BTC", "ETH/USDT"]),
        (["5m", "15m"], ["ADA/BTC", "DASH/BTC", "ETH/USDT"]),
    ],
)
def test_vopai_informative_pairs(mocker, vopai_conf, timeframes, corr_pairs):
    vopai_conf["vopai"]["feature_parameters"].update(
        {
            "include_timeframes": timeframes,
            "include_corr_pairlist": corr_pairs,
        }
    )
    strategy = get_patched_vopai_strategy(mocker, vopai_conf)
    exchange = get_patched_exchange(mocker, vopai_conf)
    pairlists = PairListManager(exchange, vopai_conf)
    strategy.dp = DataProvider(vopai_conf, exchange, pairlists)
    pairlist = strategy.dp.current_whitelist()

    pairs_a = strategy.informative_pairs()
    assert len(pairs_a) == 0
    pairs_b = strategy.gather_informative_pairs()
    # we expect unique pairs * timeframes
    assert len(pairs_b) == len(set(pairlist + corr_pairs)) * len(timeframes)


def test_start_set_train_queue(mocker, vopai_conf, caplog):
    strategy = get_patched_vopai_strategy(mocker, vopai_conf)
    exchange = get_patched_exchange(mocker, vopai_conf)
    pairlist = PairListManager(exchange, vopai_conf)
    strategy.dp = DataProvider(vopai_conf, exchange, pairlist)
    strategy.vopai_info = vopai_conf.get("vopai", {})
    vopai = strategy.vopai
    vopai.live = False

    vopai.train_queue = vopai._set_train_queue()

    assert log_has_re(
        "Set fresh train queue from whitelist.",
        caplog,
    )


def test_get_required_data_timerange(mocker, vopai_conf):
    time_range = get_required_data_timerange(vopai_conf)
    assert (time_range.stopts - time_range.startts) == 177300


def test_download_all_data_for_training(mocker, vopai_conf, caplog, tmp_path):
    caplog.set_level(logging.DEBUG)
    strategy = get_patched_vopai_strategy(mocker, vopai_conf)
    exchange = get_patched_exchange(mocker, vopai_conf)
    pairlist = PairListManager(exchange, vopai_conf)
    strategy.dp = DataProvider(vopai_conf, exchange, pairlist)
    vopai_conf["pairs"] = vopai_conf["exchange"]["pair_whitelist"]
    vopai_conf["datadir"] = tmp_path
    download_all_data_for_training(strategy.dp, vopai_conf)

    assert log_has_re(
        "Downloading",
        caplog,
    )


@pytest.mark.usefixtures("init_persistence")
@pytest.mark.parametrize("dp_exists", [(False), (True)])
def test_get_state_info(mocker, vopai_conf, dp_exists, caplog, tickers):
    if is_mac():
        pytest.skip("Reinforcement learning module not available on intel based Mac OS")

    vopai_conf.update({"vopaimodel": "ReinforcementLearner"})
    vopai_conf.update({"timerange": "20180110-20180130"})
    vopai_conf.update({"strategy": "vopai_rl_test_strat"})
    vopai_conf = make_rl_config(vopai_conf)
    vopai_conf["entry_pricing"]["price_side"] = "same"
    vopai_conf["exit_pricing"]["price_side"] = "same"

    strategy = get_patched_vopai_strategy(mocker, vopai_conf)
    exchange = get_patched_exchange(mocker, vopai_conf)
    ticker_mock = MagicMock(return_value=tickers()["ETH/BTC"])
    mocker.patch(f"{EXMS}.fetch_ticker", ticker_mock)
    strategy.dp = DataProvider(vopai_conf, exchange)

    if not dp_exists:
        strategy.dp._exchange = None

    strategy.vopai_info = vopai_conf.get("vopai", {})
    vopai = strategy.vopai
    vopai.data_provider = strategy.dp
    vopai.live = True

    Trade.use_db = True
    create_mock_trades(MagicMock(return_value=0.0025), False, True)
    vopai.get_state_info("ADA/BTC")
    vopai.get_state_info("ETH/BTC")

    if not dp_exists:
        assert log_has_re(
            "No exchange available",
            caplog,
        )
