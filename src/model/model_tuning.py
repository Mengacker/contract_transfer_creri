from datetime import datetime
import numpy as np
import pandas as pd
from pandas import DataFrame as pandasDataFrame, Series as pandasSeries
from typing import Any
import matplotlib.pyplot as plt
import mlflow
from sklearn.metrics import f1_score, recall_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from lightgbm import plot_metric
import optuna
import tempfile

from src.data_preparation.oversampler import oversampler
from src.config.config_utils import Config

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Disable matplotlib from showing a window
plt.ioff()


def find_best_params(
    X_train: pandasDataFrame = None,
    y_train: pandasSeries = None,
    experiment_path: str = None,
) -> str:
    """
    Function to do the hyperparameters tuning to find best parameters for the model.
    The calibration is done with an Optuna study.
    Args:
        X_train (Pandas DataFrame): training dataset with predictor's features.
        y_train (Pandas Series): training dataset with the target's labels.
        experiment_path (string): MLflow experiment path-name.
    Returns:
        run_id (string): the id that identifies the MLflow run with the best hyper-parameters.
    """
    # Sort predictors
    X_train = X_train[sorted(X_train.columns.to_list())]

    # Retrieve config values
    config_model = Config("../src/config/config_model.yml")
    tracking_uri = config_model.get_config_value("mlflow_config.tracking_uri")
    run_name = config_model.get_config_value("mlflow_config.run_name.tuning")
    tags = config_model.get_config_value("mlflow_config.tags.tuning")

    n_trials = config_model.get_config_value("tuning_params.n_trials")
    metric_direction = config_model.get_config_value("tuning_params.metric_direction")
    tunable_params = config_model.get_config_value("tuning_params.tunable_params")
    eval_metric = config_model.get_config_value("tuning_params.eval_metric")
    f1_score_average = config_model.get_config_value("tuning_params.f1_score_average")
    if f1_score_average != "None":
        eval_metric = eval_metric + "_" + f1_score_average
    run_name = run_name + "_with_" + eval_metric

    # MlFlow settings
    logger.info("Setting mlflow configurations")
    current_date = datetime.today().date()
    mlflow.set_registry_uri(tracking_uri)
    mlflow.set_experiment(experiment_name=experiment_path)
    experiment_id = mlflow.get_experiment_by_name(experiment_path).experiment_id

    # Execute an optimization by using the above objective function wrapped by `lambda`.
    logger.info(f"Start hyperparameters tuning with {n_trials} trials")
    logger.info(f"Finding best values for the following params: {tunable_params}")

    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=run_name,
        tags=tags,  # {"mlflow.note.content": desc_str},
    ) as run:
        run_id = run.info.run_id
        study = optuna.create_study(
            sampler=optuna.samplers.TPESampler(),
            directions=[metric_direction],
            study_name=experiment_path,
        )
        study.optimize(
            lambda trial: objective(
                trial,
                X_train,
                y_train,
            ),
            n_trials=n_trials,
        )
        # MLflow logging
        mlflow.log_param(
            "cliout_macrosegments", X_train["ENCOD_MACROSEGMENTO"].unique().tolist()
        )
        mlflow.log_param("snapshot_date", str(current_date.replace(day=1)))
        mlflow.log_param("best_trial", study.best_trial)
        mlflow.log_params(study.best_params)
        mlflow.log_param("n_iterations", study.best_trial.user_attrs["best_iteration"])
        mlflow.log_metric("best_score", study.best_value)
        with tempfile.TemporaryDirectory() as tmpdir:
            fname = tmpdir + "/tuning_trials_report.csv"
            study.trials_dataframe().to_csv(fname, index=False)
            mlflow.log_artifact(fname)

    logger.info(f"Hyperparameters tuning ended. Best eval metric: {study.best_value}")
    return run_id


def objective(
    trial: Any = None, X: pandasDataFrame = None, y: pandasSeries = None
) -> float:
    """
    Function to define the objective function to be optimized using the Optuna study.
    Args:
        trial (any): Trial object from Optuna.
        X (Pandas DataFrame): training dataset with predictor's features.
        y (Pandas Series): training dataset with the target's labels.
    Returns:
        mean_eval_score (float): the metric value to be optimized.
    """

    with mlflow.start_run(
        nested=True, tags={"mlflow.note.content": "Hyperparameters tuning child run"}
    ):
        # Load configurations
        config_model = Config("../src/config/config_model.yml")
        lgb_params = config_model.get_config_value("training_params.lgb_fixed_params")
        eval_metric = config_model.get_config_value("tuning_params.eval_metric")
        f1_score_average = config_model.get_config_value(
            "tuning_params.f1_score_average"
        )
        stopping_round = config_model.get_config_value("tuning_params.stopping_round")
        max_estimators = config_model.get_config_value("tuning_params.max_estimators")
        n_folds = config_model.get_config_value("tuning_params.n_folds")
        early_stop = config_model.get_config_value("tuning_params.early_stop")

        # Define search space
        params_search_space = {
            # Tuned params
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 7, 4095),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 500),
            # Bagging data strategy
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            # Unbalance management
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 25.0),
        }

        # Handle early stop and subsequent number of boosting round
        if early_stop:
            callbacks = [lgb.early_stopping(stopping_rounds=stopping_round)]
            lgb_params["n_estimators"] = max_estimators + stopping_round
        else:
            callbacks = None
            lgb_params["n_estimators"] = config_model.get_config_value(
                "training_params.lgb_fixed_params.n_estimators"
            )

        # List for storing n_iterations and fold-specific evaluation score
        # params_search_space.update(lgb_params)
        skfold = StratifiedKFold(n_splits=n_folds)
        eval_score = []
        best_iterations = []

        # Instatiate model, fit, predict, and evaluate
        for fold_id, (train_ids, valid_ids) in enumerate(skfold.split(X, y)):
            logger.info("Fold %d / %d" % (fold_id + 1, n_folds))

            # Get training and validation sets
            x_train, y_train = X.iloc[train_ids, :], y.iloc[train_ids]
            x_eval, y_eval = X.iloc[valid_ids, :], y.iloc[valid_ids]

            # Oversample minority class in the training set
            target = y_train.name
            df_train = pd.concat([x_train, y_train], axis=1)
            df_train = oversampler(
                df=df_train, target=target, oversampling_ratio="minority"
            )
            y_train = df_train[target]
            x_train = df_train.drop(columns=[target])

            y_train = y_train.values.ravel()
            y_eval = y_eval.values.ravel()

            clf = lgb.LGBMClassifier(**params_search_space)
            clf.fit(
                x_train,
                y_train,
                eval_set=[(x_train, y_train), (x_eval, y_eval)],
                eval_metric=[eval_metric],
                categorical_feature="auto",  # lista_cat,
                callbacks=callbacks,
            )

            preds = clf.predict(x_eval, num_iteration=clf.best_iteration_)

            if eval_metric == "f1_score":
                performance_metric = f1_score(
                    y_eval, preds, average=f1_score_average
                )  # ["micro", "macro", "samples", "weighted", "binary"]
            elif eval_metric == "recall_score":
                performance_metric = recall_score(
                    y_eval, preds, average="binary", pos_label=1
                )
            elif eval_metric == "brier_score_loss":
                preds_proba = clf.predict_proba(x_eval)[:, 1]
                performance_metric = brier_score_loss(
                    y_true=y_eval, y_proba=preds_proba
                )

            eval_score.append(performance_metric)
            best_iterations.append(clf.best_iteration_)

        # Compute avarage evaluation score
        mean_eval_score = np.mean(eval_score)
        mean_n_iterations = int(np.mean(best_iterations))
        trial.set_user_attr("best_iteration", mean_n_iterations)
        logger.info(f"Mean eval score {eval_metric}: {mean_eval_score} - {eval_score}")
        logger.info(f"Mean number of iterators: {mean_n_iterations}")

        # Logging
        mlflow.log_params(params_search_space)
        mlflow.log_metric("Weighted_" + eval_metric + "_avg", mean_eval_score)
        mlflow.log_figure(plot_metric(clf).get_figure(), "fitting_metrics_default.png")
        for i in range(len(eval_score)):
            mlflow.log_metric("weighted_" + eval_metric + "_fold_{i+1}", eval_score[i])

        # Log also the other fitting metric
        if eval_metric in ["binary", "binary_logloss"]:
            mlflow.log_figure(
                plot_metric(clf, "auc").get_figure(), "fitting_metrics_other.png"
            )
        else:
            mlflow.log_figure(
                plot_metric(clf, "binary_logloss").get_figure(),
                "fitting_metrics_other.png",
            )
    return mean_eval_score
