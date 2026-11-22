from pyspark.sql import DataFrame as pyskDataFrame
import pyspark.sql.functions as F
import yaml
import os

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _load_config(config_path=None):
    """Funzione per caricare i file di config ma che non ci dovrebbe essere perchè esiste già quella in config_utils.py che sarebbe da usare"""
    if config_path is None:
        config_path = os.getenv("ARCHIVER_CONFIG_PATH", "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def dataset_splitting_classifier(
    master: pyskDataFrame,
) -> pyskDataFrame:
    """
    Function to split the master dataset into training, calibration and test datasets, then this action is tracked with categories in the column "UTILITY_SET".
    Args:
        master (PySpark DataFrame): an input master dataset.
        pk (str): a string representing the primary key.
    Returns:
        The initial master dataset as PySpark DataFrame with the column "UTILITY_SET" added.
    """
    pk = "PK_PROPOSAL_LINE_ITEM"
    # Retrieve config values

    config_model = _load_config(
        "/Workspace/AI/ccvol/ccvol_model/files/src/config/config_model.yml"
    )
    seed_sampleBy = config_model["random_state"]
    train_perc = config_model["splitting_data_params"]["train_perc"]
    calib_perc = config_model["splitting_data_params"]["calib_perc"]
    test_perc = config_model["splitting_data_params"]["test_perc"]

    config_data = _load_config(
        "/Workspace/AI/ccvol/ccvol_model/files/src/config/config_data.yml"
    )
    target = config_data["target_features"]["training"]
    target = list(target.keys())[0]  # FLG_CLI_MOROSO

    # config = Config(config_path=config_path, name_list=['config_setup.yml','config_model.yml'])
    # target = config.get_config_value("targets")[0]["target_column"]
    # seed_sampleBy = config.get_config_value("random_state")
    # train_perc = config.get_config_value("splitting_data_params.train_perc")
    # calib_perc = config.get_config_value("splitting_data_params.calib_perc")
    # test_perc = config.get_config_value("splitting_data_params.test_perc")

    assert (
        train_perc + calib_perc + test_perc == 1
    ), 'The sum of the "splitting_data_params" must be equal to 1'

    master_df = master.where(f"{target} IS NOT NULL")
    df_null = master.where(f"{target} IS NULL").withColumn("UTILITY_SET", F.lit(None))

    # Sample a training dataset
    nrows_total = master_df.count()
    logger.info(f"{nrows_total} records in master before splitting")

    fractions = {True: 1 - test_perc, False: 1 - test_perc}
    df_to_split = master_df.select(pk, target)
    training = df_to_split.stat.sampleBy(target, fractions, seed=seed_sampleBy)

    nrows_train_and_calib = training.count()
    nrows_test = nrows_total - nrows_train_and_calib
    logger.info(
        f"{nrows_train_and_calib} records for training and calibration "  # % {100*nrows_train_and_calib/nrows_total}"
    )
    logger.info(
        f"{nrows_test} records for test "  # % {100*nrows_test/nrows_total}"
    )

    # Adjust tune_perc of whole dataset to the splitted dataset train = total-test
    calib_perc_adj = calib_perc * (1 - test_perc)

    # Subsample a calibration datasets
    fractions = {True: calib_perc_adj, False: calib_perc_adj}
    calibration = training.stat.sampleBy(target, fractions, seed=seed_sampleBy)

    nrows_calib = calibration.count()
    nrows_train = nrows_train_and_calib - nrows_calib
    logger.info(
        f"{nrows_train} records for training "  # % {100*nrows_train/nrows_total}"
    )
    logger.info(
        f"{nrows_calib} records for calibration "  # % {100*nrows_calib/nrows_total}"
    )

    # Create the "UTILITY_SET" cloumn and re-join to master dataset
    calibration = calibration.withColumn("UTILITY_SET", F.lit("calib"))

    training = training.join(
        calibration.select(pk, "UTILITY_SET"),
        on=pk,
        how="left",
    )
    training = training.fillna({"UTILITY_SET": "train"})
    master = master_df.join(
        training.select(pk, "UTILITY_SET"),
        on=pk,
        how="left",
    )
    master = master.fillna({"UTILITY_SET": "test"})
    logger.info(
        f"{master.count()-nrows_total} records of difference in master after splitting (must be 0)"
    )

    master = master.unionByName(df_null)

    return master
