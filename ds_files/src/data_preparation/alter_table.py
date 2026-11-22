from pyspark.sql import SparkSession
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def alter_table(pk_col: list[str], table_path: str) -> list[str]:
    """
    Function to alter a table to add a primary key (or a list of primary keys), set it to not to be null and enable change data feed true.
    Args:
        pk_col (list[string]): list of columns to be set as primary key.
        table_path (string): path of the table.
    Returns:
        l_messages (list[string]): list of messages.
    """
    assert isinstance(pk_col, list) and (len(pk_col) > 0), logger.warning(
        f"""Primary key {str(pk_col)} is not a list or is empty, please check your input"""
    )

    spark: SparkSession = SparkSession.builder.getOrCreate()

    l_messages = []
    for col in pk_col:
        try:
            spark.sql(
                "ALTER TABLE {tab} ALTER COLUMN {pk} SET NOT NULL".format(
                    tab=table_path, pk=col
                )
            )
            message = f"Set column {col} not nullable in the table {table_path} done"
        except Exception:
            message = (
                f"Set column {col} not nullable in the table {table_path} already done"
            )
        logger.info(message)
        l_messages.append(message)

    pk_1col = pk_col[0]
    try:
        spark.sql(
            "ALTER TABLE {tab} ADD PRIMARY KEY({pk})".format(tab=table_path, pk=pk_1col)
        )
        message = f"Set primary key to {str(pk_1col)} in the table {table_path} done"
    except Exception:
        message = (
            f"Set primary key to {str(pk_1col)} in the table {table_path} already done"
        )
    logger.info(message)
    l_messages.append(message)

    try:
        spark.sql(
            "ALTER TABLE {tab} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)".format(
                tab=table_path,
            )
        )
        message = f"Set TBLPROPERTIES delta.enableChangeDataFeed = true in the table {table_path} done"
    except Exception:
        message = f"Set TBLPROPERTIES delta.enableChangeDataFeed = true in the table {table_path} already done"
    logger.info(message)
    l_messages.append(message)
    return l_messages
