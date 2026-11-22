import pandas as pd
from pandas import DataFrame as pandasDataFrame
from imblearn.over_sampling import RandomOverSampler


def oversampler(
    df: pandasDataFrame,
    target: str,
    oversampling_ratio: str = "minority",
    random_state: int = 42,
) -> pandasDataFrame:
    """
    Function to do the oversampling of the target's minority class on a dataset.
    Args:
        df (Pandas Dataframe): an input dataset.
        target (string): name of the target column.
        oversampling_ratio (string): ratio of oversampling, default = 'minority'.
        random_state (integer): random seed for reproducibility, default = 42.
    Return:
        The oversampled dataset as Pandas DataFrame
    """

    # Separate targets from predictors
    y = df[target]
    X = df.drop(columns=target)

    # Oversampling of the minority target
    oversample = RandomOverSampler(
        sampling_strategy=oversampling_ratio, random_state=random_state
    )
    X_over, y_over = oversample.fit_resample(X, y)

    # Concatenate oversampled dataframes into a single one
    df_over = pd.concat([X_over, y_over], axis=1)
    return df_over
