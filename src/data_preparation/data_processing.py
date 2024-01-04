from typing import Dict, Optional, Tuple, Type

import pandas as pd
from imblearn.over_sampling import SMOTENC
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer


def create_target_variable(
    df: pd.DataFrame, config: Dict[str, Optional[str]]
) -> pd.DataFrame:
    """
    Create a binary target column based on a specified threshold.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - config (Dict[str, Optional[str]]): Configuration parameters.
        - feature (str): The feature column for thresholding.
        - threshold (float):
          The threshold for creating the binary target column.
        - target (str): Name of the target column.

    Returns:
    - pd.DataFrame: DataFrame with the binary target column added.
    """
    # Remove rows without data for feature
    df = df[df[config.get("feature")].notna()]
    df[config.get("target")] = df[config.get("feature")] >= config.get(
        "threshold"
    )
    return df


def split_data(
    df: pd.DataFrame, config: dict
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the input DataFrame into training and testing sets.

    Parameters:
        - df (pd.DataFrame): The DataFrame containing the dataset.
        - config (dict):
          A dictionary containing configuration parameters, including:
            - "target" (str): The target variable column name.
            - "feature" (str): The feature used to generate the target,
              to be excluded from predictors.
            - "test_size" (float):
              The proportion of the dataset to include in the test split.
            - "random_state" (int):
              Seed for random number generation for reproducibility.

    Returns:
        - Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
          X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=[config.get("target"), config.get("feature")], axis=1)
    y = df[config.get("target")]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.get("test_size"),
        random_state=config.get("random_state"),
    )
    return X_train, X_test, y_train, y_test


def sort_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort columns of a DataFrame alphabetically.

    Parameters:
    - df (pd.DataFrame): The DataFrame whose columns are to be sorted.

    Returns:
    - pd.DataFrame: DataFrame with columns sorted alphabetically.
    """
    return df.sort_index(axis=1)


def handle_missing_data(config, X_train, X_test=None):
    """
    Handle missing data in a DataFrame using different approaches.

    Parameters:
    - df (pd.DataFrame): Input DataFrame with missing values.
    - config (Dict[str, Optional[str]]): Configuration parameters.
        - imputation_method (str): Approach to handle missing data.
            Can be one of
            - DN (Do Nothing);
            - CCA (Complete-Case Analysis); or
            - IWM (Imputation with Median/Mode)

    Returns:
    - pd.DataFrame: DataFrame with missing values handled.
    """
    imputation_method = config.get("imputation_method")

    if imputation_method == "DN":
        # Approach 0: Do Nothing
        imputed_train = X_train
        if X_test is not None:
            imputed_test = X_test

    elif imputation_method == "CCA":
        # Approach 1: Complete-Case Analysis
        imputed_train = X_train.dropna().copy()
        if X_test is not None:
            imputed_test = X_test.dropna().copy()

    elif imputation_method == "IWM":
        # Approach 2: Imputation of Missing Data Using Median and Mode

        # Identify categorical columns
        numerical_features = X_train.select_dtypes(include="number").columns
        boolean_features = numerical_features[
            X_train[numerical_features].nunique() == 2
        ]
        numerical_features = numerical_features.drop(boolean_features)
        categorical_features = X_train.select_dtypes(include="object").columns
        categorical_features = (
            categorical_features.to_list() + boolean_features.to_list()
        )

        # Impute numerical columns with median
        imputer_numerical = SimpleImputer(strategy="median")
        numerical_train = pd.DataFrame(
            imputer_numerical.fit_transform(X_train[numerical_features]),
            columns=numerical_features,
        )

        # Impute categorical columns with most frequent value (mode)
        imputer_categorical = SimpleImputer(strategy="most_frequent")
        categorical_train = pd.DataFrame(
            imputer_categorical.fit_transform(X_train[categorical_features]),
            columns=categorical_features,
        )

        # Combine imputed numerical and categorical data
        imputed_train = pd.concat([numerical_train, categorical_train], axis=1)
        # Restore index
        imputed_train.index = X_train.index

        if X_test is not None:
            numerical_test = pd.DataFrame(
                imputer_numerical.transform(X_test[numerical_features]),
                columns=numerical_features,
            )
            categorical_test = pd.DataFrame(
                imputer_categorical.transform(X_test[categorical_features]),
                columns=categorical_features,
            )
            # Combine imputed numerical and categorical data
            imputed_test = pd.concat(
                [numerical_test, categorical_test], axis=1
            )
            # Restore index
            imputed_test.index = X_test.index

    else:
        raise ValueError('Invalid approach. Choose "DN", "CCA", or "IWM".')

    # Convert data types automatically
    imputed_train = imputed_train.convert_dtypes()
    # Sort columns alphabetically
    imputed_train = sort_columns(imputed_train)

    if X_test is not None:
        # Convert data types automatically
        imputed_test = imputed_test.convert_dtypes()
        # Sort columns alphabetically
        imputed_test = sort_columns(imputed_test)

        return imputed_train, imputed_test
    else:
        return imputed_train


def convert_range_to_min_max(
    df: pd.DataFrame,
    config: Dict[str, Optional[str]],
    drop_column: bool = True,
) -> pd.DataFrame:
    """
    Converts a range column into two separate columns:
    'Minimum' and 'Maximum'.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the range column.
    - config (Dict[str, Optional[str]]): Configuration parameters.
        - range_column (str): The name of the range column.
    - drop_column (bool): Defaults to True,
      in which case the original column is removed from the output.

    Returns:
    - pd.DataFrame: The DataFrame with two additional columns:
      'Minimum' and 'Maximum'.
    """

    column = config.get("range_column")

    # Define regular expression patterns
    pattern_brackets_lower = r"(?:\[(\d+),|>= (\d+)|> (\d+))"
    pattern_brackets_upper = r"(?:,(\d+)\)|<= (\d+)|< (\d+))"

    # Extract lower and upper limits using separate patterns
    df[f"Minimum {column}"] = (
        df[column].str.extract(pattern_brackets_lower).fillna("").sum(axis=1)
    )
    df[f"Maximum {column}"] = (
        df[column].str.extract(pattern_brackets_upper).fillna("").sum(axis=1)
    )

    # Convert the new columns to numeric values
    df[f"Minimum {column}"] = pd.to_numeric(df[f"Minimum {column}"])
    df[f"Maximum {column}"] = pd.to_numeric(df[f"Maximum {column}"])

    if drop_column:
        df.drop(columns=[column], inplace=True)

    return df


def convert_feature_to_boolean(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    """
    Convert a categorical feature to binary indicator columns.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - feature (str): Name of the column to be converted.

    Returns:
    - pd.DataFrame: DataFrame with binary indicator columns added.
    """
    value = df[feature].iloc[0]
    df = df.copy()
    df[f"Is {value.title()}"] = df[feature].eq(value)
    return df.drop(columns=[feature])


def convert_features_to_boolean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert all suitable features in the DataFrame to binary indicator columns.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.

    Returns:
    - pd.DataFrame: DataFrame with binary indicator columns added.
    """
    for feature in df.select_dtypes(exclude="number").columns:
        if (df[feature].notna().all()) and (df[feature].nunique() == 2):
            df = convert_feature_to_boolean(df, feature)
    for feature in df.select_dtypes("number").columns:
        if df[feature].nunique() == 2:
            df[feature] = df[feature].astype("bool")
    return df


def one_hot_encode_feature(
    df: pd.DataFrame, feature: str, drop_column: bool = True
) -> pd.DataFrame:
    """
    One-hot encode a categorical column and
    add the resulting columns to the DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - feature (str): Name of the column to be one-hot encoded.
    - drop_column (bool): Defaults to True,
    in which case the original categorical column is removed from the output.

    Returns:
    - pd.DataFrame: DataFrame with one-hot encoded columns added.
    """

    if drop_column:
        prefix = feature
    else:
        prefix = None

    df = pd.concat([df, pd.get_dummies(df[feature], prefix=prefix)], axis=1)

    if drop_column:
        df.drop(columns=[feature], inplace=True)

    return df


def one_hot_encode_features(
    df: pd.DataFrame, drop_columns: bool = True
) -> pd.DataFrame:
    """
    One-hot encode all suitable columns in the DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - drop_columns (bool): Defaults to True,
    in which case the original categorical columns are
    removed from the output.

    Returns:
    - pd.DataFrame: DataFrame with one-hot encoded columns added.
    """
    for feature in df.select_dtypes(exclude=["number", "bool"]).columns:
        df = one_hot_encode_feature(df, feature, drop_columns)
    return df


def scale_numerical_features(
    df: pd.DataFrame, scaler: Optional[Type[PowerTransformer]] = None
) -> pd.DataFrame:
    """
    Scale numerical features in a DataFrame using RobustScaler.

    Parameters:
        - df (pd.DataFrame): The DataFrame containing numerical features.
        - scaler (Optional[Type[PowerTransformer]]):
          An optional instance of PowerTransformer.
          If provided, it will be used to transform the numerical features.
          If None, a new scaler will be initialized and fitted to the data.

    Returns:
        - pd.DataFrame: The DataFrame with scaled numerical features.
        - Optional[Type[PowerTransformer]]:
          The scaler used for transformation, if scaler is not provided.
    """
    # Select only numerical columns with more than 2 unique values
    numerical_features = df.select_dtypes(include=["number"]).columns
    numerical_features = numerical_features[
        df[numerical_features].nunique() > 2
    ]

    if scaler is None:
        # Initialise RobustScaler
        scaler = PowerTransformer()
        # Fit and transform the numerical data
        df[numerical_features] = scaler.fit_transform(df[numerical_features])
        return df, scaler
    else:
        df[numerical_features] = scaler.transform(df[numerical_features])
        return df


def upsample_minority(X_train, y_train, config):
    categorical_features = X_train.columns[
        X_train.nunique() <= config.get("max_num_categories")
    ].to_list()
    # Apply SMOTE-NC to upsample the minority class in training data
    smotenc = SMOTENC(
        categorical_features=categorical_features,
        random_state=config.get("random_state"),
    )
    X_train, y_train = smotenc.fit_resample(X_train, y_train)
    return X_train, y_train


def process_data(
    df: pd.DataFrame, config: Dict[str, Optional[str]]
) -> pd.DataFrame:
    """
    Perform various data processing operations on the DataFrame
    based on configuration parameters.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - config (Dict[str, Optional[str]]): Configuration parameters.
        - range_column (str): The name of the range column.
        - feature (str): The feature column for thresholding.
        - threshold (float):
          The threshold for creating the binary target column.
        - target_column (str): Name of the target column.

    Returns:
    - pd.DataFrame: Processed DataFrame.
    """
    df = create_target_variable(df, config)
    X_train, X_test, y_train, y_test = split_data(df, config)
    X_train, X_test = handle_missing_data(config, X_train, X_test)
    X_train = convert_range_to_min_max(X_train, config)
    X_test = convert_range_to_min_max(X_test, config)
    X_train = convert_features_to_boolean(X_train)
    X_test = convert_features_to_boolean(X_test)
    X_train, scaler = scale_numerical_features(X_train)
    X_test = scale_numerical_features(X_test, scaler)
    if config.get("imputation_method") != "DN":
        X_train = sort_columns(X_train).dropna()
        y_train = y_train[X_train.index]
        X_test = sort_columns(X_test).dropna()
        y_test = y_test[X_test.index]
    X_train, y_train = upsample_minority(X_train, y_train, config)
    X_train = one_hot_encode_features(X_train)
    X_test = one_hot_encode_features(X_test)
    return X_train, X_test, y_train, y_test
