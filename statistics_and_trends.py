"""
Statistical Analysis of Student Placement Dataset

This script performs structured statistical analysis on a student
placement dataset. It includes data inspection, preprocessing,
feature engineering, visualisation, and statistical calculations
to evaluate employability trends.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load dataset from CSV file.

    Parameters
    ----------
    filepath : str
        Path to the dataset file.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.
    """
    return pd.read_csv(filepath)


def inspect_data(df: pd.DataFrame) -> None:
    """
    Perform dataset inspection.

    Displays dataset preview, tail records,
    descriptive statistics, and correlation matrix.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    """
    print("\n===== DATA PREVIEW =====")
    print(df.head())

    print("\n===== DATA TAIL =====")
    print(df.tail())

    print("\n===== DESCRIPTIVE STATISTICS =====")
    print(df.describe())

    print("\n===== CORRELATION MATRIX =====")
    print(df.corr(numeric_only=True))


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean dataset and correct data types.

    Converts categorical columns and removes
    invalid CGPA and salary values.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset.

    Returns
    -------
    pd.DataFrame
        Cleaned dataset.
    """
    categorical_cols = ["Placement_Offer", "Gender", "Degree"]

    for col in categorical_cols:
        df[col] = df[col].astype("category")

    df = df[df["CGPA"] <= 4.0]
    df = df[df["Salary_Offered_USD"] >= 0]

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived analytical features.

    Adds total skills score and CGPA classification.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset.

    Returns
    -------
    pd.DataFrame
        Enhanced dataset.
    """
    df["Total_Skills_Score"] = (
        df["Technical_Skills_Score_100"]
        + df["Communication_Skills_Score_100"]
        + df["Aptitude_Test_Score_100"]
    )

    df["High_CGPA"] = np.where(df["CGPA"] >= 3.5, "Yes", "No")

    return df


def relational_plot(df: pd.DataFrame) -> None:
    """
    Generate relational scatter plot.

    Visualises CGPA vs Salary relationship.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    """
    plt.figure(figsize=(8, 5))

    sns.scatterplot(
        data=df,
        x="CGPA",
        y="Salary_Offered_USD",
        hue="Placement_Offer",
    )

    plt.title("CGPA vs Salary Offered")
    plt.xlabel("CGPA")
    plt.ylabel("Salary (USD)")
    plt.show()


def categorical_plot(df: pd.DataFrame) -> None:
    """
    Generate categorical placement distribution plot.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    """
    plt.figure(figsize=(8, 5))

    sns.countplot(
        data=df,
        x="Degree",
        hue="Placement_Offer",
    )

    plt.title("Placement Distribution by Degree")
    plt.xticks(rotation=45)
    plt.show()


def statistical_plot(df: pd.DataFrame) -> None:
    """
    Generate statistical heatmap.

    Displays correlation between numerical variables.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    """
    plt.figure(figsize=(10, 6))

    corr = df.corr(numeric_only=True)

    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
    )

    plt.title("Correlation Heatmap")
    plt.show()


def statistical_analysis(df: pd.DataFrame) -> None:
    """
    Compute statistical moments.

    Includes mean, standard deviation,
    skewness, and kurtosis.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    """
    metrics = df[
        [
            "CGPA",
            "Technical_Skills_Score_100",
            "Communication_Skills_Score_100",
            "Aptitude_Test_Score_100",
            "Salary_Offered_USD",
        ]
    ]

    print("\n===== MEAN =====")
    print(metrics.mean())

    print("\n===== STANDARD DEVIATION =====")
    print(metrics.std())

    print("\n===== SKEWNESS =====")
    print(metrics.skew())

    print("\n===== KURTOSIS =====")
    print(metrics.kurtosis())


def main() -> None:
    """
    Execute full statistical workflow.
    """
    sns.set(style="whitegrid")

    filepath = "./data.csv"

    df = load_data(filepath)
    inspect_data(df)
    df = clean_data(df)
    df = engineer_features(df)
    relational_plot(df)
    categorical_plot(df)
    statistical_plot(df)
    statistical_analysis(df)

    print("\nAnalysis Completed Successfully.")


if __name__ == "__main__":
    main()
