import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def load_data(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)


def inspect_data(df: pd.DataFrame) -> None:
    print("\n===== DATA PREVIEW =====")
    print(df.head())

    print("\n===== DATA TAIL =====")
    print(df.tail())

    print("\n===== DESCRIPTIVE STATISTICS =====")
    print(df.describe())

    print("\n===== CORRELATION MATRIX =====")
    print(df.corr(numeric_only=True))


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    categorical_cols = ["Placement_Offer", "Gender", "Degree"]

    for col in categorical_cols:
        df[col] = df[col].astype("category")

    df = df[df["CGPA"] <= 4.0]
    df = df[df["Salary_Offered_USD"] >= 0]

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df["Total_Skills_Score"] = (
        df["Technical_Skills_Score_100"]
        + df["Communication_Skills_Score_100"]
        + df["Aptitude_Test_Score_100"]
    )

    df["High_CGPA"] = np.where(df["CGPA"] >= 3.5, "Yes", "No")

    return df


def relational_plot(df: pd.DataFrame) -> None:
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
