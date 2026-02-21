
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    return df


def inspect_data(df: pd.DataFrame) -> None:
    print("Dataset Preview:")
    print(df.head())

    print("\nDataset Shape:", df.shape)

    print("\nDataset Info:")
    print(df.info())

    print("\nMissing Values in Each Column:")
    print(df.isnull().sum())


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df["Placement_Offer"] = df["Placement_Offer"].astype("category")
    df["Gender"] = df["Gender"].astype("category")
    df["Degree"] = df["Degree"].astype("category")

    df = df[df["CGPA"] <= 4.0]
    df = df[df["Salary_Offered_USD"] >= 0]

    print("\nData Cleaning Completed.")
    print("New Dataset Shape:", df.shape)

    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df["Total_Skills_Score"] = (
        df["Technical_Skills_Score_100"]
        + df["Communication_Skills_Score_100"]
        + df["Aptitude_Test_Score_100"]
    )

    df["High_CGPA"] = np.where(df["CGPA"] >= 3.5, "Yes", "No")

    print("\nFeature Engineering Completed.")
    print(df.head())

    return df


def create_visualisations(df: pd.DataFrame) -> None:
    sns.set(style="whitegrid")

    # CGPA vs Salary
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=df,
        x="CGPA",
        y="Salary_Offered_USD",
        hue="Placement_Offer",
    )
    plt.title("Relationship Between CGPA and Salary Offered")
    plt.xlabel("CGPA")
    plt.ylabel("Salary Offered (USD)")
    plt.show()

    # Placement by Degree
    plt.figure(figsize=(8, 5))
    sns.countplot(
        data=df,
        x="Degree",
        hue="Placement_Offer",
    )
    plt.title("Placement Distribution by Degree")
    plt.xticks(rotation=45)
    plt.show()

    # Correlation Heatmap
    plt.figure(figsize=(10, 6))
    corr = df.corr(numeric_only=True)

    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
    )
    plt.title("Correlation Heatmap of Student Metrics")
    plt.show()

    # Skills vs Salary
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=df,
        x="Total_Skills_Score",
        y="Salary_Offered_USD",
    )
    plt.title("Total Skills Score vs Salary")
    plt.savefig("skills_vs_salary.png")
    plt.close()


def statistical_moment_analysis(df: pd.DataFrame) -> None:
    metrics = df[
        [
            "CGPA",
            "Technical_Skills_Score_100",
            "Communication_Skills_Score_100",
            "Aptitude_Test_Score_100",
            "Salary_Offered_USD",
        ]
    ]

    print("\n==========")
    print("STATISTICAL MOMENTS ANALYSIS")
    print("==========")

    print("\nMEAN:")
    print(metrics.mean())

    print("\nVARIANCE:")
    print(metrics.var())

    print("\nSKEWNESS:")
    print(metrics.skew())

    print("\nKURTOSIS:")
    print(metrics.kurtosis())


def main() -> None:
    filepath = "./data.csv"

    df = load_data(filepath)
    inspect_data(df)
    df = clean_data(df)
    df = feature_engineering(df)
    create_visualisations(df)
    statistical_moment_analysis(df)

    print("\nAnalysis Completed Successfully.")


if __name__ == "__main__":
    main()
