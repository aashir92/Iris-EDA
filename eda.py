"""
Iris EDA Script
---------------
Loads, inspects, and visualizes the Iris dataset using seaborn, pandas, and matplotlib.

Requirements:
- pandas
- matplotlib
- seaborn
"""

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def print_section_header(title: str) -> None:
    """Print a clear section header for console output."""
    print(f"\n{'=' * 70}")
    print(f"{title}")
    print(f"{'=' * 70}")


def save_figure(filename: str) -> None:
    """Save the current matplotlib figure to the script directory."""
    output_path = Path(__file__).resolve().parent / filename
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot: {output_path}")


def load_iris_dataset() -> pd.DataFrame:
    """
    Load Iris dataset from seaborn with error handling.

    Returns:
        pd.DataFrame: Loaded Iris dataset.

    Raises:
        RuntimeError: If dataset cannot be loaded.
    """
    try:
        df_local = sns.load_dataset("iris")
        if df_local is None or df_local.empty:
            raise RuntimeError("Loaded dataset is empty.")
        return df_local
    except Exception as exc:
        raise RuntimeError(
            "Failed to load Iris dataset via seaborn. "
            "Check internet/library availability and try again."
        ) from exc


def inspect_data(df_local: pd.DataFrame) -> None:
    """Print shape, columns, head, info, and descriptive statistics."""
    print_section_header("STEP 2: DATA INSPECTION & SUMMARY STATISTICS")

    rows, cols = df_local.shape
    print(f"Dataset shape: {rows} rows x {cols} columns")
    print(f"Column names: {df_local.columns.tolist()}")

    print("\nFirst 5 rows:")
    print(df_local.head())

    print("\nTechnical summary (.info()):")
    print("-" * 70)
    df_local.info()
    print("-" * 70)

    print("\nDescriptive statistics (numerical columns):")
    print(df_local.describe())


def plot_scatter_and_pairplot(df_local: pd.DataFrame) -> None:
    """Create scatter plot and pairplot for feature relationships."""
    print_section_header("STEP 3A: RELATIONSHIPS (SCATTER + PAIRPLOT)")

    # Scatter plot: sepal_length vs sepal_width with species hue
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df_local,
        x="sepal_length",
        y="sepal_width",
        hue="species",
        s=80,
        alpha=0.8,
    )
    plt.title("Iris Sepal Length vs Sepal Width by Species")
    plt.xlabel("Sepal Length (cm)")
    plt.ylabel("Sepal Width (cm)")
    plt.tight_layout()
    save_figure("iris_scatter_sepal_length_vs_width.png")
    plt.show()

    # Pairplot for all numeric relationships by species
    pair_plot = sns.pairplot(
        df_local,
        hue="species",
        diag_kind="hist",
        corner=False,
        plot_kws={"alpha": 0.75, "s": 40},
    )
    pair_plot.fig.set_size_inches(12, 10)
    pair_plot.fig.suptitle(
        "Pairwise Feature Relationships Across Iris Species",
        y=1.02,
    )
    plt.tight_layout()
    pair_plot.fig.savefig(
        Path(__file__).resolve().parent / "iris_pairplot_all_features.png",
        dpi=300,
        bbox_inches="tight",
    )
    print(
        "Saved plot: "
        f"{Path(__file__).resolve().parent / 'iris_pairplot_all_features.png'}"
    )
    plt.show()


def plot_histograms(df_local: pd.DataFrame) -> None:
    """Create 2x2 histograms with KDE overlays for numeric features."""
    print_section_header("STEP 3B: DISTRIBUTIONS (HISTOGRAMS + KDE)")

    numeric_features = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, feature in enumerate(numeric_features):
        sns.histplot(
            data=df_local,
            x=feature,
            kde=True,
            ax=axes[idx],
            bins=15,
            color="steelblue",
            edgecolor="black",
        )
        axes[idx].set_title(f"Distribution of {feature.replace('_', ' ').title()}")
        axes[idx].set_xlabel(f"{feature.replace('_', ' ').title()} (cm)")
        axes[idx].set_ylabel("Frequency")

    plt.tight_layout()
    fig.savefig(
        Path(__file__).resolve().parent / "iris_histograms_with_kde.png",
        dpi=300,
        bbox_inches="tight",
    )
    print(
        "Saved plot: "
        f"{Path(__file__).resolve().parent / 'iris_histograms_with_kde.png'}"
    )
    plt.show()


def plot_boxplots(df_local: pd.DataFrame) -> None:
    """Create 2x2 boxplots by species for outlier inspection."""
    print_section_header("STEP 3C: OUTLIERS (BOXPLOTS BY SPECIES)")

    numeric_features = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, feature in enumerate(numeric_features):
        sns.boxplot(
            data=df_local,
            x="species",
            y=feature,
            ax=axes[idx],
            palette="Set2",
        )
        axes[idx].set_title(
            f"{feature.replace('_', ' ').title()} by Species (Outlier Check)"
        )
        axes[idx].set_xlabel("Species")
        axes[idx].set_ylabel(f"{feature.replace('_', ' ').title()} (cm)")

    plt.tight_layout()
    fig.savefig(
        Path(__file__).resolve().parent / "iris_boxplots_by_species.png",
        dpi=300,
        bbox_inches="tight",
    )
    print(
        "Saved plot: "
        f"{Path(__file__).resolve().parent / 'iris_boxplots_by_species.png'}"
    )
    plt.show()


def main() -> None:
    """Run full Iris EDA workflow."""
    # Set cohesive seaborn theme for all plots.
    sns.set_theme(style="whitegrid")

    print_section_header("STEP 1: SETUP & DATA LOADING")
    try:
        df = load_iris_dataset()
        print("Iris dataset loaded successfully.")
    except RuntimeError as err:
        print(f"Error: {err}")
        sys.exit(1)

    inspect_data(df)
    plot_scatter_and_pairplot(df)
    plot_histograms(df)
    plot_boxplots(df)

    print_section_header("EDA COMPLETED")
    print("All inspections and visualizations were generated successfully.")


if __name__ == "__main__":
    main()