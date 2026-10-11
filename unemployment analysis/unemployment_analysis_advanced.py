
from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(style="whitegrid")

FILES = [
    "Unemployment in India.csv",
    "Unemployment_Rate_upto_11_2020.csv",
]
OUT = Path("output_advanced")
PLOTS = OUT / "plots"
REPORTS = OUT / "reports"
PLOTS.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)


def normalize_columns(df):
    df = df.copy()
    df.columns = [
        re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower())).strip("_")
        for c in df.columns
    ]
    return df


def load_and_clean(filename):
    df = normalize_columns(pd.read_csv(filename))

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip().replace({"nan": np.nan, "": np.nan})

    # Map common Kaggle dataset column names to consistent names.
    rename = {}
    for col in df.columns:
        if col in {"region", "state", "states"}:
            rename[col] = "region"
        elif col in {"date", "month", "date_month"}:
            rename[col] = "date"
        elif "unemployment" in col and "rate" in col:
            rename[col] = "unemployment_rate"
        elif "labour_participation_rate" in col or "labor_participation_rate" in col:
            rename[col] = "labour_participation_rate"
        elif "estimated_employed" in col:
            rename[col] = "estimated_employed"
        elif "frequency" in col:
            rename[col] = "frequency"
    df = df.rename(columns=rename)

    # Convert known metrics to numeric.
    for col in ["unemployment_rate", "labour_participation_rate", "estimated_employed"]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", "", regex=False), errors="coerce"
            )

    if "date" in df.columns:
        # Dataset dates are commonly in DD-MM-YYYY format.
        df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)

    if "region" in df.columns:
        df["region"] = df["region"].astype("string").str.strip()

    # Remove fully duplicated rows and rows without a usable unemployment value.
    df = df.drop_duplicates()
    if "unemployment_rate" in df.columns:
        df = df.dropna(subset=["unemployment_rate"])
        df = df[df["unemployment_rate"].between(0, 100)]

    return df


def save_plot(name):
    plt.tight_layout()
    plt.savefig(PLOTS / name, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {PLOTS / name}")


def phase_label(date):
    if pd.isna(date):
        return np.nan
    if date < pd.Timestamp("2020-03-25"):
        return "Pre-lockdown (Jan–24 Mar)"
    if date <= pd.Timestamp("2020-05-31"):
        return "Strict lockdown (25 Mar–May)"
    return "Reopening (Jun–Nov)"


def covid_phase_analysis(df, label):
    if not {"date", "unemployment_rate"}.issubset(df.columns):
        print("  Skipping COVID phase analysis: date or unemployment-rate column unavailable.")
        return

    work = df.dropna(subset=["date", "unemployment_rate"]).copy()
    work = work[work["date"].dt.year == 2020]
    if work.empty:
        print("  Skipping COVID phase analysis: no valid 2020 dates found.")
        return

    work["covid_phase"] = work["date"].apply(phase_label)
    phase_order = [
        "Pre-lockdown (Jan–24 Mar)",
        "Strict lockdown (25 Mar–May)",
        "Reopening (Jun–Nov)",
    ]
    summary = (
        work.groupby("covid_phase", observed=False)["unemployment_rate"]
        .agg(["mean", "median", "std", "count"])
        .reindex(phase_order)
        .dropna(how="all")
        .round(2)
    )
    summary.to_csv(REPORTS / f"{label}_covid_phase_summary.csv")
    print("\n  COVID phase summary:")
    print(summary.to_string())

    if not summary.empty:
        plt.figure(figsize=(10, 5))
        sns.barplot(
            x=summary.index,
            y=summary["mean"],
            color="slateblue",
        )
        plt.title(f"Average Unemployment by COVID Phase — {label}")
        plt.xlabel("")
        plt.ylabel("Mean unemployment rate (%)")
        plt.xticks(rotation=15, ha="right")
        save_plot(f"{label}_covid_phase_comparison.png")

    monthly = work.set_index("date").resample("MS")["unemployment_rate"].mean().dropna()
    if not monthly.empty:
        plt.figure(figsize=(11, 5))
        plt.plot(monthly.index, monthly.values, marker="o", linewidth=2)
        plt.axvline(pd.Timestamp("2020-03-25"), linestyle="--", label="National lockdown began")
        plt.title(f"Monthly Unemployment Trend — {label}")
        plt.xlabel("Month")
        plt.ylabel("Mean unemployment rate (%)")
        plt.legend()
        plt.xticks(rotation=30)
        save_plot(f"{label}_monthly_trend_lockdown.png")


def state_analysis(df, label):
    if not {"region", "unemployment_rate"}.issubset(df.columns):
        print("  Skipping state analysis: region or unemployment-rate column unavailable.")
        return None

    state = df.groupby("region")["unemployment_rate"].agg(
        mean_unemployment="mean",
        median_unemployment="median",
        unemployment_std="std",
        observations="count",
    )
    if "labour_participation_rate" in df.columns:
        state["mean_labour_participation"] = df.groupby("region")[
            "labour_participation_rate"
        ].mean()
    if "estimated_employed" in df.columns:
        state["mean_estimated_employed"] = df.groupby("region")["estimated_employed"].mean()

    state = state.sort_values("mean_unemployment", ascending=False).round(2)
    state.to_csv(REPORTS / f"{label}_state_summary.csv")
    print("\n  State/region summary (highest average unemployment first):")
    print(state.head(10).to_string())

    plt.figure(figsize=(11, max(6, len(state) * 0.32)))
    ordered = state["mean_unemployment"].sort_values(ascending=True)
    plt.barh(ordered.index.astype(str), ordered.values)
    plt.title(f"Average Unemployment Rate by State/Region — {label}")
    plt.xlabel("Mean unemployment rate (%)")
    plt.ylabel("State / region")
    save_plot(f"{label}_state_ranking.png")

    if "date" in df.columns:
        # Compare early available observations to later observations for each state.
        dated = df.dropna(subset=["date"]).copy()
        if not dated.empty:
            dated["period"] = np.where(
                dated["date"] < pd.Timestamp("2020-06-01"),
                "Jan–May 2020",
                "Jun–Nov 2020",
            )
            pivot = dated.pivot_table(
                index="region", columns="period", values="unemployment_rate", aggfunc="mean"
            )
            if {"Jan–May 2020", "Jun–Nov 2020"}.issubset(pivot.columns):
                pivot["change_later_minus_earlier"] = pivot["Jun–Nov 2020"] - pivot["Jan–May 2020"]
                pivot = pivot.sort_values("change_later_minus_earlier", ascending=False).round(2)
                pivot.to_csv(REPORTS / f"{label}_state_period_change.csv")
                print("\n  States with largest increase from Jan–May to Jun–Nov:")
                print(pivot.head(8).to_string())

    return state


def correlation_analysis(df, label):
    numeric = df.select_dtypes(include=np.number)
    if numeric.shape[1] < 2:
        return
    corr = numeric.corr()
    corr.to_csv(REPORTS / f"{label}_correlation_matrix.csv")
    plt.figure(figsize=(9, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, fmt=".2f")
    plt.title(f"Correlation Heatmap — {label}")
    save_plot(f"{label}_correlation_heatmap.png")


def clustering_analysis(df, label):
    """Group regions by average indicators; cluster labels are exploratory profiles."""
    required = {"region", "unemployment_rate"}
    if not required.issubset(df.columns):
        return

    metrics = ["unemployment_rate"]
    for col in ["labour_participation_rate", "estimated_employed"]:
        if col in df.columns:
            metrics.append(col)

    profiles = df.groupby("region")[metrics].mean().dropna(how="all")
    profiles = profiles.replace([np.inf, -np.inf], np.nan)
    profiles = profiles.dropna(subset=["unemployment_rate"])
    if len(profiles) < 3:
        print("  Skipping clustering: fewer than 3 regions available.")
        return

    n_clusters = min(3, len(profiles))
    pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        KMeans(n_clusters=n_clusters, random_state=42, n_init=10),
    )
    profiles["cluster"] = pipeline.fit_predict(profiles[metrics])
    profiles = profiles.sort_values("cluster")
    profiles.to_csv(REPORTS / f"{label}_state_clusters.csv")
    print("\n  State clusters (cluster IDs are arbitrary, not rankings):")
    print(profiles.to_string())

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=profiles.reset_index(),
        x="labour_participation_rate" if "labour_participation_rate" in profiles.columns else "unemployment_rate",
        y="unemployment_rate",
        hue="cluster",
        palette="Set2",
        s=90,
    )
    plt.title(f"Exploratory State Clusters — {label}")
    plt.xlabel("Labour participation rate (%)" if "labour_participation_rate" in profiles.columns else "Unemployment rate (%)")
    plt.ylabel("Average unemployment rate (%)")
    save_plot(f"{label}_state_clusters.png")


def model_comparison(df, label):
    """Compare simple regressors with a held-out split; avoid date/region leakage."""
    if "unemployment_rate" not in df.columns:
        return

    features = [
        c for c in ["labour_participation_rate", "estimated_employed"]
        if c in df.columns and df[c].notna().sum() > 5
    ]
    model_df = df[features + ["unemployment_rate"]].replace([np.inf, -np.inf], np.nan)
    model_df = model_df.dropna(subset=["unemployment_rate"])
    if not features or len(model_df) < 20:
        print("  Skipping model comparison: not enough usable rows/features.")
        return

    X = model_df[features]
    y = model_df["unemployment_rate"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    models = {
        "Linear Regression": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(), LinearRegression()
        ),
        "Random Forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(n_estimators=200, random_state=42, min_samples_leaf=2),
        ),
    }
    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rows.append({
            "model": name,
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
            "R2": r2_score(y_test, pred),
        })

    results = pd.DataFrame(rows).sort_values("RMSE")
    results.to_csv(REPORTS / f"{label}_model_comparison.csv", index=False)
    print("\n  Exploratory model comparison (lower MAE/RMSE is better):")
    print(results.round(3).to_string(index=False))
    print("  Caution: random splitting of this small observational dataset can overstate generalization.")


def run_one(filename):
    print(f"\n{'=' * 76}\nDATASET: {filename}\n{'=' * 76}")
    df = load_and_clean(filename)
    label = Path(filename).stem.lower().replace(" ", "_")
    print(f"Rows after cleaning: {len(df):,}")
    print("Columns:", ", ".join(df.columns))
    print("\nMissing values:")
    print(df.isna().sum().sort_values(ascending=False).head(12).to_string())

    if "unemployment_rate" not in df.columns:
        print("\nCould not identify the unemployment rate column. Check the CSV headers.")
        return

    print("\nUnemployment rate descriptive statistics:")
    print(df["unemployment_rate"].describe().round(2).to_string())

    df.to_csv(REPORTS / f"{label}_cleaned.csv", index=False)
    covid_phase_analysis(df, label)
    state_analysis(df, label)
    correlation_analysis(df, label)
    clustering_analysis(df, label)
    model_comparison(df, label)


def main():
    found = False
    for filename in FILES:
        if not Path(filename).exists():
            print(f"File not found: {filename}")
            continue
        found = True
        try:
            run_one(filename)
        except Exception as exc:
            print(f"\nCould not fully analyze {filename}: {type(exc).__name__}: {exc}")

    if not found:
        print("\nNo datasets found. Place both CSV files beside this script and run again.")
    else:
        print(f"\nFinished. Review reports in '{REPORTS}' and plots in '{PLOTS}'.")
        print("Interpret all results as exploratory; correlation and clustering do not establish causation.")


if __name__ == "__main__":
    main()
