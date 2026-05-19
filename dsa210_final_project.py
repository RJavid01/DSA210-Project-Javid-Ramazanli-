"""
DSA210 Final Project - Javid Ramazanli

This is my final project code. I use the Stack Overflow Developer Survey 2024
and add a country-level cost-of-living dataset. The main thing I want to check
is whether raw developer salary still tells the same story after I adjust it by
cost of living.

The script does the full project pipeline: cleaning, EDA, hypothesis tests, and
simple machine learning models from the course.
"""

from pathlib import Path
import zipfile
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore", category=FutureWarning)

RANDOM_STATE = 42
MAX_ROWS_FOR_ML = 5000  # I keep this smaller so it can run on a normal laptop.

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "final_outputs"
PLOT_DIR = OUT_DIR / "plots"
TABLE_DIR = OUT_DIR / "tables"


def find_file(names):
    """Look for an input file in the usual places."""
    search_places = [BASE_DIR, BASE_DIR.parent, Path.cwd()]
    for place in search_places:
        for name in names:
            p = place / name
            if p.exists():
                return p
    return None


def load_survey_data():
    """Load the Stack Overflow survey file."""
    csv_file = find_file(["survey_results_public.csv"])
    gz_file = find_file(["survey_results_public.csv.gz"])
    zip_file = find_file(["stack-overflow-developer-survey-2024.zip"])

    if csv_file is not None:
        print("Reading:", csv_file.name)
        return pd.read_csv(csv_file, low_memory=False)

    if gz_file is not None:
        print("Reading:", gz_file.name)
        return pd.read_csv(gz_file, low_memory=False)

    if zip_file is not None:
        print("Reading from zip:", zip_file.name)
        with zipfile.ZipFile(zip_file) as z:
            survey_name = None
            for file_name in z.namelist():
                if file_name.endswith("survey_results_public.csv"):
                    survey_name = file_name
                    break
            if survey_name is None:
                raise FileNotFoundError("survey_results_public.csv was not found inside the zip file.")
            with z.open(survey_name) as f:
                return pd.read_csv(f, low_memory=False)

    raise FileNotFoundError("Could not find Stack Overflow survey_results_public.csv or zip file.")


def load_cost_data():
    """Load the extra cost-of-living dataset."""
    cost_file = find_file(["Cost_of_Living_Index_by_Country_2024.csv"])
    if cost_file is None:
        raise FileNotFoundError("Could not find Cost_of_Living_Index_by_Country_2024.csv")
    print("Reading:", cost_file.name)
    return pd.read_csv(cost_file)


def parse_years(value):
    """Turn experience answers into numbers when possible."""
    if pd.isna(value):
        return np.nan
    if value == "Less than 1 year":
        return 0.5
    if value == "More than 50 years":
        return 50.0
    try:
        return float(value)
    except Exception:
        return np.nan


def simplify_education(value):
    """Make education categories a bit simpler."""
    if pd.isna(value):
        return np.nan
    text = str(value)
    low = text.lower()
    if "professional degree" in low or "doctoral" in low:
        return "Professional/Doctorate"
    if "master" in low:
        return "Master"
    if "bachelor" in low:
        return "Bachelor"
    if "associate" in low:
        return "Associate"
    if "some college" in low:
        return "Some college"
    if "secondary" in low or "primary" in low:
        return "School or less"
    return "Other"


def first_part(value):
    """Keep the first answer in multi-answer columns."""
    if pd.isna(value):
        return np.nan
    return str(value).split(";")[0].strip()


def clean_and_merge_data(survey, cost):
    """Clean the data and merge the two datasets."""

    original_rows = len(survey)

    # I keep professional developers with a positive annual salary.
    data = survey[survey["MainBranch"] == "I am a developer by profession"].copy()
    data = data[data["ConvertedCompYearly"].notna()].copy()
    data = data[data["ConvertedCompYearly"] > 0].copy()
    professional_salary_rows = len(data)

    # Country names are not exactly the same in the two datasets, so I fix common cases.
    country_fix = {
        "Bosnia and Herzegovina": "Bosnia And Herzegovina",
        "Hong Kong (S.A.R.)": "Hong Kong (China)",
        "Iran, Islamic Republic of...": "Iran",
        "Republic of Korea": "South Korea",
        "Republic of Moldova": "Moldova",
        "Republic of North Macedonia": "North Macedonia",
        "Russian Federation": "Russia",
        "Syrian Arab Republic": "Syria",
        "Trinidad and Tobago": "Trinidad And Tobago",
        "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
        "United Republic of Tanzania": "Tanzania",
        "United States of America": "United States",
        "Venezuela, Bolivarian Republic of...": "Venezuela",
        "Viet Nam": "Vietnam",
        "Kosovo": "Kosovo (Disputed Territory)",
        "Libyan Arab Jamahiriya": "Libya",
    }

    data["Country_for_merge"] = data["Country"].replace(country_fix)
    data["YearsCodePro_num"] = data["YearsCodePro"].apply(parse_years)
    data["WorkExp_num"] = pd.to_numeric(data.get("WorkExp"), errors="coerce")
    data["JobSat_num"] = pd.to_numeric(data.get("JobSat"), errors="coerce")
    data["EduGroup"] = data["EdLevel"].apply(simplify_education)
    data["Employment_main"] = data["Employment"].apply(first_part)
    data["DevType_main"] = data["DevType"].apply(first_part)

    data["ExpGroup"] = pd.cut(
        data["YearsCodePro_num"],
        bins=[-1, 2, 5, 10, 20, 100],
        labels=["0-2", "3-5", "6-10", "11-20", "21+"],
    )

    cost = cost.rename(columns={"Country": "Country_cost"})
    merged = pd.merge(data, cost, left_on="Country_for_merge", right_on="Country_cost", how="left")

    matched = merged[merged["Cost of Living Plus Rent Index"].notna()].copy()
    unmatched_rows = len(merged) - len(matched)

    # This adjusted salary is the main new variable in the project.
    matched["COL_adjusted_salary"] = matched["ConvertedCompYearly"] / (
        matched["Cost of Living Plus Rent Index"] / 100.0
    )
    matched["COL_adjusted_salary_no_rent"] = matched["ConvertedCompYearly"] / (
        matched["Cost of Living Index"] / 100.0
    )

    # For plots, I use capped variables so extreme values do not ruin the figures.
    matched["RawSalary_plot"] = matched["ConvertedCompYearly"].clip(
        upper=matched["ConvertedCompYearly"].quantile(0.99)
    )
    matched["AdjustedSalary_plot"] = matched["COL_adjusted_salary"].clip(
        upper=matched["COL_adjusted_salary"].quantile(0.99)
    )

    quick_numbers = {
        "original_rows": original_rows,
        "professional_positive_salary_rows": professional_salary_rows,
        "matched_rows": len(matched),
        "unmatched_rows": unmatched_rows,
        "match_rate": len(matched) / len(merged),
        "job_sat_nonmissing": int(matched["JobSat_num"].notna().sum()),
        "adjusted_salary_median": float(matched["COL_adjusted_salary"].median()),
        "adjusted_salary_mean": float(matched["COL_adjusted_salary"].mean()),
        "raw_salary_median": float(matched["ConvertedCompYearly"].median()),
        "raw_salary_mean": float(matched["ConvertedCompYearly"].mean()),
    }

    return matched, quick_numbers


def save_eda_tables(df, quick_numbers):
    """Save the tables that I use in the report."""
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([quick_numbers]).to_csv(TABLE_DIR / "dataset_quick_numbers.csv", index=False)

    columns_to_save = [
        "ResponseId", "Country", "Country_for_merge", "ConvertedCompYearly",
        "COL_adjusted_salary", "COL_adjusted_salary_no_rent",
        "RemoteWork", "Age", "Employment_main", "DevType_main", "EdLevel", "EduGroup",
        "YearsCodePro", "YearsCodePro_num", "ExpGroup", "WorkExp_num", "JobSat_num",
        "Cost of Living Index", "Rent Index", "Cost of Living Plus Rent Index",
        "Local Purchasing Power Index",
    ]
    df[columns_to_save].to_csv(OUT_DIR / "final_cleaned_dataset.csv", index=False)

    # These are the group tables I use while writing the report.
    for col, name in [
        ("RemoteWork", "remote_work_summary.csv"),
        ("ExpGroup", "experience_group_summary.csv"),
        ("EduGroup", "education_group_summary.csv"),
    ]:
        summary = (
            df.groupby(col, observed=False)["COL_adjusted_salary"]
            .agg(["count", "median", "mean", "std"])
            .reset_index()
        )
        summary.to_csv(TABLE_DIR / name, index=False)

    # Countries with enough observations, so the ranking is not based on 1-2 people.
    country_summary = (
        df.groupby("Country_for_merge")["COL_adjusted_salary"]
        .agg(["count", "median", "mean"])
        .reset_index()
    )
    country_summary = country_summary[country_summary["count"] >= 100].sort_values("median", ascending=False)
    country_summary.to_csv(TABLE_DIR / "country_adjusted_salary_summary_min100.csv", index=False)

    jobsat_summary = (
        df.dropna(subset=["JobSat_num"])
        .groupby("JobSat_num")["COL_adjusted_salary"]
        .agg(["count", "median", "mean"])
        .reset_index()
    )
    jobsat_summary.to_csv(TABLE_DIR / "jobsat_salary_summary.csv", index=False)


def make_eda_plots(df):
    """Make the EDA plots for the report."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # I keep the plot style plain on purpose. It looks more like a normal
    # student report and less like a generated dashboard.
    plt.rcParams.update({
        "figure.dpi": 160,
        "font.size": 10,
        "font.family": "DejaVu Serif",
        "axes.edgecolor": "0.25",
        "axes.labelcolor": "0.10",
        "xtick.color": "0.10",
        "ytick.color": "0.10",
    })

    # 1. Raw salary and adjusted salary. I cap only for plotting, not for the
    # main summary table, because otherwise a few very large salaries dominate
    # the graph.
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.hist(df["RawSalary_plot"], bins=50, alpha=0.75, label="Raw salary", color="0.70", edgecolor="0.40")
    ax.hist(df["AdjustedSalary_plot"], bins=50, alpha=0.60, label="Adjusted salary", color="0.25", edgecolor="0.10")
    ax.set_title("Raw salary and adjusted salary")
    ax.set_xlabel("Salary value, capped at 99th percentile")
    ax.set_ylabel("Number of respondents")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "01_raw_vs_adjusted_salary_distribution.png")
    plt.close(fig)

    # 2. Work setting comparison.
    remote_order = ["Remote", "Hybrid (some remote, some in-person)", "In-person"]
    remote_data = [df.loc[df["RemoteWork"] == x, "AdjustedSalary_plot"].dropna() for x in remote_order]
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.boxplot(
        remote_data,
        tick_labels=["Remote", "Hybrid", "In-person"],
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": "0.85", "edgecolor": "0.25"},
        medianprops={"color": "0.05", "linewidth": 1.4},
        whiskerprops={"color": "0.25"},
        capprops={"color": "0.25"},
    )
    ax.set_yscale("log")
    ax.set_title("Adjusted salary by work setting")
    ax.set_ylabel("Adjusted salary, log scale")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "02_adjusted_salary_by_work_setting.png")
    plt.close(fig)

    # 3. Experience groups.
    exp_order = ["0-2", "3-5", "6-10", "11-20", "21+"]
    exp_data = [df.loc[df["ExpGroup"] == x, "AdjustedSalary_plot"].dropna() for x in exp_order]
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.boxplot(
        exp_data,
        tick_labels=exp_order,
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": "0.85", "edgecolor": "0.25"},
        medianprops={"color": "0.05", "linewidth": 1.4},
        whiskerprops={"color": "0.25"},
        capprops={"color": "0.25"},
    )
    ax.set_yscale("log")
    ax.set_title("Adjusted salary by experience")
    ax.set_xlabel("Years of professional coding experience")
    ax.set_ylabel("Adjusted salary, log scale")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "03_adjusted_salary_by_experience.png")
    plt.close(fig)

    # 4. Education groups.
    edu_order = [
        "School or less", "Some college", "Associate", "Bachelor",
        "Master", "Professional/Doctorate", "Other",
    ]
    edu_data = [df.loc[df["EduGroup"] == x, "AdjustedSalary_plot"].dropna() for x in edu_order]
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.boxplot(
        edu_data,
        tick_labels=["School\nor less", "Some\ncollege", "Associate", "Bachelor", "Master", "Prof/\nDoc", "Other"],
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": "0.85", "edgecolor": "0.25"},
        medianprops={"color": "0.05", "linewidth": 1.4},
        whiskerprops={"color": "0.25"},
        capprops={"color": "0.25"},
    )
    ax.set_yscale("log")
    ax.set_title("Adjusted salary by education group")
    ax.set_ylabel("Adjusted salary, log scale")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "04_adjusted_salary_by_education.png")
    plt.close(fig)

    # 5. Top countries by median adjusted salary, with enough observations.
    country_summary = pd.read_csv(TABLE_DIR / "country_adjusted_salary_summary_min100.csv")
    top_countries = country_summary.head(10).sort_values("median")
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    ax.barh(top_countries["Country_for_merge"], top_countries["median"], color="0.45", edgecolor="0.20")
    ax.set_title("Top countries by median adjusted salary")
    ax.set_xlabel("Median adjusted salary, minimum 100 respondents")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "05_top_countries_adjusted_salary.png")
    plt.close(fig)

    # 6. Job satisfaction relationship.
    jobsat = pd.read_csv(TABLE_DIR / "jobsat_salary_summary.csv")
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.plot(jobsat["JobSat_num"], jobsat["median"], marker="o", color="0.15", linewidth=1.4)
    ax.set_title("Median adjusted salary by job satisfaction")
    ax.set_xlabel("Job satisfaction score")
    ax.set_ylabel("Median adjusted salary")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "06_jobsat_adjusted_salary.png")
    plt.close(fig)

    # 7. Cost of living vs raw salary.
    sample = df.sample(min(3000, len(df)), random_state=RANDOM_STATE)
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.scatter(
        sample["Cost of Living Plus Rent Index"],
        sample["ConvertedCompYearly"].clip(upper=df["ConvertedCompYearly"].quantile(0.99)),
        alpha=0.28,
        s=12,
        color="0.25",
    )
    ax.set_title("Raw salary and cost-of-living index")
    ax.set_xlabel("Cost of Living Plus Rent Index")
    ax.set_ylabel("Raw yearly salary, capped")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "07_cost_index_vs_raw_salary.png")
    plt.close(fig)


def run_hypothesis_tests(df):
    """Run the hypothesis tests used in the report."""
    results = []

    def kruskal_test(group_col, readable_name):
        groups = []
        for _, group in df.groupby(group_col, observed=False):
            vals = group["COL_adjusted_salary"].dropna()
            if len(vals) > 0:
                groups.append(vals)
        stat, p = stats.kruskal(*groups)
        n = sum(len(g) for g in groups)
        k = len(groups)
        eps = (stat - k + 1) / (n - k)
        results.append({
            "test": "Kruskal-Wallis",
            "question": readable_name,
            "statistic": stat,
            "p_value": p,
            "effect_or_note": f"epsilon_sq={eps:.4f}",
            "decision_alpha_0.05": "reject H0" if p <= 0.05 else "fail to reject H0",
        })

    kruskal_test("RemoteWork", "Adjusted salary differs across work settings")
    kruskal_test("ExpGroup", "Adjusted salary differs across experience groups")
    kruskal_test("EduGroup", "Adjusted salary differs across education groups")

    # Job satisfaction is ordinal/numeric-ish, so I use Spearman correlation.
    jobsat_df = df.dropna(subset=["JobSat_num", "COL_adjusted_salary"])
    rho, p = stats.spearmanr(jobsat_df["JobSat_num"], jobsat_df["COL_adjusted_salary"])
    results.append({
        "test": "Spearman correlation",
        "question": "Job satisfaction and adjusted salary are related",
        "statistic": rho,
        "p_value": p,
        "effect_or_note": "rho value",
        "decision_alpha_0.05": "reject H0" if p <= 0.05 else "fail to reject H0",
    })

    # For the chi-square test I turn adjusted salary into high and low groups.
    median_salary = df["COL_adjusted_salary"].median()
    chi_df = df.copy()
    chi_df["High_Adjusted_Salary"] = np.where(chi_df["COL_adjusted_salary"] >= median_salary, "High", "Low")
    table = pd.crosstab(chi_df["RemoteWork"], chi_df["High_Adjusted_Salary"])
    chi2, p, dof, expected = stats.chi2_contingency(table)
    results.append({
        "test": "Chi-square independence",
        "question": "Work setting and high/low adjusted salary are associated",
        "statistic": chi2,
        "p_value": p,
        "effect_or_note": f"dof={dof}",
        "decision_alpha_0.05": "reject H0" if p <= 0.05 else "fail to reject H0",
    })

    tests = pd.DataFrame(results)
    tests.to_csv(TABLE_DIR / "hypothesis_tests.csv", index=False)
    return tests


def make_ml_dataset(df):
    """Prepare the data for the ML part."""
    ml_df = df.copy()

    # I remove the most extreme 1% for ML because self-reported salary has extreme outliers.
    upper_limit = ml_df["COL_adjusted_salary"].quantile(0.99)
    ml_df = ml_df[ml_df["COL_adjusted_salary"] <= upper_limit].copy()

    median_salary = ml_df["COL_adjusted_salary"].median()
    ml_df["High_Adjusted_Salary"] = (ml_df["COL_adjusted_salary"] >= median_salary).astype(int)

    # I sample rows so the script runs quickly on a normal laptop.
    if len(ml_df) > MAX_ROWS_FOR_ML:
        ml_df = ml_df.sample(MAX_ROWS_FOR_ML, random_state=RANDOM_STATE)

    numeric_features = [
        "YearsCodePro_num",
        "WorkExp_num",
        "JobSat_num",
        "Cost of Living Index",
        "Rent Index",
        "Cost of Living Plus Rent Index",
        "Local Purchasing Power Index",
    ]
    categorical_features = [
        "Age",
        "RemoteWork",
        "EduGroup",
        "Employment_main",
        "DevType_main",
        "Country_for_merge",
    ]

    X = ml_df[numeric_features + categorical_features].copy()
    y = ml_df["High_Adjusted_Salary"].copy()

    return X, y, numeric_features, categorical_features, median_salary, ml_df


def build_preprocessor(numeric_features, categorical_features, scale_numeric=True):
    """Make the preprocessing steps for the ML models."""
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(steps=numeric_steps)

    try:
        one_hot = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        one_hot = OneHotEncoder(handle_unknown="ignore", sparse=True)

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", one_hot),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )


def evaluate_model(name, pipeline, X_train, X_test, y_train, y_test):
    """Fit one model and save its results."""
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }

    cm = confusion_matrix(y_test, y_pred)
    pd.DataFrame(cm, index=["actual_low", "actual_high"], columns=["pred_low", "pred_high"]).to_csv(
        TABLE_DIR / f"{name.lower().replace(' ', '_')}_confusion_matrix.csv"
    )

    disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap="Greys")
    disp.ax_.set_title(f"{name} confusion matrix")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"{name.lower().replace(' ', '_')}_confusion_matrix.png")
    plt.close()

    with open(TABLE_DIR / f"{name.lower().replace(' ', '_')}_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(classification_report(y_test, y_pred, zero_division=0))

    return metrics, pipeline


def get_feature_names(preprocessor, numeric_features):
    """Get feature names after one-hot encoding."""
    names = list(numeric_features)
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_names = list(cat_encoder.get_feature_names_out())
    return names + cat_names


def run_ml_models(df):
    """Run the ML models."""
    X, y, numeric_features, categorical_features, median_salary, ml_df = make_ml_dataset(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    scaled_preprocessor = build_preprocessor(numeric_features, categorical_features, scale_numeric=True)
    tree_preprocessor = build_preprocessor(numeric_features, categorical_features, scale_numeric=False)

    models = [
        ("Baseline", Pipeline([("model", DummyClassifier(strategy="most_frequent"))])),
        ("Logistic Regression", Pipeline([
            ("preprocess", scaled_preprocessor),
            ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ])),
        ("kNN", Pipeline([
            ("preprocess", scaled_preprocessor),
            ("model", KNeighborsClassifier(n_neighbors=7)),
        ])),
        ("Decision Tree", Pipeline([
            ("preprocess", tree_preprocessor),
            ("model", DecisionTreeClassifier(max_depth=6, min_samples_leaf=40, random_state=RANDOM_STATE)),
        ])),
        ("Random Forest", Pipeline([
            ("preprocess", tree_preprocessor),
            ("model", RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_leaf=20,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ])),
    ]

    all_metrics = []
    fitted_models = {}
    for name, model in models:
        metrics, fitted = evaluate_model(name, model, X_train, X_test, y_train, y_test)
        all_metrics.append(metrics)
        fitted_models[name] = fitted

    metrics_df = pd.DataFrame(all_metrics).sort_values("f1", ascending=False)
    metrics_df.to_csv(TABLE_DIR / "model_metrics.csv", index=False)

    # Plot metric comparison. I use plain grayscale bars to keep the report simple.
    plot_df = metrics_df.set_index("model")[["accuracy", "precision", "recall", "f1"]]
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    plot_df.plot(kind="bar", ax=ax, color=["0.25", "0.45", "0.65", "0.82"], edgecolor="0.15")
    ax.set_ylim(0, 1)
    ax.set_title("Model performance on test data")
    ax.set_ylabel("Score")
    ax.set_xlabel("Model")
    ax.legend(loc="lower right", frameon=False)
    plt.xticks(rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "08_model_metrics_comparison.png")
    plt.close(fig)

    # Random forest feature importance.
    rf = fitted_models["Random Forest"]
    rf_pre = rf.named_steps["preprocess"]
    rf_model = rf.named_steps["model"]
    feature_names = get_feature_names(rf_pre, numeric_features)
    importance = pd.DataFrame({"feature": feature_names, "importance": rf_model.feature_importances_})
    importance = importance.sort_values("importance", ascending=False)
    importance.to_csv(TABLE_DIR / "random_forest_feature_importance.csv", index=False)

    top = importance.head(10).sort_values("importance")
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.barh(top["feature"], top["importance"], color="0.45", edgecolor="0.20")
    ax.set_title("Top random forest feature importances")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "09_random_forest_feature_importance.png")
    plt.close(fig)

    ml_info = {
        "ml_rows_used": len(ml_df),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "target_median_adjusted_salary": median_salary,
        "target_low_count": int((y == 0).sum()),
        "target_high_count": int((y == 1).sum()),
    }
    pd.DataFrame([ml_info]).to_csv(TABLE_DIR / "ml_quick_numbers.csv", index=False)

    return metrics_df, importance, ml_info


def write_text_summary(quick_numbers, tests, metrics_df, feature_importance, ml_info):
    """Write a small result summary for the report."""
    with open(OUT_DIR / "final_summary.txt", "w", encoding="utf-8") as f:
        f.write("DSA210 Final Project Summary - Javid Ramazanli\n")
        f.write("Project: Purchasing-power-adjusted developer salaries\n\n")
        f.write("Data\n")
        f.write(f"- Original Stack Overflow rows: {quick_numbers['original_rows']}\n")
        f.write(f"- Professional developers with positive salary: {quick_numbers['professional_positive_salary_rows']}\n")
        f.write(f"- Rows matched to cost-of-living data: {quick_numbers['matched_rows']}\n")
        f.write(f"- Match rate: {quick_numbers['match_rate']:.2%}\n")
        f.write(f"- Median raw salary: {quick_numbers['raw_salary_median']:.2f}\n")
        f.write(f"- Median adjusted salary: {quick_numbers['adjusted_salary_median']:.2f}\n\n")

        f.write("Hypothesis tests\n")
        for _, row in tests.iterrows():
            f.write(f"- {row['question']}: {row['test']}, statistic={row['statistic']:.4f}, p={row['p_value']:.4g}, decision={row['decision_alpha_0.05']}\n")
        f.write("\n")

        f.write("Machine learning\n")
        f.write(f"- ML rows used: {ml_info['ml_rows_used']}\n")
        f.write(f"- Train rows: {ml_info['train_rows']}\n")
        f.write(f"- Test rows: {ml_info['test_rows']}\n")
        f.write(f"- Target median adjusted salary: {ml_info['target_median_adjusted_salary']:.2f}\n")
        f.write("\nModel metrics sorted by F1:\n")
        f.write(metrics_df.to_string(index=False))
        f.write("\n\nTop Random Forest features:\n")
        f.write(feature_importance.head(10).to_string(index=False))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    survey = load_survey_data()
    cost = load_cost_data()

    df, quick_numbers = clean_and_merge_data(survey, cost)
    save_eda_tables(df, quick_numbers)
    make_eda_plots(df)

    tests = run_hypothesis_tests(df)
    metrics_df, feature_importance, ml_info = run_ml_models(df)
    write_text_summary(quick_numbers, tests, metrics_df, feature_importance, ml_info)

    print("Finished final project analysis.")
    print("Outputs are saved in:", OUT_DIR)


if __name__ == "__main__":
    main()
