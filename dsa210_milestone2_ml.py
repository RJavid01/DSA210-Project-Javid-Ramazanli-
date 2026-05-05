# DSA210 Milestone 2 - Machine Learning
# Javid Ramazanli
#
# In milestone 1, I cleaned the Stack Overflow survey data and joined it with
# a cost of living dataset. In this file, I continue from that idea and try
# some machine learning models we learned in class.
#
# My prediction question:
# Can we predict if a developer has a high purchasing-power-adjusted salary?
#
# I am keeping the models simple on purpose:
# - baseline model
# - logistic regression
# - k-nearest neighbors
# - decision tree
# - random forest

from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


RANDOM_STATE = 42
MAX_ROWS_FOR_ML = 5000  # using a sample so the file runs faster on my laptop


def find_my_file(folder, names):
    # I use this because sometimes the files are in the same folder,
    # and sometimes I run the code from a nearby folder.
    for name in names:
        first_try = folder / name
        second_try = folder.parent / name

        if first_try.exists():
            return first_try
        if second_try.exists():
            return second_try

    return None


def read_stackoverflow_data(folder):
    # The Stack Overflow data can be csv, gz, or the official zip file.
    csv_path = find_my_file(folder, ["survey_results_public.csv"])
    gz_path = find_my_file(folder, ["survey_results_public.csv.gz"])
    zip_path = find_my_file(folder, ["stack-overflow-developer-survey-2024.zip"])

    if csv_path is not None:
        print("Reading Stack Overflow csv:", csv_path.name)
        return pd.read_csv(csv_path, low_memory=False)

    if gz_path is not None:
        print("Reading Stack Overflow gz file:", gz_path.name)
        return pd.read_csv(gz_path, low_memory=False)

    if zip_path is not None:
        print("Reading Stack Overflow data from zip:", zip_path.name)
        with zipfile.ZipFile(zip_path) as zip_file:
            survey_file_name = None
            for file_name in zip_file.namelist():
                if file_name.endswith("survey_results_public.csv"):
                    survey_file_name = file_name
                    break

            if survey_file_name is None:
                raise FileNotFoundError("I could not find survey_results_public.csv inside the zip file.")

            with zip_file.open(survey_file_name) as f:
                return pd.read_csv(f, low_memory=False)

    raise FileNotFoundError("I could not find the Stack Overflow survey data file.")


def years_to_number(answer):
    # Stack Overflow writes some year values as text, so I convert them.
    if pd.isna(answer):
        return np.nan
    if answer == "Less than 1 year":
        return 0.5
    if answer == "More than 50 years":
        return 50.0

    try:
        return float(answer)
    except Exception:
        return np.nan


def shorten_education(answer):
    # The original education answers are long. I group them so the model
    # does not get too many messy categories.
    if pd.isna(answer):
        return np.nan

    text = str(answer)
    small_text = text.lower()

    if "doctoral" in small_text or "professional degree" in small_text:
        return "Professional/Doctorate"
    if "Master" in text:
        return "Master"
    if "Bachelor" in text:
        return "Bachelor"
    if "Associate" in text:
        return "Associate"
    if "Some college" in text:
        return "Some college"
    if "Secondary" in text or "Primary" in text:
        return "School or less"

    return "Other"


def clean_and_merge_data(survey_df, cost_df):
    # I keep professional developers only, because the project is about developer earnings.
    df = survey_df[survey_df["MainBranch"] == "I am a developer by profession"].copy()

    # Salary is my main variable, so rows without salary cannot be used here.
    df = df[df["ConvertedCompYearly"].notna()].copy()
    df = df[df["ConvertedCompYearly"] > 0].copy()

    # Some countries have slightly different names in the two datasets.
    country_name_fix = {
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

    df["Country_for_merge"] = df["Country"].replace(country_name_fix)
    df["YearsCodePro_num"] = df["YearsCodePro"].apply(years_to_number)
    df["WorkExp_num"] = pd.to_numeric(df["WorkExp"], errors="coerce")
    df["JobSat_num"] = pd.to_numeric(df["JobSat"], errors="coerce")
    df["EduGroup"] = df["EdLevel"].apply(shorten_education)

    cost_df = cost_df.rename(columns={"Country": "Country_cost"})
    df = pd.merge(df, cost_df, left_on="Country_for_merge", right_on="Country_cost", how="left")

    # I can only adjust salary when the country exists in the cost of living dataset.
    df = df[df["Cost of Living Plus Rent Index"].notna()].copy()

    # This is the main engineered variable of the project.
    df["COL_adjusted_salary"] = df["ConvertedCompYearly"] / (
        df["Cost of Living Plus Rent Index"] / 100
    )

    # Salary data has very large outliers. I remove the top 1% so the models are not
    # mostly learning strange extreme values.
    cutoff = df["COL_adjusted_salary"].quantile(0.99)
    df = df[df["COL_adjusted_salary"] <= cutoff].copy()

    # For classification, I turn adjusted salary into a simple high/low target.
    # Above median is 1, below median is 0.
    median_salary = df["COL_adjusted_salary"].median()
    df["High_Adjusted_Salary"] = (df["COL_adjusted_salary"] >= median_salary).astype(int)

    print("Rows after cleaning:", len(df))
    print("Median adjusted salary:", round(median_salary, 2))
    print("Target counts:")
    print(df["High_Adjusted_Salary"].value_counts())
    print()

    return df


def make_preprocessing(numeric_cols, categorical_cols):
    # Numeric columns: fill missing values with median, then scale.
    # Scaling is important especially for kNN because kNN uses distance.
    numeric_part = Pipeline(
        steps=[
            ("fill_missing", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )

    # Categorical columns: fill missing values with most common value, then one-hot encode.
    try:
        one_hot = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        # older scikit-learn uses sparse instead of sparse_output
        one_hot = OneHotEncoder(handle_unknown="ignore", sparse=True)

    categorical_part = Pipeline(
        steps=[
            ("fill_missing", SimpleImputer(strategy="most_frequent")),
            ("one_hot", one_hot),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_part, numeric_cols),
            ("cat", categorical_part, categorical_cols),
        ]
    )


def train_and_check_model(model_name, model, X_train, X_test, y_train, y_test, output_folder):
    print("\n" + "=" * 55)
    print(model_name)
    print("=" * 55)

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    acc = accuracy_score(y_test, predictions)
    prec = precision_score(y_test, predictions, zero_division=0)
    rec = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)

    print("Accuracy:", round(acc, 4))
    print("Precision:", round(prec, 4))
    print("Recall:", round(rec, 4))
    print("F1-score:", round(f1, 4))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))
    print("Classification report:")
    print(classification_report(y_test, predictions, zero_division=0))

    # Save a confusion matrix picture for the report/GitHub.
    disp = ConfusionMatrixDisplay.from_predictions(y_test, predictions)
    disp.ax_.set_title(model_name + " confusion matrix")
    plt.tight_layout()

    clean_name = model_name.lower().replace(" ", "_").replace("-", "_")
    plt.savefig(output_folder / f"{clean_name}_confusion_matrix.png")
    plt.close()

    return {
        "model": model_name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
    }, model


def get_model_feature_names(preprocessor, numeric_cols):
    # This is only used for Random Forest feature importance.
    feature_names = list(numeric_cols)

    try:
        cat_pipe = preprocessor.named_transformers_["cat"]
        encoder = cat_pipe.named_steps["one_hot"]
        cat_cols = preprocessor.transformers_[1][2]
        feature_names += list(encoder.get_feature_names_out(cat_cols))
    except Exception:
        pass

    return feature_names


def save_rf_importance(fitted_rf_pipeline, output_folder):
    # Random Forest can tell which features were used more often for splitting.
    # I do not treat this as causal, just as model interpretation.
    try:
        preprocessor = fitted_rf_pipeline.named_steps["prep"]
        rf_model = fitted_rf_pipeline.named_steps["model"]
        numeric_cols = preprocessor.transformers_[0][2]
        names = get_model_feature_names(preprocessor, numeric_cols)

        importance_table = pd.DataFrame(
            {
                "feature": names,
                "importance": rf_model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

        importance_table.head(20).to_csv(
            output_folder / "random_forest_top_features.csv", index=False
        )

        top_features = importance_table.head(10).iloc[::-1]
        plt.figure(figsize=(8, 5))
        plt.barh(top_features["feature"], top_features["importance"])
        plt.title("Top Random Forest features")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(output_folder / "random_forest_top_features.png")
        plt.close()

        print("\nTop Random Forest features:")
        print(importance_table.head(10).to_string(index=False))
        print()

    except Exception as error:
        print("I could not save Random Forest feature importance:", error)


def main():
    folder = Path(__file__).resolve().parent
    output_folder = folder / "milestone2_outputs"
    output_folder.mkdir(exist_ok=True)

    survey_df = read_stackoverflow_data(folder)

    cost_path = find_my_file(folder, ["Cost_of_Living_Index_by_Country_2024.csv"])
    if cost_path is None:
        raise FileNotFoundError("Cost_of_Living_Index_by_Country_2024.csv was not found.")

    cost_df = pd.read_csv(cost_path)
    df = clean_and_merge_data(survey_df, cost_df)

    # I picked these features because they are related to experience, work situation,
    # country-level cost, and developer background.
    # I do NOT use ConvertedCompYearly or adjusted salary as input features,
    # because that would leak the answer into the model.
    numeric_cols = [
        "YearsCodePro_num",
        "WorkExp_num",
        "JobSat_num",
        "Cost of Living Index",
        "Rent Index",
        "Cost of Living Plus Rent Index",
        "Local Purchasing Power Index",
    ]

    categorical_cols = [
        "Age",
        "RemoteWork",
        "EduGroup",
        "Employment",
        "DevType",
        "OrgSize",
        "Industry",
        "Country_for_merge",
    ]

    ml_df = df[numeric_cols + categorical_cols + ["High_Adjusted_Salary"]].copy()

    # I use a fixed random sample to keep runtime normal. Since random_state is fixed,
    # the same rows should be chosen each time.
    if len(ml_df) > MAX_ROWS_FOR_ML:
        ml_df = ml_df.sample(n=MAX_ROWS_FOR_ML, random_state=RANDOM_STATE)
        print("Rows used for ML after sampling:", len(ml_df))
        print()

    X = ml_df[numeric_cols + categorical_cols]
    y = ml_df["High_Adjusted_Salary"]

    # Stratify keeps the 0/1 target ratio similar in train and test.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("Training rows:", len(X_train))
    print("Test rows:", len(X_test))
    print()

    # These are models from the course topics. I keep the settings simple.
    models = {
        "Baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "kNN": KNeighborsClassifier(n_neighbors=7),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
    }

    all_results = []
    fitted_models = {}

    for model_name, model in models.items():
        # I put preprocessing and the model into one pipeline. This helps avoid
        # accidentally fitting the scaler/imputer on the test set.
        full_model = Pipeline(
            steps=[
                ("prep", make_preprocessing(numeric_cols, categorical_cols)),
                ("model", model),
            ]
        )

        result, fitted_model = train_and_check_model(
            model_name, full_model, X_train, X_test, y_train, y_test, output_folder
        )
        all_results.append(result)
        fitted_models[model_name] = fitted_model

    results_df = pd.DataFrame(all_results).sort_values("f1", ascending=False)
    results_df.to_csv(output_folder / "milestone2_classification_results.csv", index=False)

    print("\nModel comparison table:")
    print(results_df.to_string(index=False))
    print()

    plt.figure(figsize=(7, 4))
    plt.bar(results_df["model"], results_df["f1"])
    plt.title("Model comparison by F1-score")
    plt.ylabel("F1-score")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(output_folder / "model_f1_comparison.png")
    plt.close()

    if "Random Forest" in fitted_models:
        save_rf_importance(fitted_models["Random Forest"], output_folder)

    print("Milestone 2 code finished.")
    print("Saved outputs are inside:", output_folder)


if __name__ == "__main__":
    main()
