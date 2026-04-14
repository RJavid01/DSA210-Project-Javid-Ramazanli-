from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kruskal, spearmanr

# file names
base = Path(__file__).resolve().parent
survey_file = base / "survey_results_public.csv"
cost_file = base / "Cost_of_Living_Index_by_Country_2024.csv"

# if the files are not in the same folder, try one folder up
if not survey_file.exists():
    survey_file = base.parent / "survey_results_public.csv"
if not cost_file.exists():
    cost_file = base.parent / "Cost_of_Living_Index_by_Country_2024.csv"

# read the files
survey = pd.read_csv(survey_file, low_memory=False)
cost = pd.read_csv(cost_file)

# I only want professional developers and positive yearly salary
survey = survey[survey["MainBranch"] == "I am a developer by profession"]
survey = survey[survey["ConvertedCompYearly"].notna()]
survey = survey[survey["ConvertedCompYearly"] > 0].copy()

# some country names are written differently in the two files
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
    "Libyan Arab Jamahiriya": "Libya"
}

new_country_list = []
for c in survey["Country"]:
    if c in country_fix:
        new_country_list.append(country_fix[c])
    else:
        new_country_list.append(c)

survey["Country_for_merge"] = new_country_list

# years of professional coding is text, so I turn it into numbers
new_years = []
for y in survey["YearsCodePro"]:
    if pd.isna(y):
        new_years.append(np.nan)
    elif y == "Less than 1 year":
        new_years.append(0.5)
    elif y == "More than 50 years":
        new_years.append(50.0)
    else:
        try:
            new_years.append(float(y))
        except:
            new_years.append(np.nan)

survey["YearsCodePro_num"] = new_years

# make experience groups
exp_group_list = []
for y in survey["YearsCodePro_num"]:
    if pd.isna(y):
        exp_group_list.append(np.nan)
    elif y <= 2:
        exp_group_list.append("0-2")
    elif y <= 5:
        exp_group_list.append("3-5")
    elif y <= 10:
        exp_group_list.append("6-10")
    elif y <= 20:
        exp_group_list.append("11-20")
    else:
        exp_group_list.append("21+")

survey["ExpGroup"] = exp_group_list

# simpler education groups
edu_list = []
for e in survey["EdLevel"]:
    if pd.isna(e):
        edu_list.append(np.nan)
    else:
        e = str(e)
        if "Professional degree" in e:
            edu_list.append("Professional/Doctorate")
        elif "Master" in e:
            edu_list.append("Master")
        elif "Bachelor" in e:
            edu_list.append("Bachelor")
        elif "Associate" in e:
            edu_list.append("Associate")
        elif "Some college" in e:
            edu_list.append("Some college")
        elif "Secondary" in e or "Primary" in e:
            edu_list.append("School or less")
        else:
            edu_list.append("Other")

survey["EduGroup"] = edu_list
survey["JobSat_num"] = pd.to_numeric(survey["JobSat"], errors="coerce")

# merge the two datasets by country
cost = cost.rename(columns={"Country": "Country_cost"})
merged = pd.merge(survey, cost, left_on="Country_for_merge", right_on="Country_cost", how="left")

# keep only matched rows for this stage
merged = merged[merged["Cost of Living Plus Rent Index"].notna()].copy()

# main new variable for my project
merged["COL_adjusted_salary"] = merged["ConvertedCompYearly"] / (merged["Cost of Living Plus Rent Index"] / 100)

# save cleaned file
keep_cols = [
    "ResponseId", "Country", "Country_for_merge", "Currency", "ConvertedCompYearly",
    "RemoteWork", "EdLevel", "EduGroup", "YearsCodePro", "YearsCodePro_num",
    "ExpGroup", "Employment", "DevType", "JobSat", "JobSat_num",
    "Cost of Living Index", "Rent Index", "Cost of Living Plus Rent Index",
    "Local Purchasing Power Index", "COL_adjusted_salary"
]
merged[keep_cols].to_csv("dsa210_stage1_analysis_student_cleaned.csv", index=False)

# to make plots look better, I clip very extreme values
plot_salary = merged["COL_adjusted_salary"].clip(upper=merged["COL_adjusted_salary"].quantile(0.99))

plt.figure(figsize=(7, 4.5))
plt.hist(plot_salary, bins=50)
plt.title("Adjusted salary distribution")
plt.xlabel("Adjusted salary")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("student_plot_salary_hist.png")
plt.close()

remote_order = ["Remote", "Hybrid (some remote, some in-person)", "In-person"]
remote_data = []
remote_labels = []
for r in remote_order:
    vals = merged.loc[merged["RemoteWork"] == r, "COL_adjusted_salary"].dropna()
    if len(vals) > 0:
        remote_data.append(vals.clip(upper=vals.quantile(0.99)))
        if r == "Hybrid (some remote, some in-person)":
            remote_labels.append("Hybrid")
        else:
            remote_labels.append(r)

plt.figure(figsize=(7, 4.5))
plt.boxplot(remote_data, tick_labels=remote_labels, showfliers=False)
plt.yscale("log")
plt.title("Adjusted salary by work setting")
plt.ylabel("Adjusted salary (log scale)")
plt.tight_layout()
plt.savefig("student_plot_remote_box.png")
plt.close()

exp_order = ["0-2", "3-5", "6-10", "11-20", "21+"]
exp_data = []
exp_labels = []
for g in exp_order:
    vals = merged.loc[merged["ExpGroup"] == g, "COL_adjusted_salary"].dropna()
    if len(vals) > 0:
        exp_data.append(vals.clip(upper=vals.quantile(0.99)))
        exp_labels.append(g)

plt.figure(figsize=(7, 4.5))
plt.boxplot(exp_data, tick_labels=exp_labels, showfliers=False)
plt.yscale("log")
plt.title("Adjusted salary by experience group")
plt.xlabel("Years of professional coding")
plt.ylabel("Adjusted salary (log scale)")
plt.tight_layout()
plt.savefig("student_plot_experience_box.png")
plt.close()

# hypothesis test 1: remote work groups
remote_test_data = []
for r in remote_order:
    vals = merged.loc[merged["RemoteWork"] == r, "COL_adjusted_salary"].dropna()
    if len(vals) > 0:
        remote_test_data.append(vals)

remote_h, remote_p = kruskal(*remote_test_data)

# hypothesis test 2: experience groups
exp_test_data = []
for g in exp_order:
    vals = merged.loc[merged["ExpGroup"] == g, "COL_adjusted_salary"].dropna()
    if len(vals) > 0:
        exp_test_data.append(vals)

exp_h, exp_p = kruskal(*exp_test_data)

# hypothesis test 3: education groups
edu_order = ["School or less", "Some college", "Associate", "Bachelor", "Master", "Professional/Doctorate", "Other"]
edu_test_data = []
for g in edu_order:
    vals = merged.loc[merged["EduGroup"] == g, "COL_adjusted_salary"].dropna()
    if len(vals) > 0:
        edu_test_data.append(vals)

edu_h, edu_p = kruskal(*edu_test_data)

# hypothesis test 4: job satisfaction and salary
job_df = merged[["JobSat_num", "COL_adjusted_salary"]].dropna()
rho, rho_p = spearmanr(job_df["JobSat_num"], job_df["COL_adjusted_salary"])

# print short results
print("Rows after my filtering:", len(merged))
print()
print("H1: work setting vs adjusted salary")
print("Kruskal H =", round(remote_h, 4))
print("p-value =", remote_p)
print()
print("H2: experience vs adjusted salary")
print("Kruskal H =", round(exp_h, 4))
print("p-value =", exp_p)
print()
print("H3: education vs adjusted salary")
print("Kruskal H =", round(edu_h, 4))
print("p-value =", edu_p)
print()
print("H4: job satisfaction vs adjusted salary")
print("Spearman rho =", round(rho, 4))
print("p-value =", rho_p)
