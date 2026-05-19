# DSA210 Project - Javid Ramazanli

This is my DSA210 term project.

The project uses the Stack Overflow Developer Survey 2024 together with a country-level cost-of-living dataset. I wanted to check developer salaries in a more realistic way, because raw salary alone can be misleading when countries have very different living costs.

## Main idea

I first clean the survey data, then merge it with the cost-of-living data by country. After that, I create an adjusted salary variable. This is the main variable I use in the project.

The project includes:

- data cleaning
- exploratory data analysis
- hypothesis tests
- machine learning models
- final plots and result tables

## Data files needed

The code expects these files to be in this folder or one folder above it:

- `survey_results_public.csv` or `stack-overflow-developer-survey-2024.zip`
- `Cost_of_Living_Index_by_Country_2024.csv`

The Stack Overflow survey is the main dataset. The cost-of-living file is the extra dataset I use to enrich it.

## Final code file

The final script is:

```text
 dsa210_final_project.py
```

When it runs, it creates the folder:

```text
final_outputs/
```

Inside that folder, it saves the cleaned dataset, plots, hypothesis test results, model metrics, confusion matrices, and random forest feature importance.

## How to run the project

Install the packages:

```bash
pip install -r requirements.txt
```

Then run:

```bash
python dsa210_final_project.py
```

## Models used

For the machine learning part, I used models that match the course topics:

- baseline classifier
- logistic regression
- kNN
- decision tree
- random forest

The target is whether a developer is in the high or low adjusted salary group compared with the median.

## Note

This project uses observational survey data. Because of that, I treat the results as associations and predictions. I do not claim that one variable directly causes higher salary.
