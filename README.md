# DSA210 Project - Javid Ramazanli

This repository is for my DSA210 term project.

The project uses the Stack Overflow Developer Survey 2024 and a country-level cost of living dataset. My main idea is to look at developer salaries after adjusting them by cost of living, instead of only looking at raw yearly salary.

## Milestone 2

For this milestone, I added machine learning methods to the project.

The task I used is a simple classification task:

Can the model predict whether a developer has a high purchasing-power-adjusted salary or not?

I created the target like this:

- `High_Adjusted_Salary = 1` means the adjusted salary is above or equal to the median
- `High_Adjusted_Salary = 0` means the adjusted salary is below the median

I used this target because it fits the classification models we learned in class.

## Models I used

- Baseline classifier
- Logistic Regression
- k-Nearest Neighbors
- Decision Tree
- Random Forest

I used train/test split and evaluated the models on the test data. The code prints accuracy, precision, recall, F1-score, and confusion matrix.

## Main files

- `dsa210_stage1_analysis.py` - milestone 1 code for data cleaning, EDA, and hypothesis tests
- `dsa210_milestone2_ml.py` - milestone 2 code for machine learning
- `stack-overflow-developer-survey-2024.zip` - Stack Overflow Developer Survey 2024 dataset
- `Cost_of_Living_Index_by_Country_2024.csv` - cost of living dataset
- `requirements.txt` - Python packages used in the project

## How to run

Keep the code file and the two dataset files in the same folder. Then run:

```bash
python dsa210_milestone2_ml.py
```

The code creates a folder called `milestone2_outputs`. The model results and plots are saved there.

## Small note

These models are for prediction. I do not claim that a feature causes higher salary. I only interpret the results as patterns/associations in this dataset.
