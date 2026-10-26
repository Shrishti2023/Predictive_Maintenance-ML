# Predictive Maintenance ML — Machine Failure Prediction
This project explores machine-failure prediction using operational data from the AI4I 2020 Predictive Maintenance Dataset. The work is implemented in a Jupyter Notebook and uses exploratory data analysis, two classification models, model evaluation, and model-interpretation tools.

# Project Objective
The objective is to train machine-learning classifiers using machine operating conditions and predict the Machine failure label. The notebook also creates a Failure Type label from the dataset's existing failure indicators and runs a separate classification experiment with that label.
# Dataset
The notebook loads a CSV file named ai4i2020.csv. The dataset shown in the notebook contains 10,000 records and these original columns:

| Column | Description |
|---|---|
| `UDI` | Record identifier |
| `Product ID` | Product identifier |
| `Type` | Product type (`L`, `M`, or `H`) |
| `Air temperature [K]` | Air temperature in Kelvin |
| `Process temperature [K]` | Process temperature in Kelvin |
| `Rotational speed [rpm]` | Rotational speed in revolutions per minute |
| `Torque [Nm]` | Torque in Newton-metres |
| `Tool wear [min]` | Tool wear in minutes |
| `Machine failure` | Binary machine-failure label |
| `TWF` | Tool Wear Failure indicator |
| `HDF` | Heat Dissipation Failure indicator |
| `PWF` | Power Failure indicator |
| `OSF` | Overstrain Failure indicator |
| `RNF` | Random Failure indicator |

# Features used for classification
The notebook uses these six input features:
1. Type
2. Air temperature [K]
3. Process temperature [K]
4. Rotational speed [rpm]
5. Torque [Nm]
6. Tool wear [min]

The Type column is converted to numeric values:
- L → 0
- M → 1
- H → 2

The UDI and Product ID columns are not included in the selected model features.

# Failure type label
The notebook creates a Failure Type column by checking the indicators TWF, HDF, PWF, OSF, and RNF in that order. It returns the first indicator whose value is 1; if none is 1, it returns No Failure.

Because the function returns only the first matching indicator, this derived label represents one category per row and does not preserve multiple simultaneous failure indicators as separate labels.

# Workflow
1. Load and inspect the data
The notebook uses Pandas to load the CSV and inspect sample rows, column names, data types, descriptive statistics, and missing values.
2. Exploratory data analysis
The notebook uses Pandas, Matplotlib, and Seaborn for:
- Histograms of numeric columns
- Counts of the Machine failure label
- Counts of the derived Failure Type label
- Counts of product types
- A relationship plot of air temperature and process temperature by failure type
- A scatter plot of air temperature and process temperature by machine-failure label
- Boxplots for air temperature, process temperature, rotational speed, torque, and tool wear
The notebook's written observation notes that torque and rotational speed have more points outside their usual boxplot quartile ranges.

The saved class-count output shows:
- No machine failure (0): 9,661 records
- Machine failure (1): 339 records
This means the binary target is imbalanced, with many more normal records than failure records.
3. Prepare features and labels
The notebook maps product types to numbers, selects the six features listed above, and sets Machine failure as the target for the first classification experiment.
4. Split the data
The notebook uses train_test_split with:
- Test set size: 0.25 (25%)
- random_state=42
For 10,000 records, this gives 7,500 training records and 2,500 test records.

The split does not specify stratification.
5. Train the classifiers
The notebook trains these scikit-learn classifiers:
- Random Forest Classifier, configured with n_estimators=100 and random_state=42
- Logistic Regression, using the default constructor settings
A helper function fits each classifier and prints its score on the test set.

6. Evaluate predictions
For the binary Machine failure experiment, the notebook generates Random Forest predictions and displays:
- A classification report
- A confusion matrix plot
It also plots a confusion matrix after fitting Logistic Regression, but the notebook reuses the earlier Random Forest prediction variable for that plot. Therefore, that second confusion-matrix plot should not be interpreted as a valid Logistic Regression evaluation.
7. Interpret the model
The notebook imports ELI5 and uses it to:
- Display Logistic Regression model weights
- Calculate permutation importance on the test data
- Explain an individual prediction
The notebook also creates a sample input with six values and attempts to explain its prediction.
# Results Recorded in the Notebook
The following figures are taken from the notebook's saved outputs.
Binary machine-failure classification

| Model | Test accuracy |
|---|---:|
| Random Forest | 98.64% |
| Logistic Regression | 97.52% |
The Logistic Regression output includes a convergence warning: the optimizer reached its iteration limit. The notebook does not apply feature scaling or increase the iteration limit.
Random Forest classification report
| Metric | No failure (`0`) | Failure (`1`) |
|---|---:|---:|
| Precision | 0.99 | 0.88 |
| Recall | 1.00 | 0.61 |
| F1-score | 0.99 | 0.72 |
| Support | 2,428 | 72 |

Additional saved metrics:
| Metric | Value |
|---|---:|
| Accuracy | 98.64% |
| Macro-average F1-score | 0.86 |
| Weighted-average F1-score | 0.99 |

Although the Random Forest accuracy is high, its failure-class recall is 0.61. In the test set, the model therefore identifies about 61% of actual failure cases and misses some failures. Accuracy alone is not enough to assess a predictive-maintenance model, especially when failure cases are relatively rare.

# Failure-type classification experiment
Later in the notebook, the target is changed from Machine failure to the derived Failure Type label, and the classifiers are trained again.

| Model | Test accuracy |
|---|---:|
| Random Forest | 98.36% |
| Logistic Regression | 97.32% |

These results concern the derived Failure Type target, not the binary machine-failure target. The notebook does not show a classification report for this second experiment, so per-class precision, recall, and F1-score are not reported here.
All metrics above are saved notebook outputs and have not been independently reproduced.

# Technologies and Libraries
The notebook uses:
- Python — programming language
- Pandas — data loading and tabular analysis
- NumPy — numerical operations and sample-array preparation
- Matplotlib — plots and visualizations
- Seaborn — statistical visualizations
- scikit-learn — train/test split, Logistic Regression, Random Forest, classification report, and confusion matrix
- ELI5 — model weights, permutation importance, and prediction explanations
- Jupyter Notebook — interactive environment used to run the analysis

# Project Files
The files supplied for this task include:
predictive_maintenance.ipynb
Pasted text.txt
The notebook expects a separate dataset file named ai4i2020.csv. That CSV was not included among the two supplied files, so it must be available locally before the notebook can run from the beginning.

# Setup and Run
1. Install the dependencies
Use Python with these packages installed:

pip install pandas numpy matplotlib seaborn scikit-learn eli5 jupyter

3. Add the dataset
Place ai4i2020.csv in the location you intend to use and update the CSV path in the notebook.

The current notebook uses this machine-specific Windows path:

df = pd.read_csv(r"D:\predictive_maintenance_ml\data\ai4i2020.csv")

For example, if the CSV is in the same directory as the notebook, change the line to:

df = pd.read_csv("ai4i2020.csv")

3. Start Jupyter Notebook

jupyter notebook
Open predictive_maintenance.ipynb and run the cells in order.

# Disclaimer
This is an educational experiment on a benchmark dataset. The saved test results do not guarantee performance on real industrial equipment and should not be used as the sole basis for maintenance or safety decisions.
