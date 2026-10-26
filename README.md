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
