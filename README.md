# Predictive_Maintenance-ML
An end-to-end Machine Learning pipeline for predicting industrial machine failure modes and monitoring operational parameters using the ai4i 2020 Predictive Maintenance Dataset.
# Dataset Overview

This project uses the ai4i2020.csv dataset, which contains 10,000 rows and 14 features representing industrial machinery sensor metrics:

Identifiers: UDI, Product ID, Type (Quality variant L, M, or H)

Sensor Metrics: Air temperature [K], Process temperature [K], Rotational speed [rpm], Torque [Nm], Tool wear [min]

Target Flag: Machine failure (0 = Normal, 1 = Failed)

Failure Modes:

TWF (Tool Wear Failure)

HDF (Heat Dissipation Failure)

PWF (Power Failure)

OSF (Overstrain Failure)

RNF (Random Failures)

# Model Performance & Results

I tested two primary algorithms on the dataset to see how effectively they could distinguish normal operation from machine failure:

Logistic Regression: Achieved around 97% accuracy. It serves as a solid baseline for overall predictive performance.

Random Forest Classifier: Achieved around 95% accuracy. It does a great job handling complex interactions between multiple sensor readings at once.

A Note on Accuracy & Imbalanced Data

Because machine failures are relatively rare in real life, over 95% of the records in this dataset represent normal, healthy operation. This means a model can get high raw accuracy just by guessing "no failure." When looking at these results, it's always important to pay attention to Precision, Recall, and F1-Score to ensure we're actually catching genuine failures when they occur.

# Key Takeaways

Torque and Rotational Speed were the biggest warning signs for mechanical failures—sudden spikes or imbalance here usually meant trouble.

Temperature Difference (Air vs. Process temp) was the main driver behind heat dissipation failures (HDF)
