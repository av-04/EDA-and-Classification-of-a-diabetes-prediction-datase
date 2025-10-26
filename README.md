Diabetes Risk Prediction Project
This project analyzes the Diabetes Health Indicators Dataset (100,000 records) to build a machine learning model for classifying a patient's diabetes risk level.

A key part of this analysis was identifying and correcting for severe data leakage (resulting in 100% accuracy) to build a more realistic and useful model.

Project Workflow
Data Loading: Imported the dataset using the kagglehub library.

Exploratory Data Analysis (EDA):

Checked for missing values (none found).

Analyzed outliers using box plots for all numerical features.

Plotted correlation heatmaps to understand feature relationships.

Data Preprocessing:

Outlier Handling: Capped (Winsorized) extreme outliers using the 1.5 * IQR rule.

Encoding: Converted all categorical features (e.g., gender, smoking_status) to numerical data using Ordinal and One-Hot encoding.

Scaling: Applied StandardScaler (Standardization) and MinMaxScaler (Normalization) to features, which is critical for distance-based models (SVM, KNN).

Modeling & Data Leakage:

Transformed the target variable diabetes_risk_score (0-100) into a 3-class problem (Low, Medium, High) using pd.qcut.

Initial models returned 99-100% accuracy.

Diagnosed Data Leakage: The target diabetes_risk_score is a calculated score. The features (x) included the "ingredients" used in its calculation (e.g., hba1c, glucose_fasting, diagnosed_diabetes, bmi).

🛠️ Libraries Used:

pandas & numpy for data manipulation
matplotlib & seaborn for visualization
kagglehub for data loading
scikit-learn (sklearn) for:
StandardScaler & MinMaxScaler
train_test_split
DecisionTreeClassifier
RandomForestClassifier
LinearSVC (SVM)
KNeighborsClassifier (KNN)
classification_report & confusion_matrix

Model Accuracy Comparison 🎯
##  Model Accuracy Comparison

| Model | Accuracy (%) | Notes |
| :--- | :--- | :--- |
| **Random Forest** | ~88.0% | Best all-rounder (requires no scaling). |
| **SVM (LinearSVC)** | **81.04%** | Top-performing of your three models. **Scaling is required.** |
| **Decision Tree** | 78.62% | Best for explainability (seeing the "rules"). |
| **KNN** | 78.38% | Simple but effective. **Scaling is required.** |
