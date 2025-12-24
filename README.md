# Rainfall Prediction

## Goal

Predict whether rainfall will occur using weather measurement data in a binary classification task. The target variable is **rainfall** (binary: 0 or 1).

## Dataset

**Features (11 total):**
- All features are numeric (no categorical variables)
- Weather measurements: pressure, maxtemp, temperature, mintemp, dewpoint, humidity, cloud, sunshine, winddirection, windspeed
- Time: day
- **No missing values in training data**
- One missing value in test data (winddirection)

## Approach

### 1. Exploratory Data Analysis
- Examined data shape, types, and distributions
- Identified outliers in: pressure, dewpoint, humidity, cloud, windspeed
- Created correlation matrix to understand feature relationships
- Discovered strong correlations between temperature-related features (pressure, maxtemp, temperature, mintemp, dewpoint)

### 2. Feature Importance Analysis
- Built baseline XGBoost classifier on all features
- Used SHAP (SHapley Additive exPlanations) values to understand feature importance
- Generated waterfall and force plots to visualize feature contributions
- Identified most impactful features for prediction

### 3. Feature Selection
- Selected 7 key features based on SHAP analysis: cloud, dewpoint, humidity, pressure, mintemp, day, windspeed
- Reduced feature set to improve model efficiency and reduce noise

### 4. Model Comparison
- Tested multiple classification algorithms:
  - **XGBoost Classifier** (XGBClassifier)
  - **Random Forest Classifier** (RandomForestClassifier)
  - **Stochastic Gradient Descent Classifier** (SGDClassifier)
  - **Support Vector Classifier** (SVC with linear kernel)
  - **Logistic Regression** (LogisticRegression)

### 5. Model Selection & Validation
- Evaluated models using cross-validation scores (5-fold CV)
- Selected Logistic Regression as final model
- Validated performance using ROC-AUC score on held-out validation set

## Evaluation Metrics

- **Cross-validation accuracy** for model comparison
- **ROC-AUC score** for final model evaluation
- Train-test split: 25% training, 75% validation

## Results

- Successfully reduced feature space from 11 to 7 features using SHAP analysis
- Compared 5 different classification algorithms
- Logistic Regression selected as final model based on cross-validation performance
- Model trained on selected features and predictions generated for test set