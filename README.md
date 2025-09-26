# college-graduation-prediction
Machine learning analysis of U.S. colleges using the College.csv dataset. Includes regression models to predict graduation rates and model interpretation through feature importance.

Goal: 

Predict Graduation Rate (Grad.Rate) as a regression problem using features from College.csv.

Data & Features: 

Loads College.csv (columns like Apps, Accept, Enroll, Top10perc, Outstate, S.F.Ratio, Expend, Private, etc.).

Uses Private in the feature set (encoded to numeric).

Builds features from an engineered dataframe (df_eng), then sets
X = df_eng.drop(columns=["College", "Grad.Rate"] + raw_skewed)
(i.e., removes identifier, target, and raw skewed columns after creating transformed versions).

Target: y = df["Grad.Rate"].

Preprocessing / Feature Engineering: 

Handles missing values (uses isna/fillna/dropna patterns).

Identifies skewed numeric features and applies log1p transformations (keeps transformed features, drops raw skewed versions).

Removes near-constant features with VarianceThreshold.

Applies StandardScaler (standardization) for model stability/interpretability (used in the workflow).

Train/Test Split: 

train_test_split(X, y, test_size=0.2, random_state=42).

Models Trained: 

Linear Regression (LinearRegression()).

Random Forest Regressor (RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)).

Evaluation: 

Computes metrics on train and test:

RMSE (root_mean_squared_error)

MAE (mean_absolute_error)

R² (r2_score)

Includes typical diagnostic plots:

Correlation heatmap, boxplots, histograms (distribution/outliers).

Regression plots / predicted vs. actual and residual visuals.

Model Interpretation: 

Uses permutation_importance to rank feature importance (especially for the Random Forest).

Outputs / Takeaways: 

Compares Linear Regression vs Random Forest on RMSE/MAE/R².

Highlights the most influential predictors for Grad.Rate (via permutation importance).

Demonstrates a complete supervised regression pipeline: clean → engineer → split → train → evaluate → interpret.
