# ==========================================
# Assignment 7 (MUDE, IITM): Regression Analysis and Diagnostics
# Dataset: Concrete Compressive Strength (Kaggle)
# https://www.kaggle.com/datasets/zain280/concrete-data/data
# ==========================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

# ==========================================
# Load dataset (common to all tasks)
# ==========================================
df = pd.read_csv("Concrete_Data.csv")  # adjust file name if needed
y = df["Strength"].values
X = df.drop(columns=["Strength"])

# predictor names
predictors = X.columns.tolist()

# ==========================================
# Task 1: Quick sanity checks & exploratory plots
# ==========================================
def task1_exploration():
    print("Shape:", df.shape)
    print("Data types:\n", df.dtypes)
    print("\nFirst 10 rows:\n", df.head(10))

    # summary stats
    print("\nSummary statistics:")
    print(df.describe().T[["mean", "std", "min", "max"]])
    
    # plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(y, bins=30, color="skyblue")
    axes[0].set_title("Histogram of Strength")

    axes[1].scatter(df["Cement"], y, alpha=0.6)
    axes[1].set_xlabel("Cement")
    axes[1].set_ylabel("Strength")

    axes[2].scatter(df["Water"], y, alpha=0.6)
    axes[2].set_xlabel("Water")
    axes[2].set_ylabel("Strength")

    plt.tight_layout()

# ==========================================
# Task 2: Single-predictor fits & ranking
# ==========================================
def task2_single_predictors():
    results = []
    for var in predictors:
        x = df[var].values
        slope, intercept, r_value, p_value, stderr = stats.linregress(x, y)
        results.append([var, intercept, slope, stderr, r_value**2, p_value])

    results_df = pd.DataFrame(
        results, columns=["Predictor", "Intercept", "Slope", "StdErr", "R2", "p-value"]
    ).sort_values(by="p-value")

    print("\nRanking predictors by p-value:\n", results_df)

    # Example: residuals + QQ for one predictor (say "Cement")
    x = df["Cement"].values
    slope, intercept, _, _, _ = stats.linregress(x, y)
    yhat = intercept + slope * x
    residuals = y - yhat

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(yhat, residuals, alpha=0.6)
    axes[0].axhline(0, color="red")
    axes[0].set_title("Residuals vs Fitted")

    stats.probplot(residuals, dist="norm", plot=axes[1])
    axes[1].set_title("Q-Q Plot")

    plt.tight_layout()

# ==========================================
# Task 3: Multivariable OLS, residual diagnostics
# ==========================================

def task3_multivariable_ols():
    # Add intercept
    Xmat = np.c_[np.ones(X.shape[0]), X.values]
    beta, _, _, _ = np.linalg.lstsq(Xmat, y, rcond=None)
    yhat = Xmat @ beta
    residuals = y - yhat

    # Make 2x3 grid (so we can fit residuals vs 2 predictors + Q-Q plot)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    # Residuals vs fitted
    axes[0].scatter(yhat, residuals, alpha=0.5)
    axes[0].axhline(0, color="red")
    axes[0].set_title("Residuals vs Fitted")

    # Residuals vs key predictors
    for i, var in enumerate(["Cement", "Water"]):
        axes[1+i].scatter(df[var], residuals, alpha=0.5)
        axes[1+i].axhline(0, color="red")
        axes[1+i].set_title(f"Residuals vs {var}")

    # Q-Q plot of residuals
    stats.probplot(residuals, dist="norm", plot=axes[3])
    axes[3].set_title("Q-Q Plot")

    # Hide unused axes (if any)
    for j in range(4, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()

# ==========================================
# Task 4: Outliers & influence
# ==========================================
def task4_outliers_influence():
    # Build design matrix with intercept
    Xmat = np.c_[np.ones(X.shape[0]), X.values]
    
    # Solve least squares regression
    beta, _, _, _ = np.linalg.lstsq(Xmat, y, rcond=None)
    
    # Predicted values 
    yhat = Xmat @ beta
    
    # Residuals
    residuals = y - yhat

    # Mean Squared Error
    mse = np.mean(residuals**2)

    # leverage (H matrix diagonal)
    H = Xmat @ np.linalg.inv(Xmat.T @ Xmat) @ Xmat.T
    leverage = np.diag(H)

    # studentized residuals
    stud_resid = residuals / np.sqrt(mse * (1 - leverage))

    # Cook’s distance
    cooks_d = (stud_resid**2 / Xmat.shape[1]) * (leverage / (1 - leverage))

    infl_df = pd.DataFrame({
        "Index": range(len(y)),
        "Residual": residuals,
        "StudResid": stud_resid,
        "CooksD": cooks_d
    }).sort_values(by="CooksD", ascending=False)

    print("\nTop 5 influencers:\n", infl_df.head(5))

# ==========================================
# Task 5: Interactions, quadratic terms & ANOVA
# ==========================================
def task5_interactions_anova():
    x1 = df["Cement"].values
    x2 = df["Water"].values

    X_base = np.c_[np.ones(len(y)), x1, x2]
    beta_base, _, _, _ = np.linalg.lstsq(X_base, y, rcond=None)
    rss_base = np.sum((y - X_base @ beta_base) ** 2)

    X_aug = np.c_[np.ones(len(y)), x1, x2, x1*x2, x1**2, x2**2]
    beta_aug, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
    rss_aug = np.sum((y - X_aug @ beta_aug) ** 2)

    df_base = X_base.shape[1]
    df_aug = X_aug.shape[1]
    f_stat = ((rss_base - rss_aug) / (df_aug - df_base)) / (rss_aug / (len(y) - df_aug))
    p_val = 1 - stats.f.cdf(f_stat, df_aug - df_base, len(y) - df_aug)

    print(f"ANOVA Comparison: F={f_stat:.3f}, p={p_val:.4f}")

# ==========================================
# Task 6: Heteroscedasticity remedies & prediction intervals
# ==========================================
def task6_remedies_prediction():
    Xmat = np.c_[np.ones(X.shape[0]), X.values]
    beta, _, _, _ = np.linalg.lstsq(Xmat, y, rcond=None)
    yhat = Xmat @ beta
    residuals = y - yhat

    # log transform remedy
    y_log = np.log1p(y)
    beta_log, _, _, _ = np.linalg.lstsq(Xmat, y_log, rcond=None)
    yhat_log = Xmat @ beta_log
    residuals_log = y_log - yhat_log

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(yhat, residuals, alpha=0.5)
    axes[0].axhline(0, color="red")
    axes[0].set_title("Original Residuals")

    axes[1].scatter(yhat_log, residuals_log, alpha=0.5)
    axes[1].axhline(0, color="red")
    axes[1].set_title("Residuals after log(y)")

    plt.tight_layout()

    # prediction intervals (simplified normal theory)
    newX = np.array([[1, 400, 100, 100, 180, 5, 900, 700, 28],   # mix1
                     [1, 300, 150, 50, 200, 8, 850, 600, 14],   # mix2
                     [1, 250, 50, 50, 150, 6, 1000, 650, 7]])  # mix3
    preds = newX @ beta
    mse = np.mean(residuals**2)
    se = np.sqrt(mse)
    intervals = [(p - 1.96*se, p + 1.96*se) for p in preds]
    print("\nPredictions with 95% intervals:")
    for i, (p, ci) in enumerate(zip(preds, intervals)):
        print(f"Mix {i+1}: pred={p:.2f}, 95% CI={ci}")

# ==========================================
# Task 7: Model selection & cross-validation
# ==========================================
def task7_model_selection():
    def model_fit(Xmat, y):
        beta, _, _, _ = np.linalg.lstsq(Xmat, y, rcond=None)
        yhat = Xmat @ beta
        rss = np.sum((y - yhat) ** 2)
        n, k = Xmat.shape
        aic = n * np.log(rss/n) + 2 * k
        bic = n * np.log(rss/n) + k * np.log(n)
        r2 = 1 - rss / np.sum((y - y.mean())**2)
        adj_r2 = 1 - (1-r2)*(n-1)/(n-k)
        return {"RSS": rss, "AIC": aic, "BIC": bic, "AdjR2": adj_r2}

    # baseline
    X_base = np.c_[np.ones(len(y)), X.values]
    res_base = model_fit(X_base, y)

    # + interactions
    x1 = df["Cement"].values
    x2 = df["Water"].values
    X_int = np.c_[np.ones(len(y)), X.values, x1*x2]
    res_int = model_fit(X_int, y)

    # + quadratic
    X_quad = np.c_[np.ones(len(y)), X.values, x1**2, x2**2]
    res_quad = model_fit(X_quad, y)

    # transformed
    res_log = model_fit(X_base, np.log1p(y))

    results = pd.DataFrame([res_base, res_int, res_quad, res_log],
                           index=["Baseline", "+Interaction", "+Quadratic", "Log(y)"])
    print("\nModel selection table:\n", results)

# ==========================================
# Run if executed
# ==========================================
if __name__ == "__main__":
    # Uncomment as needed
    # task1_exploration()
    # task2_single_predictors()
    # task3_multivariable_ols()
    # task4_outliers_influence()
    # task5_interactions_anova()
    # task6_remedies_prediction()
    # task7_model_selection()

    plt.show()
