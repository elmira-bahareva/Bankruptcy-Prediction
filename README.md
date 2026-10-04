# Bankruptcy-Prediction
Predicting corporate bankruptcy from financial ratios, with a logistic regression model benchmarked against an adapted version of the Altman Z-score.

## Overview
Lenders and credit analysts must estimate the probability of corporate bankruptcy prior to granting credit or commiting capital. This project creates an interpretable classification model from financial ratio data, mitigates the extreme class imbalance characteristic of bankruptcy data, and benchmarks the model against a well-established traditional formula, to examine whether a modern technique yields a clear benefit relative to a far simpler, time-tested approach.

### Dataset
Company Bankruptcy Prediction (Kaggle, FEDESORIANO), based on data from the Taiwan Economic Journal (1999-2009). The final dataset contains 6,819 companies, 93 cleaned financial ratio features, and a binary bankruptcy outcome.

### Design Decisions

- **Two variables were removed** due to lack of variation: Net Income Flag (constant across all companies) and Liability-Assets Flag (non-zero for only 8 out of 6,819 companies).

- **Inconsistent leading whitespace in column names was stripped**.

- The cleaned dataset had **no missing values and no duplicate records**. 

### Class Imbalance

In this dataset, only 220 of 6,819 companies (3.2%) went bankrupt. Training a model on the raw data risks a classifier that looks accurate overall, but effectively never flags a single bankrupt company. To adress this:

- The data were split into training and test sets using stratification to keep the 96.8% / 3.2% class distribution in both subsets.

- SMOTE (Synthetic Minority Oversampling Technique) was used exclusively on the training set, generating synthetic bankrupt-company cases until the classes were balanced. The test set was left untouched in its original, real-world imbalance, ensuring that performance metrics reflect how the model would behave on realistic, unseen data.

### Models

**Logisitc Regression** was fitted using scaled features and trained on the SMOTE-balanced training data. The resulting coefficients provide a clear, linear interpretation of how each financial ratio affects the odds of bankruptcy, which is essential in credit risk settings, where model output must support transparent, defensible decisions for regulators, auditors and business users.

#### Results
| Metric | Class 0 (not bankrupt) | Class 1 (bankrupt) |
|--------|--------------------|------------|
| Precision| 0.99 | 0.20|
| Recall| 0.90 | 0.73 |

AUC: 0.884

#### Precision/Recall trade-off by threshold
| Threshold | Precision | Recall |
|-----------|-----------|--------|
|0.3|0.14|0.80|
|0.4|0.17|0.77|
|0.5 (default)| 0.20|0.73|
|0.6| 0.23|0.68|
|0.7|0.26|0.66|

In credit risk, the default 0.5 threshold is usually suboptimal because the consequeces of different errors are highly asymmetric. Since the penalty for missing a true bankruptcy (false negative) is far more damaging than chasing a possible bankruptcy (false positive), lenders tend to favour a lower probability threshold that increases recall, trading extra investigation work for a lower chance of missing a real default. The precise threshold should be chosen by weighting the unit cost of an extra review against the unit cost of an undetected bankruptcy.


**Benchmark: adapted Altman Z_score**

The Altman Z-score (1968) is a long-established formula for predicting bankruptcy risk from five financial ratios:

$$ Z = 1.2 \times X_1 + 1.4 \times X_2 + 3.3 \times X_3 + 0.6 \times X_4 + 1.0 \times X_5 $$

The dataset is pre-normalised and lacks the raw market-value and liability inputs specified in the original formula, so each term was approximated using the closest proxy variable. Accordingly, this should be viewed as an adaptation of the original formula, not a literal replication and is treated as such throughout.

|Component| Original Definition| Proxy Used|
|---------|--------------------|-----------|
| $X_1$ | Working Capital / Total Assets | Working Capital to Total Assets |
| $X_2$ | Retained Earnings / Total Assets | Retained Earnings to Total Assets|
| $X_3$ | EBIT / Total Assets | ROA(A) before interest and % after tax|
| $X_4$ | Market Value of Equity / Total Liabilities | Net worth/Assets $\div$ Debt ratio % |
| $X_5$ | Sales / Total Assets | Total Asset Turnover |

The original 1968 weights were retained unchanged. For the $X_4$ component, the Debt Ratio % was set to a minimum of 1% prior to division, to prevent near-zero-debt companies from producing extreme or infinite ratios.

**Does the adapted Z-score separate bankrupt from non-bankrupt companies?**
|   | Mean Z-score | Std |
|---|--------------|-----|
|Not Bankrupt| 11.38| 7.11|
|Bankrupt| 7.05 | 4.59|

Bankrupt companies score meaningfully lower, consistent with Altman's original theory that lower scores indicate higher risk, although the two distributions overlap considerably and the formula alone does not provide a clean separation.

### Comparison

|Approach| AUC|
|--------|----|
|Logistic Regression (93 features, SMOTE-balanced)| 0.884|
|Adapted Altman Z-score (5 ratios, 1968 formula)| 0.889|

Despite its age, the five-ratio Altman-style formula achieved performance comparable to, and in some metrics marginally superior to, a logisitc regression trained on the full set of 93 financial ratios with class balancing. This finding has a clear practical implications: where a simple, interpretable, long-established scoring method performs as well as a modern statistical model, it becomes a compwlling choice for real-world credit assesment. The Altman-style. formula requires no training data, no specialised software, and can be explained instantly to loan officers, regulators, and the companies being assessed.

>[!NOTE] 
>Methodology note: These AUC figures are not perfectly comparable. The logistic regression was assessed only on the unseen test set, while the Z-score was evaluated across all observations, as its a predetermined formula with no fitting procedure. While this is a standard and fair practice for comparing a learned model to a static rule, the distinction in evaluation bases should be clearly stated rather than implying methodological equivalence.

### Key Findings 

- **Borrowing dependency, debt ratios, and leverage were the strongest predictors of bankruptcy risk** in the logistic regression, consistent with financial theory.

- **Probability and liquidity measures** (Net Income or Total Assets, Persistent EPS,Current Ratio, Equity to Liability) were the strongest protective factors.

- The adapted Altman Z-score performed on par with the full logistic regression model, despite using a fraction of the information and no training process at all.

### Tools

Python, pandas, scikit-learn, imbalanced-learn, matplotlib

### Possible Extensions

- Investigate the ROA multicollinearity issue directly (correlation matrix or VIF analysis across the three ROA variants and Net Income to Total Assets)

- Try a model type less sensitive to class imbalance by design (e.g. a cost-sensitive classifier with class weighting, as an alternative to SMOTE)

- Test whether combining the Z-score itself as in input feathure to the logistic regression improves on either approach alone
