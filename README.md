# USPS Occupancy Regression

A comprehensive, end-to-end machine learning pipeline that scrapes, cleans, explores, and models USPS facility data to predict building occupancy dates. By converting a sprawling Freedom of Information Act (FOIA) dataset into actionable insights, this project demonstrates how to identify which postal facilities may need retrofits first, based on age and size characteristics.

Read the full (ML design process) report here:

---

## Table of Contents

- [Background & Motivation](#background--motivation)  
- [Dataset & FOIA Context](#dataset--foia-context)  
- [Repository Structure](#repository-structure)  
- [Environment & Dependencies](#environment--dependencies)  
- [Data Collection](#data-collection)  
- [Data Cleaning & Merging](#data-cleaning--merging)  
- [Feature Engineering & Dimensionality Reduction](#feature-engineering--dimensionality-reduction)  
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)  
- [Modeling Pipeline](#modeling-pipeline)  
- [Evaluation Metrics & Results](#evaluation-metrics--results)  
- [Key Insights & Interpretation](#key-insights--interpretation)  
- [Future Directions](#future-directions)  
- [Verification of README Length](#verification-of-readme-length)  

---

## Background & Motivation

Postal facilities across the United States number in the tens of thousands, yet comprehensive metadata—such as the date each building opened for public use—is not centrally catalogued in an accessible dataset. For grant writers, energy-efficiency specialists, and infrastructure planners, knowing the age of a facility is critical to prioritize retrofits, allocate budget, and plan maintenance schedules.

This project began as a curiosity-driven exploration of publicly available USPS FOIA data. It demonstrates how to:

1. **Automate data collection** from a FOIA website guarded by anti-scraping measures  
2. **Consolidate** dozens of per-state CSV files into a unified DataFrame  
3. **Clean and transform** semi-structured records into machine-readable features  
4. **Explore patterns** in facility size, location, and age  
5. **Build and compare** regression models to predict occupancy dates  
6. **Highlight** how predictive insights can guide energy-efficiency retrofits  

---

## Dataset & FOIA Context

- **Source URL:**  
  `https://about.usps.com/who/legal/foia/owned-facilities.htm`  
- **Data License:**  
  Public domain under the Freedom of Information Act  
- **Format:**  
  Individual `.csv` files per state, linked via HTML on the FOIA page  
- **Challenges Addressed:**  
  - HTTP blocking of non-browser clients  
  - Inconsistent header/footer rows in each CSV  
  - Hundreds of extraneous “artifact” columns from regional site data  

---

## Repository Structure

├── README.md # This overview

├── scrape_beautifulsoup.py # Web scraper for FOIA CSVs

├── webscraping_cleanup.ipynb # Jupyter notebook covering cleaning, EDA, modeling

└── checking_output.ipynb # Utility to extract best CV score from training logs

## Core libraries used:
requests, beautifulsoup4 — scraping
pandas, numpy, glob — data handling
scikit-learn — preprocessing & modeling
seaborn, matplotlib, plotly — visualizations
tensorflow, tensorflow_decision_forests — optional device placement & alternative models



# Data Collection

**Script:** `scrape_beautifulsoup.py`

- **Custom Headers to Bypass Blocking**  
  In lines 7–15, we define a `User-Agent` header string that mimics Chrome on Windows. USPS blocks direct programmatic GETs, so this header prevents early 403 responses.

- **HTML Parsing & Link Extraction**  
  Lines 16–25 use BeautifulSoup to find all `<a href="…">` tags ending in `.csv`. By wrapping each `href` in `requests.compat.urljoin`, we build absolute URLs for download.

- **Download Loop with Throttling**  
  Lines 26–45 create `webscraping_USPS/csv_results/` and, for each CSV link:  
  1. Pause 2 seconds (`time.sleep(2)`) to avoid IP blocking  
  2. Extract the last two letters of the filename (state prefix)  
  3. Save as `file_<state>.csv`  
  4. Print progress  

> **Reference:** See pages 2–4 of the PDF report for a narrative of these choices and their rationale.

---

# Data Cleaning & Merging

**Notebook:** `webscraping_cleanup.ipynb` (cells 1–20)

- **Single-File Validation** (lines 5–12)  
  1. Read one CSV (`file_ak.csv`) with `skiprows=5`, `header=0`  
  2. Drop last two rows to align header names  

- **Merge All CSVs** (lines 13–22)  
  1. Use `glob.glob` to list all `*.csv` files  
  2. Concatenate into `merged_df` via `pd.concat(…)` with `ignore_index=True`  

- **Initial Inspection** (lines 23–27)  
  - `merged_df.info()` reveals ~29 raw columns and ~19,483 rows  
  - Many nulls and unwanted regional columns  

- **Column Pruning** (lines 28–35)  
  1. Drop state-specific artifacts (e.g. “OHIO 1”, “AKRON”)  
  2. Retain nine core columns:  
     ```
     ['District', 'PO Name', 'Unit Name',
      'County', 'City', 'ST', 'ZIP Code',
      'Bldg Occu Date', 'Int Sq Ft']
     ```  

> **Reference:** See PDF pages 6–8 for discussion on column selection and one-hot encoding plans.

---

# Feature Engineering & Dimensionality Reduction

- **One-Hot Encoding** (lines 36–55)  
  - For each categorical field (`District`, `PO Name`, `Unit Name`, `County`, `City`, `ST`, `ZIP Code`):
    - Lowercase & strip spaces  
    - Fill nulls with “unknown”  
    - Apply `pd.get_dummies`, suffix columns with the original field name  
  - **Outcome:** feature count grows from 9 → ~22,000  

- **Numeric Conversions & Imputation** (lines 56–75)  
  - **`Int Sq Ft`:** remove commas → numeric → fill NaNs with median → cast to `int`  
  - **`Bldg Occu Date`:** parse to datetime → subtract epoch `1900-01-01` → convert to days → drop NaNs  
  - **Rationale:** Pages 8–10 of the PDF explain epoch choice to avoid ambiguous YYYYMMDD conversions.  

- **Outlier Trimming** (lines 76–85)  
  - Retain rows within the 1st–88th percentile for `Int Sq Ft` and the 1st–99th percentile for `Bldg Occu Date`  

- **VarianceThreshold** (lines 86–95)  
  - Remove features with variance < 0.01  
  - **Result:** dims collapse from ~22,000 → ~103, enabling fast model training  

> For full code details, see the notebook cells referenced above.

---

# Exploratory Data Analysis (EDA)

1. **Distribution of Internal Square Footage**  
   - **Insert Figure 1:** histogram of `Int Sq Ft` **BEFORE** outlier removal  
     _(e.g. `images/int_sq_ft_before.png`)_  
   - **Insert Figure 2:** histogram of `Int Sq Ft` **AFTER** cleaning  
     _(e.g. `images/int_sq_ft_after.png`)_  

   **Discussion (PDF p.9–10):**  
   - Initial plot reveals extreme outliers (≈1.75 × 10⁶ sq ft)  
   - After trimming, most buildings fall under 3,000 sq ft, aligning with real-world facility sizes  

2. **Top 10 Districts by Count**  
   - **Insert Figure 3:** barplot of top 10 districts  
     _(e.g. `images/top10_districts.png`)_  

   **Discussion (PDF p.10):**  
   - Regions like **KS-MO** and **CHICAGO** dominate facility counts  
   - Reflects service density and distribution of the USPS network  

3. **Occupancy Date Distribution**  
   - **Insert Figure 4:** histogram of `Bldg Occu Date` with 40 bins  
     _(e.g. `images/occu_date_40bins.png`)_  
   - **Insert Figure 5:** histogram with 400 bins  
     _(e.g. `images/occu_date_400bins.png`)_  

   **Discussion (PDF p.11):**  
   - Days since 1900 span ~5,000–40,000 (≈1937–2009)  
   - Peak around 30,000 (~1983), suggesting a building boom  

4. **Occupancy Date vs. Square Footage**  
   - **Insert Figure 6:** scatterplot  
     _(e.g. `images/date_vs_sqft.png`)_  

   **Discussion (PDF p.12):**  
   - No clear linear trend; suggests complex interactions between age and size  

---

# Modeling Pipeline

From this point forward, code snippets are conceptual only:

- **Train/Test Split**  
  75% training, 25% testing with fixed `random_state` for reproducibility

- **Baseline Models**  
  - Linear Regression (default)  
  - Random Forest Regressor (default, `random_state=42`)

- **Ensemble Models**  
  - Gradient Boosting Regressor (default hyperparameters)  
  - Gradient Boosting Regressor tuned via RandomizedSearchCV  
    - **Parameter grid:**  
      - `n_estimators`: [100, 200, 500]  
      - `learning_rate`: [0.1, 0.05, 0.01]  
      - `max_depth`: [3, 5, 10]  
      - `min_samples_split`: [2, 5, 10]  
      - `min_samples_leaf`: [1, 2, 5]  
    - **Rationale:** small grid balances compute and performance (PDF p.15–16)

- **Cross-Validation & Stability**  
  5-fold CV measuring mean and standard deviation of R² for each model

---

# Evaluation Metrics & Results

| Model                             | MAE      | MSE            | R²    | R² std |
|-----------------------------------|----------|----------------|-------|--------|
| Linear Regression                 | 3,094.09 | 31,125,694.88  | 0.217 | ±0.008 |
| Random Forest (default)           | 2,511.54 | 28,464,127.06  | 0.280 | ±0.011 |
| Gradient Boosting (default)       | 2,699.59 | 25,383,656.15  | 0.358 | ±0.009 |
| Gradient Boosting + Random Search | 2,598.26 | 24,785,840.83  | 0.373 | ±0.007 |

**Additional Metrics:**  
- **Median Absolute Error (MedAE):** reflects typical prediction error robust to outliers  
- **Max Error:** identifies worst-case prediction gap  

---

# Key Insights & Interpretation

- **Importance of cleaning:** outlier trimming and median imputation were essential to reduce noise (PDF p.9).  
- **Dimensionality matters:** dropping low-variance features cut runtime dramatically without sacrificing predictive power (PDF p.13).  
- **Model trade-offs:**  
  - Linear Regression: fast but underfits non-linearities  
  - Random Forest: better but can overfit  
  - Gradient Boosting: best default performance; tuning yields further gains  
- **Error distribution:** bell-shaped residuals indicate symmetric errors; actual vs. predicted scatter (PDF p.17) shows boundary cases remain challenging.

---

# Future Directions

- Enhanced outlier detection: explore IQR- or clustering-based methods  
- Additional features: integrate geographic coordinates, facility subtype, or economic indicators  
- Advanced algorithms: test XGBoost, LightGBM, neural networks  
- Interactive dashboard: deploy a Streamlit or Dash app  
