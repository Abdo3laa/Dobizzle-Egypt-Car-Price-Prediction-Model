# 🚗 Dubizzle Egypt Car Price Prediction – ML Pipeline, App & Dashboard

A comprehensive data science and machine learning pipeline designed to predict used car prices from **Dubizzle Egypt** listings (available at [https://www.dubizzle.com.eg](https://www.dubizzle.com.eg)). This project encompasses **data scraping**, **exploratory data analysis (EDA)**, **preprocessing**, **model training**, a professional **Streamlit web app**, and an interactive **Power BI dashboard** for price prediction and market insights.

![Dubizzle Cars Logo](dubizzle-cars-logo.png)

---

## 📁 Project Structure

| File / Folder                             | Description |
|-------------------------------------------|-------------|
| `Dubizzle_Scraping.py`                   | Python script utilizing `requests` and `BeautifulSoup` to scrape car listing links and raw data from Dubizzle Egypt. |
| `dubizzle_full_dataset.csv`              | Raw dataset post-scraping, potentially containing missing or inconsistent values. |
| `Dubizzle_Preprocessing.ipynb`           | Jupyter notebook for data cleaning, null handling, outlier management, feature engineering (e.g., log transformations), and encoding. |
| `dubizzle_cleaned_dataset.csv`           | Processed dataset optimized for modeling and app deployment. |
| `Dubizzle_Modeling.ipynb`                | Notebook training multiple regression models (Linear, Ridge, Random Forest, etc.) with GridSearchCV, evaluated via R², RMSE, and residuals. |
| `LinearRegression_model.pkl`             | Serialized Linear Regression model saved using `pickle`. |
| `Deployment.py`                          | Streamlit app featuring three main pages: Home (KPIs), Visualizations (Plotly-based), and Price Prediction. |
| `dubizzle-cars-logo.png`                 | Logo integrated into the Streamlit app homepage. |
| `PowerBI_Dashboard/`                     | Contains Power BI dashboard file (`.pbix`) and visual exports for 3 report pages. |

---

## 📊 Power BI Dashboard

An interactive dashboard built with **Power BI** for visualizing car market trends across Egypt. It provides intuitive insights from the cleaned dataset, designed to assist in business decision-making and car price comparisons.

**Folder**: `PowerBI_Dashboard/`  
**Includes**:
- `Dubizzil_Cars_Dashboard.pbix`: Main dashboard file.
- `Market Overview.png`: Total listings, price insights, and fuel type distribution.
- `Brands Insights.png`: Top brands by listing count, price averages, and KPIs.
- `Demographics.png`: Insights on car colors, body types, seating, and locations.

Each page highlights key metrics using Dubizzle's visual style and color palette.

---

## 🧠 Project Goals

- Scrape and organize real-time car listings data from Dubizzle Egypt.
- Clean and transform features (e.g., log-scaling kilometers).
- Encode categorical variables with OneHotEncoder.
- Train and tune multiple regression models.
- Visualize market trends: top brands, price distributions, and location insights.
- Deploy a robust **Streamlit web app** with comprehensive prediction capabilities.
- Build a business-oriented dashboard with **Power BI**.

---

## 🌐 Streamlit Web App Features

The app (`Deployment.py`) offers an interactive experience structured as follows:

### 1. 🏠 Home Page
- Welcome message with logo.
- Project overview and trained model summary.

### 2. 📊 Visualizations Page
Leverages `Plotly` for interactive insights:
- 📈 Average price by brand (top 15).
- 📍 Listings distribution by city (top 15).
- 🏷️ Price distribution histogram.
- 🚙 Year vs. price scatter plot by fuel type.
- 🏘️ Area-wise average prices (e.g., Madinaty, New Cairo).

### 3. 📈 Predict Price Page
- Dynamic form: Select brand to filter models.
- Input features: year, transmission, color, etc.
- Predicts using the **Linear Regression model**.
- Displays **estimated market price in EGP** with a 95% confidence interval.

---

## 📊 Model Comparison – Dubizzle Car Price Prediction

### 🧠 Overview

Multiple regression models were trained and evaluated to predict car prices based on Dubizzle Egypt listings, with preprocessing including log transformation of skewed data (`price`, `kilometers`).

### ✅ Test Performance Summary

| Model                   | R² Score | MAE (EGP) | RMSE (EGP) |
|-------------------------|----------|-----------|------------|
| **Linear Regression**       | **0.7865** | **291,317**   | **1,016,598**  |
| Random Forest Regressor | 0.7551   | 308,642   | 1,088,705  |
| Gradient Boosting       | 0.7315   | 371,683   | 1,140,086  |
| XGBoost Regressor       | 0.7678   | 315,759   | 1,060,266  |
| LightGBM Regressor      | 0.6081   | 448,007   | 1,377,232  |
| CatBoost Regressor      | 0.7103   | 385,354   | 1,184,142  |

> ℹ️ Metrics are calculated on actual price values post `np.expm1` log reversal.

---

### 🏆 Best Performing Model

✅ **Linear Regression**  
- Achieved the **highest R² score** and **lowest error values** overall  
- Despite its simplicity, it outperformed more complex models like XGBoost and Random Forest  
- Recommended for deployment, price prediction, and dashboard integration

---

### 🔎 Key Insights

- **Linear Regression** outperformed all advanced models, showing the effectiveness of clean data and proper preprocessing  
- **XGBoost** came in close and remains a flexible alternative  
- **Random Forest** and **Gradient Boosting** had decent but slightly lower accuracy  
- **LightGBM** underperformed in this context, possibly due to minimal hyperparameter tuning  
- **CatBoost** handled categorical features well, but didn’t match the performance of simpler models

---

### 💡 Recommendation

Adopt **Linear Regression** for:
- Model serialization (`.pkl`)
- Streamlit app deployment
- Business dashboard integration

To further enhance performance:
- Incorporate feature engineering (e.g., `car_age`, `avg_price_city`)
- Apply outlier detection/removal
- Explore ensemble models for marginal improvements

---

## 🧹 Data Preprocessing Summary

Preprocessing steps applied prior to modeling:
- ✅ Removed rows with missing values in critical columns (`price`, `city`)
- ✅ Replaced invalid kilometer values (`'-'`) with NaN, then cleaned
- ✅ Applied log transformation to `price` and `kilometers` to mitigate skewness
- ✅ Split `location` into `area` and `city`
- ✅ Categorized `date_posted` into `hours ago`, `days ago`, `weeks ago`
- ✅ Cleaned `price` column (removed "EGP") and converted to integer
- ✅ Filled missing values with:
  - Mode for: `body_type`, `color`, `payment_option`, `interior`, `seats`
  - `'Unknown'` for: `model`
- ✅ Managed extreme outliers via boxplot analysis
- ✅ Removed duplicate entries
- ✅ Applied OneHotEncoding to all categorical features
- ✅ Scaled numerical features with `StandardScaler`

> Final dataset exported as: `dubizzle_cleaned_dataset.csv`

---

## 👤 Author

**Abdelrahman Alaa**  
🔗 [LinkedIn – linkedin.com/in/3bdo-3laa1](https://www.linkedin.com/in/3bdo-3laa1)

---

## 📦 Requirements

```bash
pip install streamlit pandas numpy matplotlib seaborn plotly scikit-learn
