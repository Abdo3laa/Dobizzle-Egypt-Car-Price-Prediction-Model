## 📊 Model Comparison – Dubizzle Car Price Prediction

### 🧠 Overview

Multiple regression models were trained and evaluated to predict car prices based on Dubizzle Egypt listings.  
The goal was to find the best-performing model in terms of accuracy and error, after proper preprocessing and **log transformation** of skewed data (`price`, `kilometers`).

---

### ✅ Test Performance Summary

| Model                   | R² Score | MAE (EGP) | RMSE (EGP) |
|-------------------------|----------|-----------|------------|
| Linear Regression       | 0.7865   | 291,317   | 1,016,598  |
| Random Forest Regressor| 0.7551   | 308,642   | 1,088,705  |
| Gradient Boosting       | 0.7315   | 371,683   | 1,140,086  |
| **XGBoost Regressor**   | **0.7678** | **315,759** | **1,060,266** |
| LightGBM Regressor      | 0.6081   | 448,007   | 1,377,232  |
| CatBoost Regressor      | 0.7103   | 385,354   | 1,184,142  |

> ℹ️ All metrics are calculated on **actual price values**, after reversing the log transformation (`np.expm1`).

---

### 🏆 Best Performing Model

✅ **XGBoost Regressor**
- Achieved the best balance between R² score and error values
- Suitable for deployment or dashboard integration
- Robust to outliers and skewed distributions

---

### 🔎 Key Insights

- Linear Regression performed very well considering its simplicity.
- Random Forest and Gradient Boosting had strong but slightly less accurate results.
- LightGBM underperformed, possibly due to lack of tuning.
- CatBoost handled categorical data smoothly but slightly less accurate than XGBoost.

---

### 💡 Recommendation

Proceed with **XGBoost Regressor** for:

- Saving model (`.pkl`)  
- Deployment or Streamlit prediction  
- Business dashboard integration  

To improve performance:
- Add feature engineering (e.g., `car_age`, `avg_price_city`)  
- Apply outlier treatment  
- Try model stacking or ensembling

---

## 🧹 Data Preprocessing Summary

The following preprocessing steps were applied before modeling:

- ✅ **Removed rows with missing values** in critical columns like `price`, `city`
- ✅ Replaced invalid kilometer values (`'-'`) with NaN, then cleaned
- ✅ Applied **log transformation** on `price` and `kilometers` to reduce skewness
- ✅ Split `location` into `area` and `city`
- ✅ Converted `date_posted` to 3 clean categories: `hours ago`, `days ago`, `weeks ago`
- ✅ Cleaned currency in `price` column (removed "EGP") and converted to integer
- ✅ **Filled missing values** with:
  - Mode for: `body_type`, `color`, `payment_option`, `interior`, `seats`
  - `'Unknown'` for: `model`
- ✅ Handled extreme outliers visually (via boxplots)
- ✅ Removed duplicates
- ✅ OneHotEncoded all categorical features
- ✅ Scaled numerical features using `StandardScaler`

Final dataset was exported as: **`dubizzle_cleaned_dataset.csv`**

---
## 👤 Author

**Abdelrahman Alaa**  
🔗 [LinkedIn – linkedin.com/in/3bdo-3laa1](https://www.linkedin.com/in/3bdo-3laa1)

---
