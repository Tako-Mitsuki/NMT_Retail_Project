# Auto-generated from Copy_of_NMT_Online_Retail.ipynb
# Module: feature_engineering

'''
Review this file: the code was heuristically split from the notebook.
Move functions/classes around as needed.
'''

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Feature & target
X = rfm[["Recency","Frequency","Monetary"]]
y = rfm["Future_Monetary"]

# Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Dự đoán & đánh giá
y_pred = model.predict(X_val)
print("RMSE:", np.sqrt(mean_squared_error(y_val, y_pred)))
print("R²:", r2_score(y_val, y_pred))


