import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# 1. CREATE DATA (Simulating the Superstore Sales data)
np.random.seed(42)
dates = pd.date_range(start='2024-01-01', periods=200, freq='D')
sales = 100 + np.arange(200) * 0.5 + np.sin(np.arange(200) * (2 * np.pi / 30)) * 20 + np.random.normal(0, 5, 200)

df = pd.DataFrame({'Order Date': dates, 'Sales': sales})

# 2. FEATURE ENGINEERING (Required by Task)
df['Year'] = df['Order Date'].dt.year
df['Month'] = df['Order Date'].dt.month
df['Day'] = df['Order Date'].dt.day
df['DayOfWeek'] = df['Order Date'].dt.dayofweek

# 3. SPLIT DATA
X = df[['Year', 'Month', 'Day', 'DayOfWeek']]
y = df['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 4. TRAIN MODEL
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. FORECASTING & VISUALIZATION
predictions = model.predict(X_test)

plt.figure(figsize=(12, 6))
plt.plot(df['Order Date'], df['Sales'], label='Historical Sales', color='blue')
plt.plot(df['Order Date'].iloc[X_test.index], predictions, label='Forecasted Sales', color='red', linestyle='--')
plt.title('Sales Demand Forecast - Mohammed Najeeb (FIT/JAN26/ML4668)')
plt.xlabel('Date')
plt.ylabel('Sales Volume')
plt.legend()
plt.savefig('forecast_plot.png') # Saves the image for your README
plt.show()

print("Task 1 Completed Successfully. Plot saved as forecast_plot.png")