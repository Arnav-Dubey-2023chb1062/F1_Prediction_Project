# -*- coding: utf-8 -*-
"""2025 BahrainGP Real.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1wP-iYAgjs_tEh5DgbBdHfq58ymjN2Mge
"""

!pip install fastf1 scikit-learn pandas numpy --quiet
import os
import fastf1

os.makedirs("/content/f1_cache", exist_ok=True)

fastf1.Cache.enable_cache("/content/f1_cache")
import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Load Bahrain 2024 Race Data
session_2024 = fastf1.get_session(2024, "Bahrain", "R")
session_2024.load()

# Extract key lap info
laps_2024 = session_2024.laps[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]].copy()
laps_2024.dropna(inplace=True)

# Convert timedelta to seconds
for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
    laps_2024[f"{col} (s)"] = laps_2024[col].dt.total_seconds()

sector_times_2024 = laps_2024.groupby("Driver")[[
    "Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"
]].mean().reset_index()

qualifying_2025 = pd.DataFrame({
    "Driver": ["VER", "NOR", "PIA", "LEC", "RUS", "HAM", "GAS", "ALO", "TSU", "SAI", "HUL", "OCO", "STR"],
    "QualifyingTime (s)": [90.423, 90.267, 89.841, 90.175, 90.009, 90.772, 90.216, 91.886, 91.303, 90.680, 92.067, 91.886, 92.283]
})

# Wet performance mapping
driver_wet_performance = {
    "VER": 0.975196, "HAM": 0.976464, "LEC": 0.975862, "NOR": 0.978179,
    "ALO": 0.972655, "RUS": 0.968678, "SAI": 0.978754, "TSU": 0.996338,
    "OCO": 0.981810, "GAS": 0.978832, "STR": 0.979857
}
qualifying_2025["WetPerformanceFactor"] = qualifying_2025["Driver"].map(driver_wet_performance)

# Season points (placeholder)
season_points = {
    "VER": 61, "NOR": 62, "PIA": 80, "LEC": 20, "RUS": 20, "HAM": 20, "GAS": 20,
    "ALO": 20, "TSU": 20, "SAI": 20, "HUL": 2, "OCO": 8, "STR": 11
}
qualifying_2025["SeasonPoints"] = qualifying_2025["Driver"].map(season_points)

# Weather: Bahrain International Circuit
API_KEY = "4e078bd52791a7bbc6b52a90051e12f7"
weather_url = f"http://api.openweathermap.org/data/2.5/forecast?lat=26.0325&lon=50.5106&appid={API_KEY}&units=metric"

try:
    response = requests.get(weather_url)
    response.raise_for_status()
    weather_data = response.json()

    forecast_time = "2025-04-30 15:00:00"  # Local race time
    forecast_data = next((f for f in weather_data.get("list", []) if f["dt_txt"] == forecast_time), None)

    rain_probability = forecast_data["pop"] if forecast_data else 0
    temperature = forecast_data["main"]["temp"] if forecast_data else 20

except Exception as e:
    print("Weather fetch failed:", e)
    rain_probability = 0
    temperature = 20

merged_data = qualifying_2025.merge(sector_times_2024, on="Driver", how="left")
merged_data["RainProbability"] = rain_probability
merged_data["Temperature"] = temperature

X = merged_data[[
    "QualifyingTime (s)", "Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)",
    "WetPerformanceFactor", "RainProbability", "Temperature", "SeasonPoints"
]].fillna(0)

y = merged_data.merge(
    laps_2024.groupby("Driver")["LapTime (s)"].mean(),
    left_on="Driver", right_index=True
)["LapTime (s)"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=38)

model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=38)
model.fit(X_train, y_train)

predicted_race_times = model.predict(X)
merged_data["PredictedRaceTime (s)"] = predicted_race_times
merged_data = merged_data.sort_values(by="PredictedRaceTime (s)")

print("\n Predicted 2025 Bahrain GP Winner \n")
print(merged_data[["Driver", "PredictedRaceTime (s)"]])
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"\n Model Error (MAE): {mae:.2f} seconds")

feature_importance = model.feature_importances_
features = X.columns

plt.figure(figsize=(8, 5))
plt.barh(features, feature_importance, color='skyblue')
plt.xlabel("Importance")
plt.title("Feature Importance in Race Time Prediction")
plt.tight_layout()
plt.show()