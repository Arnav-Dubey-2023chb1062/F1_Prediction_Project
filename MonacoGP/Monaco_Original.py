import fastf1
import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

fastf1.Cache.enable_cache("f1_cache")

# Load the 2024 Monaco session data
session_2024 = fastf1.get_session(2024, 6, "R")  # Monaco is race 6
session_2024.load()
laps_2024 = session_2024.laps[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time", 
                              "Compound", "TyreLife", "TrackStatus", "Position"]].copy()
laps_2024.dropna(inplace=True)

# Convert lap and sector times to seconds
for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
    laps_2024[f"{col} (s)"] = laps_2024[col].dt.total_seconds()

# Aggregate sector times and additional metrics by driver
driver_stats_2024 = laps_2024.groupby("Driver").agg({
    "Sector1Time (s)": "mean",
    "Sector2Time (s)": "mean", 
    "Sector3Time (s)": "mean",
    "TyreLife": "mean",
    "Position": "mean"
}).reset_index()

driver_stats_2024["TotalSectorTime (s)"] = (
    driver_stats_2024["Sector1Time (s)"] +
    driver_stats_2024["Sector2Time (s)"] +
    driver_stats_2024["Sector3Time (s)"]
)

# Calculate tire degradation factor
tire_deg = {
    "VER": 0.08, "HAM": 0.12, "LEC": 0.10, "NOR": 0.09, "ALO": 0.11,
    "PIA": 0.09, "RUS": 0.11, "SAI": 0.10, "STR": 0.13, "HUL": 0.12,
    "OCO": 0.13, "ALB": 0.12, "GAS": 0.13
}

# Historical DNF probability at Monaco
dnf_probability = {
    "VER": 0.15, "HAM": 0.18, "LEC": 0.25, "NOR": 0.20, "ALO": 0.22,
    "PIA": 0.23, "RUS": 0.20, "SAI": 0.23, "STR": 0.25, "HUL": 0.22,
    "OCO": 0.24, "ALB": 0.23, "GAS": 0.24
}

# Driver experience at Monaco (years)
monaco_experience = {
    "VER": 8, "HAM": 16, "LEC": 5, "NOR": 4, "ALO": 19,
    "PIA": 1, "RUS": 3, "SAI": 8, "STR": 6, "HUL": 8,
    "OCO": 5, "ALB": 3, "GAS": 5
}

# Clean air race pace from racepace.py
clean_air_race_pace = {
    "VER": 93.191067, "HAM": 94.020622, "LEC": 93.418667, "NOR": 93.428600, "ALO": 94.784333,
    "PIA": 93.232111, "RUS": 93.833378, "SAI": 94.497444, "STR": 95.318250, "HUL": 95.345455,
    "OCO": 95.682128, "ALB": 94.891233, "GAS": 95.123456
}

# Quali data from Monaco GP
qualifying_2025 = pd.DataFrame({
    "Driver": ["VER", "NOR", "PIA", "RUS", "SAI", "ALB", "LEC", "OCO",
               "HAM", "STR", "GAS", "ALO", "HUL"],
    "QualifyingTime (s)": [  
        70.669, 69.954, 70.129, None, 71.362, 71.213, 70.063, 70.942,
        70.382, 72.563, 71.994, 70.924, 71.596
    ]
})

# Add additional features
qualifying_2025["TireDegradation"] = qualifying_2025["Driver"].map(tire_deg)
qualifying_2025["DNFProbability"] = qualifying_2025["Driver"].map(dnf_probability)
qualifying_2025["MonacoExperience"] = qualifying_2025["Driver"].map(monaco_experience)
qualifying_2025["CleanAirRacePace (s)"] = qualifying_2025["Driver"].map(clean_air_race_pace)

# Get weather data
API_KEY = "YOURAPIKEY"
weather_url = f"http://api.openweathermap.org/data/2.5/forecast?lat=43.7384&lon=7.4246&appid={API_KEY}&units=metric"
response = requests.get(weather_url)
weather_data = response.json()
forecast_time = "2025-05-25 13:00:00"
forecast_data = next((f for f in weather_data["list"] if f["dt_txt"] == forecast_time), None)

rain_probability = forecast_data["pop"] if forecast_data else 0
temperature = forecast_data["main"]["temp"] if forecast_data else 20
humidity = forecast_data["main"]["humidity"] if forecast_data else 65
wind_speed = forecast_data["wind"]["speed"] if forecast_data else 10

# Team performance data
team_points = {
    "McLaren": 279, "Mercedes": 147, "Red Bull": 131, "Williams": 51, "Ferrari": 114,
    "Haas": 20, "Aston Martin": 14, "Kick Sauber": 6, "Racing Bulls": 10, "Alpine": 7
}
max_points = max(team_points.values())
team_performance_score = {team: points / max_points for team, points in team_points.items()}

driver_to_team = {
    "VER": "Red Bull", "NOR": "McLaren", "PIA": "McLaren", "LEC": "Ferrari", 
    "RUS": "Mercedes", "HAM": "Mercedes", "GAS": "Alpine", "ALO": "Aston Martin", 
    "SAI": "Ferrari", "HUL": "Kick Sauber", "OCO": "Alpine", "STR": "Aston Martin",
    "ALB": "Williams"
}

qualifying_2025["Team"] = qualifying_2025["Driver"].map(driver_to_team)
qualifying_2025["TeamPerformanceScore"] = qualifying_2025["Team"].map(team_performance_score)

# Merge all data
merged_data = qualifying_2025.merge(
    driver_stats_2024[["Driver", "TotalSectorTime (s)", "TyreLife", "Position"]], 
    on="Driver", 
    how="left"
)

# Add weather conditions
merged_data["RainProbability"] = rain_probability
merged_data["Temperature"] = temperature
merged_data["Humidity"] = humidity
merged_data["WindSpeed"] = wind_speed

# Filter valid drivers
valid_drivers = merged_data["Driver"].isin(laps_2024["Driver"].unique())
merged_data = merged_data[valid_drivers]

# Define features and target
X = merged_data[[
    "QualifyingTime (s)", "RainProbability", "Temperature", "TeamPerformanceScore",
    "CleanAirRacePace (s)", "TireDegradation", "DNFProbability", "MonacoExperience",
    "Humidity", "WindSpeed", "TyreLife", "Position"
]]
y = laps_2024.groupby("Driver")["LapTime (s)"].mean().reindex(merged_data["Driver"])

# Impute missing values
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.3, random_state=42)

# Train gradient boosting model with more complex parameters
model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=4,
    min_samples_split=5,
    min_samples_leaf=3,
    random_state=42
)
model.fit(X_train, y_train)

# Make predictions
merged_data["PredictedRaceTime (s)"] = model.predict(X_imputed)

# Calculate race reliability score (inverse of DNF probability)
merged_data["ReliabilityScore"] = 1 - merged_data["DNFProbability"]
merged_data["FinalScore"] = merged_data["PredictedRaceTime (s)"] * merged_data["ReliabilityScore"]

# Sort results
final_results = merged_data.sort_values("FinalScore").reset_index(drop=True)

# Print results and model performance
print("\n🏁 Predicted 2025 Monaco GP Results 🏁\n")
print(final_results[["Driver", "FinalScore", "PredictedRaceTime (s)", "ReliabilityScore"]])
y_pred = model.predict(X_test)
print(f"\nModel Error (MAE): {mean_absolute_error(y_test, y_pred):.2f} seconds")

# Visualizations
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(final_results["CleanAirRacePace (s)"], final_results["PredictedRaceTime (s)"])
for i, driver in enumerate(final_results["Driver"]):
    plt.annotate(driver, (final_results["CleanAirRacePace (s)"].iloc[i], 
                         final_results["PredictedRaceTime (s)"].iloc[i]))
plt.xlabel("Clean Air Race Pace (s)")
plt.ylabel("Predicted Race Time (s)")
plt.title("Race Pace vs Predicted Time")

plt.subplot(1, 2, 2)
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=True)
plt.barh(feature_importance['Feature'], feature_importance['Importance'])
plt.xlabel("Importance")
plt.title("Feature Importance")
plt.tight_layout()
plt.show()

# Print podium predictions
print("\n🏆 Predicted Podium 🏆")
print(f"🥇 P1: {final_results.iloc[0]['Driver']} ({final_results.iloc[0]['FinalScore']:.2f})")
print(f"🥈 P2: {final_results.iloc[1]['Driver']} ({final_results.iloc[1]['FinalScore']:.2f})")
print(f"🥉 P3: {final_results.iloc[2]['Driver']} ({final_results.iloc[2]['FinalScore']:.2f})")


