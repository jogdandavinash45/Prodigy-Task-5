"""
PRODIGY INFOTECH - DATA SCIENCE INTERNSHIP
Task 05: Traffic Accident Data Analysis
=========================================
Objective: Analyze traffic accident data to identify patterns related
to road conditions, weather, and time of day. Visualize accident
hotspots and contributing factors.

Dataset: US Accidents (Kaggle) — https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents
(Simulated with realistic distributions for offline use)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. SIMULATE US ACCIDENT DATASET
# ─────────────────────────────────────────────
np.random.seed(42)
n = 3000

# Time features
hours   = np.random.choice(range(24), n, p=[
    0.015,0.010,0.008,0.007,0.006,0.010,   # 0-5 AM (low traffic)
    0.025,0.060,0.075,0.055,0.045,0.050,   # 6-11 AM (morning rush)
    0.050,0.045,0.040,0.042,0.048,0.072,   # 12-17 PM (afternoon)
    0.068,0.060,0.050,0.040,0.032,0.022    # 18-23 PM (evening/night)
])
months     = np.random.choice(range(1, 13), n, p=[
    0.09,0.08,0.09,0.08,0.08,0.07,
    0.07,0.08,0.08,0.09,0.09,0.10
])
days_week  = np.random.choice(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'],
                               n, p=[0.16,0.16,0.15,0.16,0.18,0.11,0.08])
severity   = np.random.choice([1,2,3,4], n, p=[0.05,0.55,0.30,0.10])

# Weather
weather    = np.random.choice(
    ['Clear','Cloudy','Rain','Snow','Fog','Thunderstorm','Windy','Hail'],
    n, p=[0.38,0.28,0.15,0.06,0.05,0.04,0.03,0.01])
temp_f     = np.clip(np.random.normal(60, 20, n), -10, 115)
humidity   = np.clip(np.random.normal(60, 20, n), 10, 100)
visibility = np.where(weather.isin(['Fog','Snow','Rain']) if hasattr(weather,'isin')
                      else np.isin(weather, ['Fog','Snow','Rain']),
                      np.clip(np.random.normal(3, 2, n), 0.1, 10),
                      np.clip(np.random.normal(9, 1.5, n), 0.5, 10))
wind_speed = np.clip(np.random.normal(10, 8, n), 0, 60)

# Road features
road_cond  = np.random.choice(['Dry','Wet','Ice','Snow','Slippery'], n, p=[0.55,0.25,0.08,0.07,0.05])
junction   = np.random.choice([True, False], n, p=[0.40, 0.60])
crossing   = np.random.choice([True, False], n, p=[0.20, 0.80])
traffic_sig= np.random.choice([True, False], n, p=[0.35, 0.65])
stop       = np.random.choice([True, False], n, p=[0.15, 0.85])
day_night  = np.where(np.isin(hours, range(6, 19)), 'Day', 'Night')

# Cities
cities = np.random.choice(
    ['Miami','Los Angeles','Houston','New York','Phoenix',
     'Charlotte','Dallas','Sacramento','Chicago','Atlanta'],
    n, p=[0.15,0.14,0.12,0.11,0.10,0.09,0.10,0.08,0.06,0.05])

# Lat/Long (approximate US bounding box)
lat  = np.random.uniform(25, 49, n)
lng  = np.random.uniform(-125, -65, n)

df = pd.DataFrame({
    'Hour': hours, 'Month': months, 'Day_of_Week': days_week,
    'Severity': severity, 'Weather_Condition': weather,
    'Temperature_F': np.round(temp_f, 1), 'Humidity': np.round(humidity, 1),
    'Visibility_mi': np.round(visibility, 1), 'Wind_Speed_mph': np.round(wind_speed, 1),
    'Road_Condition': road_cond, 'Junction': junction, 'Crossing': crossing,
    'Traffic_Signal': traffic_sig, 'Stop': stop,
    'Day_Night': day_night, 'City': cities,
    'Start_Lat': np.round(lat, 4), 'Start_Lng': np.round(lng, 4)
})

print("=" * 55)
print("  PRODIGY INFOTECH — DS TASK 05: Traffic Accidents")
print("=" * 55)
print(f"\nDataset Shape    : {df.shape}")
print(f"Missing Values   : {df.isnull().sum().sum()}")
print(f"\nSeverity Counts:\n{df['Severity'].value_counts().sort_index()}")
print(f"\nTop Cities:\n{df['City'].value_counts().head()}")

# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────
df['Temperature_C']  = ((df['Temperature_F'] - 32) * 5/9).round(1)
df['Time_Period']    = pd.cut(df['Hour'],
                              bins=[-1, 5, 11, 15, 19, 23],
                              labels=['Late Night (0-5)','Morning (6-11)',
                                      'Afternoon (12-15)','Evening (16-19)','Night (20-23)'])
df['Month_Name']     = pd.to_datetime(df['Month'], format='%m').dt.strftime('%b')
month_order          = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

# ─────────────────────────────────────────────
# 3. VISUALIZATIONS
# ─────────────────────────────────────────────
fig = plt.figure(figsize=(18, 14))
fig.suptitle("US Traffic Accident Analysis\nProdigy Infotech DS Task 05",
             fontsize=16, fontweight='bold', y=1.01)

# Define subplots
ax1 = fig.add_subplot(3, 3, 1)
ax2 = fig.add_subplot(3, 3, 2)
ax3 = fig.add_subplot(3, 3, 3)
ax4 = fig.add_subplot(3, 3, 4)
ax5 = fig.add_subplot(3, 3, 5)
ax6 = fig.add_subplot(3, 3, 6)
ax7 = fig.add_subplot(3, 3, 7)
ax8 = fig.add_subplot(3, 3, 8)
ax9 = fig.add_subplot(3, 3, 9)

# ── Plot 1: Accidents by Hour
hourly = df.groupby('Hour').size()
ax1.bar(hourly.index, hourly.values, color='#e67e22', edgecolor='white', linewidth=0.4)
ax1.set_title("Accidents by Hour of Day")
ax1.set_xlabel("Hour (0–23)")
ax1.set_ylabel("Count")
ax1.axvspan(6, 9, alpha=0.15, color='red', label='Morning Rush')
ax1.axvspan(16, 19, alpha=0.15, color='blue', label='Evening Rush')
ax1.legend(fontsize=8)
ax1.grid(axis='y', alpha=0.3)

# ── Plot 2: Accidents by Day of Week
day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
day_counts = df['Day_of_Week'].value_counts().reindex(day_order)
colors_day = ['#3498db' if d not in ['Saturday','Sunday'] else '#e74c3c' for d in day_order]
ax2.bar(range(7), day_counts.values, color=colors_day, edgecolor='white')
ax2.set_xticks(range(7))
ax2.set_xticklabels(['Mon','Tue','Wed','Thu','Fri','Sat','Sun'])
ax2.set_title("Accidents by Day of Week")
ax2.set_ylabel("Count")
ax2.grid(axis='y', alpha=0.3)

# ── Plot 3: Accidents by Month
month_counts = df.groupby('Month').size()
ax3.plot(month_counts.index, month_counts.values, 'o-', color='#9b59b6', linewidth=2, markersize=6)
ax3.fill_between(month_counts.index, month_counts.values, alpha=0.2, color='#9b59b6')
ax3.set_xticks(range(1, 13))
ax3.set_xticklabels(month_order, rotation=30)
ax3.set_title("Accidents by Month")
ax3.set_ylabel("Count")
ax3.grid(alpha=0.3)

# ── Plot 4: Severity Distribution
sev_colors = ['#2ecc71','#f39c12','#e67e22','#e74c3c']
sev_counts = df['Severity'].value_counts().sort_index()
bars = ax4.bar([f'Severity {s}' for s in sev_counts.index], sev_counts.values,
               color=sev_colors, edgecolor='white')
for bar, val in zip(bars, sev_counts.values):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height()+5,
             str(val), ha='center', fontsize=9, fontweight='bold')
ax4.set_title("Accident Severity Distribution")
ax4.set_ylabel("Count")
ax4.grid(axis='y', alpha=0.3)

# ── Plot 5: Weather vs Accidents
weather_counts = df['Weather_Condition'].value_counts()
ax5.barh(weather_counts.index, weather_counts.values,
         color=plt.cm.Set2(np.linspace(0, 1, len(weather_counts))), edgecolor='white')
ax5.set_title("Accidents by Weather Condition")
ax5.set_xlabel("Count")
ax5.grid(axis='x', alpha=0.3)

# ── Plot 6: Road Condition vs Severity (stacked)
road_sev = df.groupby(['Road_Condition','Severity']).size().unstack(fill_value=0)
road_sev.plot(kind='bar', stacked=True, ax=ax6,
              color=sev_colors[:len(road_sev.columns)], edgecolor='white', linewidth=0.4)
ax6.set_title("Severity by Road Condition")
ax6.set_xlabel("Road Condition")
ax6.set_ylabel("Count")
ax6.tick_params(axis='x', rotation=30)
ax6.legend(title='Severity', fontsize=8)

# ── Plot 7: Day vs Night
dn_counts = df['Day_Night'].value_counts()
ax7.pie(dn_counts, labels=dn_counts.index, autopct='%1.1f%%',
        colors=['#f1c40f','#2c3e50'], wedgeprops={'edgecolor':'white','linewidth':2}, startangle=90)
ax7.set_title("Day vs Night Accidents")

# ── Plot 8: Top Cities
top_cities = df['City'].value_counts().head(10).sort_values()
ax8.barh(top_cities.index, top_cities.values, color='#1abc9c', edgecolor='white')
ax8.set_title("Top 10 Cities by Accident Count")
ax8.set_xlabel("Count")
ax8.grid(axis='x', alpha=0.3)

# ── Plot 9: Temperature vs Severity scatter
for sev, color in zip([1,2,3,4], sev_colors):
    mask = df.Severity == sev
    ax9.scatter(df.loc[mask,'Temperature_F'], df.loc[mask,'Humidity'],
                alpha=0.3, s=12, color=color, label=f'Sev {sev}')
ax9.set_title("Temperature vs Humidity (by Severity)")
ax9.set_xlabel("Temperature (°F)")
ax9.set_ylabel("Humidity (%)")
ax9.legend(fontsize=8, markerscale=2)
ax9.grid(alpha=0.2)

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/PRODIGY_DS_05_accidents.png", dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ Plot saved: PRODIGY_DS_05_accidents.png")

# ─────────────────────────────────────────────
# 4. GEOGRAPHIC HOTSPOT MAP
# ─────────────────────────────────────────────
fig2, ax = plt.subplots(figsize=(12, 6))
sc = ax.scatter(df['Start_Lng'], df['Start_Lat'],
                c=df['Severity'], cmap='YlOrRd',
                s=10, alpha=0.5, edgecolors='none')
plt.colorbar(sc, ax=ax, label='Severity (1=Low, 4=High)')
ax.set_title("US Traffic Accident Hotspot Map\nProdigy Infotech DS Task 05",
             fontsize=13, fontweight='bold')
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_facecolor('#1a1a2e')
fig2.patch.set_facecolor('#1a1a2e')
ax.tick_params(colors='white')
ax.xaxis.label.set_color('white')
ax.yaxis.label.set_color('white')
ax.title.set_color('white')
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/PRODIGY_DS_05_hotspot_map.png", dpi=150, bbox_inches='tight',
            facecolor='#1a1a2e')
plt.show()
print("✅ Hotspot map saved: PRODIGY_DS_05_hotspot_map.png")

# ─────────────────────────────────────────────
# 5. KEY INSIGHTS
# ─────────────────────────────────────────────
print("\n--- Key Insights ---")
peak_hour    = hourly.idxmax()
peak_day     = day_counts.idxmax()
peak_weather = weather_counts.idxmax()
print(f"Peak Accident Hour     : {peak_hour}:00")
print(f"Most Accidents Day     : {peak_day}")
print(f"Most Common Weather    : {peak_weather}")
print(f"Day vs Night Accidents : {dn_counts.to_dict()}")
print(f"Avg Severity           : {df['Severity'].mean():.2f}")
print(f"Severity 3+4 Accidents : {(df.Severity >= 3).sum()} ({(df.Severity>=3).mean()*100:.1f}%)")
print("\n✅ Task 05 Complete!")
print("\n" + "="*55)
print("  ✅ ALL 5 DATA SCIENCE TASKS COMPLETED!")
print("="*55)
