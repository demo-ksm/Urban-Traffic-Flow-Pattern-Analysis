

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.float_format', '{:.2f}'.format)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.4

# ============================================================
# STEP 1: DATA COLLECTION
# ============================================================
print("=" * 55)
print("STEP 1: Data Collection")
print("=" * 55)

np.random.seed(42)
dates = pd.date_range(start='2022-01-01', end='2023-12-31 23:00:00', freq='h')
n = len(dates)

def gen_volume(dt):
    h = dt.hour
    is_weekend = dt.dayofweek >= 5
    if is_weekend:
        base = 1800 + 600 * np.sin(np.pi * (h - 10) / 12) if 8 <= h <= 20 else 400
    else:
        if 7 <= h <= 9:
            base = 4500 + (h - 7) * 400
        elif 17 <= h <= 19:
            base = 4800 + (h - 17) * 300
        elif 10 <= h <= 16:
            base = 3000 + np.random.randint(-200, 200)
        elif 0 <= h <= 5:
            base = 300 + h * 50
        else:
            base = 1500
    noise = np.random.randint(-250, 250)
    return max(50, int(base + noise))

locations = ['Highway A', 'Highway B', 'City Centre', 'Ring Road']

df_raw = pd.DataFrame({
    'date_time'     : dates,
    'traffic_volume': [gen_volume(d) for d in dates],
    'location'      : np.random.choice(locations, size=n)
})

dup_idx = np.random.choice(df_raw.index, size=150, replace=False)
df_raw = pd.concat([df_raw, df_raw.loc[dup_idx]], ignore_index=True)
miss_idx = np.random.choice(df_raw.index, size=120, replace=False)
df_raw.loc[miss_idx, 'traffic_volume'] = np.nan

print(f"Raw dataset shape : {df_raw.shape}")
print(df_raw.head())

# ============================================================
# STEP 2: DATA PREPROCESSING
# ============================================================
print("\n" + "=" * 55)
print("STEP 2: Data Preprocessing")
print("=" * 55)

df = df_raw.copy()

print("--- Before Cleaning ---")
print(f"Total records  : {len(df)}")
print(f"Duplicates     : {df.duplicated().sum()}")
print(f"Missing values : {df['traffic_volume'].isna().sum()}")

df = df.drop_duplicates()
df['date_time'] = pd.to_datetime(df['date_time'])
df = df.sort_values('date_time').reset_index(drop=True)
df['traffic_volume'] = df['traffic_volume'].fillna(df['traffic_volume'].median())
df['hour']        = df['date_time'].dt.hour
df['day_of_week'] = df['date_time'].dt.day_name()
df['month']       = df['date_time'].dt.month_name()
df['day_type']    = df['date_time'].dt.dayofweek.apply(
                        lambda x: 'Weekend' if x >= 5 else 'Weekday')

print("\n--- After Cleaning ---")
print(f"Total records  : {len(df)}")
print(f"Duplicates     : {df.duplicated().sum()}")
print(f"Missing values : {df['traffic_volume'].isna().sum()}")
print(df.head())

# ============================================================
# STEP 3: DESCRIPTIVE STATISTICAL ANALYSIS
# ============================================================
print("\n" + "=" * 55)
print("STEP 3: Descriptive Statistical Analysis")
print("=" * 55)

vol = df['traffic_volume']

stats = {
    'Mean Traffic Volume'   : vol.mean(),
    'Median Traffic Volume' : vol.median(),
    'Mode Traffic Volume'   : vol.mode()[0],
    'Std Deviation'         : vol.std(),
    'Variance'              : vol.var(),
    'Min Traffic Volume'    : vol.min(),
    'Max Traffic Volume'    : vol.max(),
    '25th Percentile (Q1)'  : vol.quantile(0.25),
    '75th Percentile (Q3)'  : vol.quantile(0.75),
    'IQR'                   : vol.quantile(0.75) - vol.quantile(0.25),
    'Skewness'              : vol.skew(),
    'Kurtosis'              : vol.kurt(),
}

stats_df = pd.DataFrame(stats.items(), columns=['Statistic', 'Value'])
stats_df['Value'] = stats_df['Value'].round(2)
print(stats_df.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(vol, bins=40, color='steelblue', edgecolor='white')
axes[0].axvline(vol.mean(),   color='red',    linestyle='--', linewidth=1.5, label=f'Mean = {vol.mean():.0f}')
axes[0].axvline(vol.median(), color='orange', linestyle='--', linewidth=1.5, label=f'Median = {vol.median():.0f}')
axes[0].set_title('Distribution of Traffic Volume', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Traffic Volume (vehicles/hour)')
axes[0].set_ylabel('Frequency')
axes[0].legend()
axes[1].boxplot(vol, vert=False, patch_artist=True,
                boxprops=dict(facecolor='lightblue'),
                medianprops=dict(color='red', linewidth=2))
axes[1].set_title('Box Plot - Traffic Volume Spread', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Traffic Volume (vehicles/hour)')
axes[1].set_yticks([])
plt.suptitle('Descriptive Analysis of Traffic Volume', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('plot1_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: plot1_distribution.png")

# ============================================================
# STEP 4: HOUR-WISE TRAFFIC ANALYSIS
# ============================================================
print("\n" + "=" * 55)
print("STEP 4: Hour-wise Traffic Analysis")
print("=" * 55)

hourly_avg = df.groupby('hour')['traffic_volume'].mean().reset_index()
hourly_avg.columns = ['Hour', 'Avg_Traffic']

peak_hour = hourly_avg.loc[hourly_avg['Avg_Traffic'].idxmax()]
low_hour  = hourly_avg.loc[hourly_avg['Avg_Traffic'].idxmin()]
print(f"Peak traffic hour : {int(peak_hour.Hour):02d}:00  -> {peak_hour.Avg_Traffic:.0f} vehicles/hr")
print(f"Lowest traffic hr : {int(low_hour.Hour):02d}:00   -> {low_hour.Avg_Traffic:.0f} vehicles/hr")

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(hourly_avg['Hour'], hourly_avg['Avg_Traffic'],
        marker='o', color='steelblue', linewidth=2.2, markersize=6)
ax.fill_between(hourly_avg['Hour'], hourly_avg['Avg_Traffic'], alpha=0.15, color='steelblue')
ax.annotate(f'Peak\n{int(peak_hour.Hour):02d}:00',
            xy=(peak_hour.Hour, peak_hour.Avg_Traffic),
            xytext=(peak_hour.Hour + 1.5, peak_hour.Avg_Traffic - 300),
            arrowprops=dict(arrowstyle='->', color='red'),
            color='red', fontsize=10, fontweight='bold')
ax.set_title('Average Traffic Volume by Hour of the Day', fontsize=14, fontweight='bold')
ax.set_xlabel('Hour of Day (24-hr)')
ax.set_ylabel('Average Traffic Volume')
ax.set_xticks(range(0, 24))
ax.set_xticklabels([f'{h:02d}:00' for h in range(24)], rotation=45, ha='right', fontsize=8)
plt.tight_layout()
plt.savefig('plot2_hourwise.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: plot2_hourwise.png")

hourly_type = df.groupby(['hour', 'day_type'])['traffic_volume'].mean().unstack()
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(hourly_type.index, hourly_type['Weekday'], marker='s', label='Weekday', color='royalblue', linewidth=2)
ax.plot(hourly_type.index, hourly_type['Weekend'], marker='o', label='Weekend', color='tomato',    linewidth=2, linestyle='--')
ax.set_title('Weekday vs Weekend Traffic - Hour-wise', fontsize=14, fontweight='bold')
ax.set_xlabel('Hour of Day')
ax.set_ylabel('Average Traffic Volume')
ax.set_xticks(range(0, 24))
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig('plot3_weekday_vs_weekend.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: plot3_weekday_vs_weekend.png")

# ============================================================
# STEP 5: DAY-WISE TRAFFIC ANALYSIS
# ============================================================
print("\n" + "=" * 55)
print("STEP 5: Day-wise Traffic Analysis")
print("=" * 55)

day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
daily_avg = (df.groupby('day_of_week')['traffic_volume']
               .mean()
               .reindex(day_order)
               .reset_index())
daily_avg.columns = ['Day', 'Avg_Traffic']
print(daily_avg.to_string(index=False))

colors = ['royalblue'] * 5 + ['tomato', 'tomato']
fig, ax = plt.subplots(figsize=(11, 5))
bars = ax.bar(daily_avg['Day'], daily_avg['Avg_Traffic'], color=colors, edgecolor='white', width=0.6)
for bar, val in zip(bars, daily_avg['Avg_Traffic']):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
            f'{val:.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
legend_elems = [Patch(facecolor='royalblue', label='Weekday'),
                Patch(facecolor='tomato',    label='Weekend')]
ax.legend(handles=legend_elems, fontsize=11)
ax.set_title('Average Traffic Volume by Day of the Week', fontsize=14, fontweight='bold')
ax.set_xlabel('Day of the Week')
ax.set_ylabel('Average Traffic Volume')
plt.tight_layout()
plt.savefig('plot4_daywise.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: plot4_daywise.png")

# ============================================================
# STEP 6: ADDITIONAL VISUALIZATIONS
# ============================================================
print("\n" + "=" * 55)
print("STEP 6: Additional Visualizations")
print("=" * 55)

month_order = ['January','February','March','April','May','June',
               'July','August','September','October','November','December']
monthly_avg = (df.groupby('month')['traffic_volume']
                 .mean()
                 .reindex(month_order)
                 .reset_index())
monthly_avg.columns = ['Month', 'Avg_Traffic']

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(monthly_avg['Month'], monthly_avg['Avg_Traffic'],
        marker='D', color='darkorange', linewidth=2.2, markersize=7)
ax.fill_between(range(len(monthly_avg)), monthly_avg['Avg_Traffic'], alpha=0.15, color='darkorange')
ax.set_title('Monthly Average Traffic Volume Trend', fontsize=14, fontweight='bold')
ax.set_xlabel('Month')
ax.set_ylabel('Average Traffic Volume')
ax.set_xticks(range(len(monthly_avg)))
ax.set_xticklabels(monthly_avg['Month'], rotation=30, ha='right')
plt.tight_layout()
plt.savefig('plot5_monthly_trend.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: plot5_monthly_trend.png")

pivot = df.pivot_table(values='traffic_volume', index='day_of_week', columns='hour', aggfunc='mean')
pivot = pivot.reindex(day_order)
fig, ax = plt.subplots(figsize=(16, 5))
im = ax.imshow(pivot.values, aspect='auto', cmap='YlOrRd')
ax.set_yticks(range(len(day_order)))
ax.set_yticklabels(day_order, fontsize=10)
ax.set_xticks(range(24))
ax.set_xticklabels([f'{h:02d}:00' for h in range(24)], rotation=45, ha='right', fontsize=8)
plt.colorbar(im, ax=ax, label='Avg Traffic Volume')
ax.set_title('Traffic Volume Heatmap - Hour of Day x Day of Week', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plot6_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: plot6_heatmap.png")

loc_avg = df.groupby('location')['traffic_volume'].mean().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.barh(loc_avg.index, loc_avg.values,
               color=['#2196F3','#FF5722','#4CAF50','#9C27B0'], edgecolor='white')
for bar, val in zip(bars, loc_avg.values):
    ax.text(val + 20, bar.get_y() + bar.get_height() / 2,
            f'{val:.0f}', va='center', fontsize=10, fontweight='bold')
ax.set_title('Average Traffic Volume by Location', fontsize=13, fontweight='bold')
ax.set_xlabel('Average Traffic Volume')
plt.tight_layout()
plt.savefig('plot7_location.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: plot7_location.png")

# ============================================================
# STEP 7: CONCLUSION
# ============================================================
print("\n" + "=" * 55)
print("STEP 7: Insights & Conclusion")
print("=" * 55)
print("""

""")