# Import Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load the Dataset
data = pd.read_csv('Global_Space_Exploration_Dataset.csv')

# Data Inspection & Cleaning

# Summary Statistics
print("Summary Statistics:")
print(data.describe())

# Data Types and Memory Usage
print("\nData Types:")
print(data.dtypes)
print("\nMemory Usage:")
print(data.memory_usage(deep=True))

# Null Values Before Cleaning
print("\nNull Values Before Cleaning:")
print(data.isnull().sum())

# Drop NA rows and reset index
data_cleaned = data.dropna().reset_index(drop=True)

# Null Values After Cleaning
print("\nNull Values After Cleaning:")
print(data_cleaned.isnull().sum())

# Statistical Tests

# Chi-Square Test (Country vs Mission Type)
contingency_table = pd.crosstab(data_cleaned['Country'], data_cleaned['Mission Type'])
chi2, p_value = stats.chi2_contingency(contingency_table)[:2]
print("\nChi-square Test (Country vs Mission Type):")
print(f"Chi2 Statistic: {chi2:.4f}, P-value: {p_value:.4f}")

# Chi-Square Test (Technology vs Environmental Impact)
contingency_table_tech_env = pd.crosstab(data_cleaned['Technology Used'], data_cleaned['Environmental Impact'])
chi2_tech_env, p_value_tech_env = stats.chi2_contingency(contingency_table_tech_env)[:2]
print("\nChi-square Test (Technology vs Environmental Impact):")
print(f"Chi2 Statistic: {chi2_tech_env:.4f}, P-value: {p_value_tech_env:.4f}")

# Z-Test (Proportion of Success Rate > 70%)
mean_sr = data_cleaned['Success Rate (%)'].mean()
n = len(data_cleaned)
p0 = 0.7
z_score = (mean_sr - p0) / np.sqrt(p0 * (1 - p0) / n)
print("\nZ-Test (Success Rate > 70%):")
print(f"Z-Score: {z_score:.4f}")

# T-Test (Compare Duration: Manned vs Unmanned)
duration_manned = data_cleaned[data_cleaned['Mission Type'] == 'Manned']['Duration (in Days)']
duration_unmanned = data_cleaned[data_cleaned['Mission Type'] == 'Unmanned']['Duration (in Days)']
t_stat_duration, t_p_duration = stats.ttest_ind(duration_manned, duration_unmanned)
print("\nT-Test (Manned vs Unmanned Duration):")
print(f"T-Statistic: {t_stat_duration:.4f}, P-value: {t_p_duration:.4f}")

# A/B Test (Average Success Rate by Technology)
print("\nA/B Test: Success Rate by Technology Used")
print(data_cleaned.groupby("Technology Used")["Success Rate (%)"].mean())

# Dashboard Visualizations 

# Bar Plot - Top 5 Countries by Average Budget
top_countries = data_cleaned.groupby('Country')['Budget (in Billion $)'].mean().nlargest(5).reset_index()
n_colors = len(top_countries['Country'])
palette = sns.color_palette("magma", n_colors=n_colors)
sns.barplot(data=top_countries, x='Country', y='Budget (in Billion $)', hue='Country', palette=palette, legend=False)
plt.title('Top 5 Countries by Average Mission Budget', fontsize=10)
plt.xticks(rotation=45, fontsize=8)
plt.xlabel('', fontsize=8)
plt.ylabel('Avg Budget (Billion $)', fontsize=8)
plt.tight_layout()
plt.show()

# Pie Chart - Proportion of Mission Types by Year (Recent Years)
recent_data = data_cleaned[data_cleaned['Year'] >= 2020]
mission_type_prop = recent_data['Mission Type'].value_counts()
n_colors = len(mission_type_prop)
palette = sns.color_palette("plasma", n_colors=n_colors)
plt.pie(mission_type_prop, labels=mission_type_prop.index, autopct='%1.1f%%', startangle=90, colors=palette)
plt.title('Mission Type Distribution (2020-2025)', fontsize=10)
plt.tight_layout()
plt.show()

# Swarm Plot - Success Rate by Technology (Sampled for clarity)
sample_data = data_cleaned.sample(100)
n_colors = len(sample_data['Technology Used'].unique())
palette = sns.color_palette("inferno", n_colors=n_colors)
sns.swarmplot(data=sample_data, x='Technology Used', y='Success Rate (%)', hue='Technology Used', palette=palette, size=4, legend=False)
plt.title('Success Rate Variability by Technology', fontsize=10)
plt.xticks(rotation=45, fontsize=8)
plt.xlabel('', fontsize=8)
plt.ylabel('Success Rate (%)', fontsize=8)
plt.tight_layout()
plt.show()

# Area Plot - Cumulative Budget Over Years
budget_by_year = data_cleaned.groupby('Year')['Budget (in Billion $)'].sum().cumsum()
palette = sns.color_palette("Blues", n_colors=1)  # Single color for area plot
plt.fill_between(budget_by_year.index, budget_by_year, color=palette[0], alpha=0.4)
plt.plot(budget_by_year.index, budget_by_year, color=palette[0], linewidth=2)
plt.title('Cumulative Budget Over Time', fontsize=10)
plt.xlabel('Year', fontsize=8)
plt.ylabel('Cumulative Budget (Billion $)', fontsize=8)
plt.xticks(rotation=45, fontsize=8)
plt.tight_layout()
plt.show()

# Heatmap - Environmental Impact vs Success Rate by Country
pivot_env_success = data_cleaned.pivot_table(values='Success Rate (%)', index='Country', columns='Environmental Impact', aggfunc='mean')
sns.heatmap(pivot_env_success, annot=True, fmt='.1f', cmap='coolwarm', cbar_kws={'label': 'Avg Success Rate (%)'})
plt.title('Environmental Impact vs Success Rate by Country', fontsize=10)
plt.tight_layout()
plt.show()

# Boxen Plot - Budget Distribution by Environmental Impact
n_colors = len(data_cleaned['Environmental Impact'].unique())
palette = sns.color_palette("Greens", n_colors=n_colors)
sns.boxenplot(data=data_cleaned, x='Environmental Impact', y='Budget (in Billion $)', hue='Environmental Impact', palette=palette, legend=False)
plt.title('Budget Distribution by Environmental Impact', fontsize=10)
plt.xticks(rotation=45, fontsize=8)
plt.xlabel('', fontsize=8)
plt.ylabel('Budget (Billion $)', fontsize=8)
plt.tight_layout()
plt.show()

# Line Plot - Success Rate vs Year by Mission Type
n_colors = len(data_cleaned['Mission Type'].unique())
palette = sns.color_palette("Purples", n_colors=n_colors)
sns.lineplot(data=data_cleaned, x='Year', y='Success Rate (%)', hue='Mission Type', marker='o', palette=palette)
plt.title('Success Rate Trends Over Years by Mission Type', fontsize=10)
plt.xlabel('Year', fontsize=8)
plt.ylabel('Success Rate (%)', fontsize=8)
plt.legend(title='Mission Type', title_fontsize=8, fontsize=6)
plt.tight_layout()
plt.show()

# Bar Plot - Top 5 Collaborating Countries
collabs = data_cleaned['Collaborating Countries'].str.split(', ').explode().value_counts().head(5)
n_colors = len(collabs)
palette = sns.color_palette("Oranges", n_colors=n_colors)
sns.barplot(x=collabs.values, y=collabs.index, hue=collabs.index, palette=palette, legend=False)
plt.title('Top 5 Collaborating Countries', fontsize=10)
plt.xlabel('Number of Collaborations', fontsize=8)
plt.ylabel('', fontsize=8)
plt.tight_layout()
plt.show()

# Stacked Bar Plot - Mission Count by Country and Satellite Type
mission_counts = data_cleaned.groupby(['Country', 'Satellite Type']).size().unstack().fillna(0)
n_colors = len(mission_counts.columns)
palette = sns.color_palette("Reds", n_colors=n_colors)
mission_counts.plot(kind='bar', stacked=True, color=palette, figsize=(8, 5))
plt.title('Mission Count by Country and Satellite Type', fontsize=10)
plt.xticks(rotation=45, fontsize=8)
plt.xlabel('', fontsize=8)
plt.ylabel('Number of Missions', fontsize=8)
plt.legend(title='Satellite Type', title_fontsize=8, fontsize=6)
plt.tight_layout()
plt.show()

# Hexbin Plot - Budget vs Duration
plt.hexbin(data_cleaned['Duration (in Days)'], data_cleaned['Budget (in Billion $)'], gridsize=20, cmap='viridis', mincnt=1)
plt.colorbar(label='Count')
plt.title('Budget vs Mission Duration (Hexbin)', fontsize=10)
plt.xlabel('Duration (Days)', fontsize=8)
plt.ylabel('Budget (Billion $)', fontsize=8)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()
plt.show()

# Bar Plot - Average Duration by Country
n_colors = len(data_cleaned['Country'].unique())
palette = sns.color_palette("crest", n_colors=n_colors)
sns.barplot(data=data_cleaned, x='Country', y='Duration (in Days)', hue='Country', palette=palette, legend=False)
plt.title('Average Mission Duration by Country', fontsize=10)
plt.xticks(rotation=45, fontsize=8)
plt.xlabel('', fontsize=8)
plt.ylabel('Duration (Days)', fontsize=8)
plt.tight_layout()
plt.show()

# Violin Plot - Success Rate by Satellite Type
n_colors = len(data_cleaned['Satellite Type'].unique())
palette = sns.color_palette("flare", n_colors=n_colors)
sns.violinplot(data=data_cleaned, x='Satellite Type', y='Success Rate (%)', hue='Satellite Type', palette=palette, legend=False)
plt.title('Success Rate Distribution by Satellite Type', fontsize=10)
plt.xticks(rotation=45, fontsize=8)
plt.xlabel('', fontsize=8)
plt.ylabel('Success Rate (%)', fontsize=8)
plt.tight_layout()
plt.show()

# KDE Plot - Budget Distribution by Mission Type
n_colors = len(data_cleaned['Mission Type'].unique())
palette = sns.color_palette("rocket", n_colors=n_colors)
sns.kdeplot(data=data_cleaned[data_cleaned['Mission Type'] == 'Manned']['Budget (in Billion $)'], label='Manned', color=palette[0])
sns.kdeplot(data=data_cleaned[data_cleaned['Mission Type'] == 'Unmanned']['Budget (in Billion $)'], label='Unmanned', color=palette[1])
plt.title('Budget Distribution by Mission Type (KDE)', fontsize=10)
plt.xlabel('Budget (Billion $)', fontsize=8)
plt.ylabel('Density', fontsize=8)
plt.legend(title='Mission Type', title_fontsize=8, fontsize=6)
plt.tight_layout()
plt.show()

# Scatter Plot - Success Rate vs Duration
n_colors = len(data_cleaned['Mission Type'].unique())
palette = sns.color_palette("mako", n_colors=n_colors)
sns.scatterplot(data=data_cleaned, x='Duration (in Days)', y='Success Rate (%)', hue='Mission Type', style='Mission Type', s=120, palette=palette)
plt.title('Success Rate vs Duration', fontsize=10)
plt.xlabel('Duration (Days)', fontsize=8)
plt.ylabel('Success Rate (%)', fontsize=8)
plt.legend(title='Mission Type', title_fontsize=8, fontsize=6)
plt.tight_layout()
plt.show()

# Heatmap - Success Rate by Country and Year
pivot = data_cleaned.pivot_table(values='Success Rate (%)', index='Country', columns='Year', aggfunc='mean')
sns.heatmap(pivot, annot=True, fmt='.1f', cmap='cividis')
plt.title('Success Rate by Country and Year', fontsize=10)
plt.xlabel('Year', fontsize=8)
plt.ylabel('Country', fontsize=8)
plt.tight_layout()
plt.show()

# Bar Plot - Average Success Rate by Technology (A/B Test Visualization)
tech_success = data_cleaned.groupby('Technology Used')['Success Rate (%)'].mean().reset_index()
n_colors = len(tech_success['Technology Used'])
palette = sns.color_palette("vlag", n_colors=n_colors)
sns.barplot(data=tech_success, x='Technology Used', y='Success Rate (%)', hue='Technology Used', palette=palette, legend=False)
plt.title('Average Success Rate by Technology Used', fontsize=10)
plt.xticks(rotation=45, fontsize=8)
plt.xlabel('', fontsize=8)
plt.ylabel('Avg Success Rate (%)', fontsize=8)
plt.tight_layout()
plt.show()

# Box Plot - Budget by Launch Site (Top 5 Sites)
top_launch_sites = data_cleaned['Launch Site'].value_counts().nlargest(5).index
n_colors = len(top_launch_sites)
palette = sns.color_palette("deep", n_colors=n_colors)
sns.boxplot(data=data_cleaned[data_cleaned['Launch Site'].isin(top_launch_sites)], x='Launch Site', y='Budget (in Billion $)', hue='Launch Site', palette=palette, legend=False)
plt.title('Budget Distribution by Top 5 Launch Sites', fontsize=10)
plt.xticks(rotation=45, fontsize=8)
plt.xlabel('', fontsize=8)
plt.ylabel('Budget (Billion $)', fontsize=8)
plt.tight_layout()
plt.show()

# Stacked Area Plot - Mission Count by Year and Technology
tech_counts = data_cleaned.groupby(['Year', 'Technology Used']).size().unstack().fillna(0)
n_colors = len(tech_counts.columns)
palette = sns.color_palette("muted", n_colors=n_colors)
tech_counts.plot(kind='area', stacked=True, color=palette, alpha=0.7, figsize=(8, 5))
plt.title('Mission Count by Year and Technology', fontsize=10)
plt.xlabel('Year', fontsize=8)
plt.ylabel('Number of Missions', fontsize=8)
plt.legend(title='Technology Used', title_fontsize=8, fontsize=6)
plt.tight_layout()
plt.show()

# Facet Grid - Success Rate by Mission Type and Environmental Impact
g = sns.FacetGrid(data_cleaned, col='Mission Type', row='Environmental Impact', height=3, aspect=1.5)
g.map(sns.histplot, 'Success Rate (%)', bins=20, color=sns.color_palette("dark")[0])
g.set_titles(col_template="{col_name}", row_template="{row_name}", size=8)
g.set_axis_labels('Success Rate (%)', 'Count', fontsize=8)
plt.suptitle('Success Rate Distribution by Mission Type and Environmental Impact', fontsize=10, y=1.05)
g.tight_layout()
plt.show()

# Clustered Bar Plot - Average Budget by Technology and Satellite Type
budget_tech_sat = data_cleaned.groupby(['Technology Used', 'Satellite Type'])['Budget (in Billion $)'].mean().reset_index()
sns.catplot(data=budget_tech_sat, x='Technology Used', y='Budget (in Billion $)', hue='Satellite Type', kind='bar', palette=sns.color_palette("pastel", n_colors=5), height=5, aspect=1.6)
plt.title('Average Budget by Technology and Satellite Type', fontsize=10)
plt.xticks(rotation=45, fontsize=8)
plt.xlabel('', fontsize=8)
plt.ylabel('Avg Budget (Billion $)', fontsize=8)
plt.tight_layout()
plt.show()


