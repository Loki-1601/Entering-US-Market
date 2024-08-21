import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import states as states
sns.set()
from scipy import stats
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.cluster.hierarchy import fcluster
import geopandas as gpd
import plotly.express as px


# Load the data
file_path = '/Users/lokeshwaripotluri/Desktop/US State Data.xlsx'
data = pd.read_excel(file_path, index_col=0)

# Fill NaN values with 0
data.fillna(0, inplace=True)

# Transpose the dataframe
data = data.transpose()

# Drop columns that are completely NaN
data = data.dropna(axis=1, how='all')

# Drop rows that are completely NaN
data = data.dropna(axis=0, how='all')

# Remove any columns with 'NaN' in their names (caused by transposing)
# Keep columns that have a proper name
data.columns = data.columns.str.strip()

# Remove columns where the column name is NaN (empty string after stripping)
data = data.loc[:, ~data.columns.str.strip().isnull()]

# Rename columns that may have been affected during transpose
data.columns = data.columns.str.strip()

# Drop duplicate columns (if they exist)
data = data.loc[:, ~data.columns.duplicated()]

print(f"Number of states after cleaning: {len(data.index)}")
print(data.index.tolist())

# Verifying the cleaned data
print("Cleaned Data Shape:", data.shape)
print("Column Names:", data.columns)

# Identify known state names
known_states = {
    'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 'Delaware',
    'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky',
    'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi',
    'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico',
    'New York', 'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania',
    'Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont',
    'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming'
}


# Convert the set to a list for indexing
data = data.loc[list(known_states.intersection(data.index))]

# Display the updated DataFrame information
print(f"Cleaned Data Shape: {data.shape}")
print(f"Final Columns: {data.columns}")
print(f"Missing States: {set(known_states) - set(data.index)}")



# 1. Population growth trends from 2010 to 2021 and Household sizes in 2020
# Calculating Average Household Size
population_growth = data['Population Growth or Decline 2010 to 2021']
average_household_size = data['Total Population 2021'] / data['Households 2020']

# Plotting the scatterplot
plt.figure(figsize=(12, 8))
sns.scatterplot(x=population_growth, y=average_household_size)

# Adding a regression line to visualize the trend
sns.regplot(x=population_growth, y=average_household_size, scatter=False, color='red')

# Adding labels and title
plt.title('Population Growth (2010 to 2021) vs Average Household Size (2020) Across States')
plt.xlabel('Population Growth or Decline (2010 to 2021)')
plt.ylabel('Average Household Size in 2020')

# Show the plot
plt.show()


# 2. Compare the total population of each state in 2021 using a horizontal bar chart
# Sorting the data
sorted_population = data['Total Population 2021'].sort_values()

# Plotting
plt.figure(figsize=(12, 10))
sns.barplot(x=sorted_population, y=sorted_population.index, palette="viridis")

# Adding labels and title
plt.title('Total Population of Each State in 2021')
plt.xlabel('Population')
plt.ylabel('States')

# Show the plot
plt.show()


# 3. Create a line chart to compare the population growth rate from 2010 to 2021 for each state
# Sorting the data
sorted_growth_rate = data['Population Growth or Decline 2010 to 2021'].sort_values()

# Plotting
plt.figure(figsize=(14, 8))
sns.lineplot(x=sorted_growth_rate.index, y=sorted_growth_rate.values, marker='o')

# Adding labels and title
plt.title('Population Growth Rate (2010 to 2021) for Each State')
plt.xlabel('States')
plt.ylabel('Growth Rate (2010 to 2021)')
plt.xticks(rotation=90)  # Rotate the state names for better readability

# Show the plot
plt.show()


# 4. Percentage change for population every 10 years
percentage_change = (data['Total Population 2021'] - data['Population Growth or Decline 2010 to 2021'] * data['Total Population 2021']) / data['Population Growth or Decline 2010 to 2021'] * 100

# Sorting the data
sorted_percentage_change = percentage_change.sort_values()

# Plotting
plt.figure(figsize=(14, 8))
sns.barplot(x=sorted_percentage_change.index, y=sorted_percentage_change.values, palette="coolwarm")

# Adding labels and title
plt.title('Percentage Change in Population (2010 to 2021) for Each State')
plt.xlabel('States')
plt.ylabel('Percentage Change')
plt.xticks(rotation=90)

# Show the plot
plt.show()


# 5. Analyze the per capita income, median household income, and poverty rates across different states

# Extract relevant columns
per_capita_income = data['Per Capita Personal Income 2021']
median_household_income = data['Median Household Income 2020'].squeeze()
poverty_rate = data['Poverty Rate 2020']

# Box plot creation
plt.figure(figsize=(16, 8))

# Box plot for Per Capita Income
plt.subplot(1, 3, 1)
sns.boxplot(y=per_capita_income)
plt.title('Per Capita Income Across States')
plt.ylabel('Per Capita Income')

# Box plot for Median Household Income
plt.subplot(1, 3, 2)
sns.boxplot(y=median_household_income)
plt.title('Median Household Income Across States')
plt.ylabel('Median Household Income')

# Box plot for Poverty Rate
plt.subplot(1, 3, 3)
sns.boxplot(y=poverty_rate)
plt.title('Poverty Rate Across States')
plt.ylabel('Poverty Rate')

plt.tight_layout()
plt.show()

# Create subplots for histograms
plt.figure(figsize=(16, 8))

# Histogram for Per Capita Income
plt.subplot(1, 3, 1)
sns.histplot(per_capita_income, kde=True)
plt.title('Distribution of Per Capita Income')
plt.xlabel('Per Capita Income')

# Histogram for Median Household Income
plt.subplot(1, 3, 2)
sns.histplot(median_household_income, kde=True)
plt.title('Distribution of Median Household Income')
plt.xlabel('Median Household Income')

# Histogram for Poverty Rate
plt.subplot(1, 3, 3)
sns.histplot(poverty_rate, kde=True)
plt.title('Distribution of Poverty Rate')
plt.xlabel('Poverty Rate')

plt.tight_layout()
plt.show()


# 7. Identify the percentage of adults 25+ with a high school diploma or more by state in 2020

# Extract the relevant data
high_school_diploma_rate = data['High School Diploma or More - Pct. of Adults 25+ 2020']
bachelors_degree_rate = data["Bachelor's Degree"]

# Sortng both datasets
sorted_high_school_diploma_rate = high_school_diploma_rate.sort_values()
sorted_bachelors_degree_rate = bachelors_degree_rate.loc[sorted_high_school_diploma_rate.index]

# Highlighting the top 5 states
highlight_top_5 = sorted_bachelors_degree_rate.nlargest(5).index

# Plotting the bar charts
plt.figure(figsize=(14, 12))

# Bar chart for High School Diploma or More
plt.subplot(2, 1, 1)
sns.barplot(x=sorted_high_school_diploma_rate, y=sorted_high_school_diploma_rate.index, palette="Blues_d")
plt.title('Percentage of Adults 25+ with a High School Diploma or More by State in 2020')
plt.xlabel('Percentage (%)')
plt.ylabel('States')
plt.xticks(rotation=45, ha='right')

# Bar chart for Bachelor's Degree or More
plt.subplot(2, 1, 2)
colors = ['darkblue' if state in highlight_top_5 else 'lightblue' for state in sorted_bachelors_degree_rate.index]
sns.barplot(x=sorted_bachelors_degree_rate, y=sorted_bachelors_degree_rate.index, palette=colors)
plt.title("Percentage of Adults 25+ with a Bachelor's Degree or More by State in 2020")
plt.xlabel('Percentage (%)')
plt.ylabel('States')
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.show()


# 8. Identify patterns in infrastructure and development that correlate with economic health, demographic makeup, and education levels, highlighting states with similar characteristics

# Select relevant columns
selected_data = data[['Per Capita Personal Income 2021', 'Poverty Rate 2020',
                      'High School Diploma or More - Pct. of Adults 25+ 2020',
                      "Bachelor's Degree"]]

# Normalize the data using min-max normalization
normalized_data = (selected_data - selected_data.min()) / (selected_data.max() - selected_data.min())

# Sort the data by Per Capita Income
sorted_data = normalized_data.sort_values(by='Per Capita Personal Income 2021')

# Plotting
plt.figure(figsize=(14, 8))

# Bar plot for Economic Health (Per Capita Income and Poverty Rate)
bar_width = 0.35
indices = np.arange(len(sorted_data))
plt.bar(indices, sorted_data['Per Capita Personal Income 2021'], bar_width, label='Per Capita Income')
plt.bar(indices + bar_width, sorted_data['Poverty Rate 2020'], bar_width, label='Poverty Rate')

# Line plot for Education Levels (High School Diploma and Bachelor's Degree)
plt.plot(indices, sorted_data['High School Diploma or More - Pct. of Adults 25+ 2020'], color='red', marker='o', label='High School Diploma Rate')
plt.plot(indices, sorted_data["Bachelor's Degree"], color='green', marker='o', label="Bachelor's Degree Rate")

# Adding labels and title
plt.xlabel('States')
plt.ylabel('Normalized Values')
plt.title('Correlation of Economic Health, Demographic Makeup, and Education Levels by State')
plt.xticks(indices + bar_width / 2, sorted_data.index, rotation=90)
plt.legend()

plt.tight_layout()
plt.show()


# 9. Measure the economic disparity by plotting per capita personal income against the poverty rate for each state

# Calculate the national average for poverty rate and education levels
national_avg_poverty_rate = poverty_rate.mean()
national_avg_high_school = data['High School Diploma or More - Pct. of Adults 25+ 2020'].mean()
national_avg_bachelors = data["Bachelor's Degree"].mean()

# Identify states with a poverty rate higher than the national average and above-average education levels
selected_states = data[
    (poverty_rate > national_avg_poverty_rate) &
    (data['High School Diploma or More - Pct. of Adults 25+ 2020'] > national_avg_high_school) &
    (data["Bachelor's Degree"] > national_avg_bachelors)
]

# Extract relevant columns
selected_states_poverty = selected_states['Poverty Rate 2020']
selected_states_high_school = selected_states['High School Diploma or More - Pct. of Adults 25+ 2020']
selected_states_bachelors = selected_states["Bachelor's Degree"]

# Normalize the data
normalized_poverty = selected_states_poverty / selected_states_poverty.max()
normalized_high_school = selected_states_high_school / selected_states_high_school.max()
normalized_bachelors = selected_states_bachelors / selected_states_bachelors.max()

# Create a figure
plt.figure(figsize=(12, 8))

# Plot all variables as line plots with normalized data
plt.plot(selected_states.index, normalized_poverty, color='red', marker='o', label='Poverty Rate (Normalized)')
plt.plot(selected_states.index, normalized_high_school, color='blue', marker='o', label='High School Diploma Rate (Normalized)')
plt.plot(selected_states.index, normalized_bachelors, color='green', marker='o', label="Bachelor's Degree Rate (Normalized)")

# Adding labels and title
plt.title('States with Poverty Rate Higher than National Average and Above-Average Education Levels')
plt.xlabel('States')
plt.ylabel('Normalized Values')
plt.xticks(rotation=45, ha='right')
plt.legend()

plt.tight_layout()
plt.show()


# 10. Plot the percentage of all jobs that are in manufacturing by state in 2021, highlighting the top and bottom 5 states

manufacturing_jobs_pct = data['Manufacturing - Pct. All Jobs in County 2021']

# Sort the data
sorted_manufacturing_jobs_pct = manufacturing_jobs_pct.sort_values()

# Identify the top 5 and bottom 5 states
top_5_states = sorted_manufacturing_jobs_pct.tail(5).index
bottom_5_states = sorted_manufacturing_jobs_pct.head(5).index

# Highlight the top and bottom 5 states in the plot
colors = ['red' if state in top_5_states else 'blue' if state in bottom_5_states else 'lightgray' for state in sorted_manufacturing_jobs_pct.index]

# Plotting
plt.figure(figsize=(14, 8))
sns.barplot(x=sorted_manufacturing_jobs_pct.index, y=sorted_manufacturing_jobs_pct.values, palette=colors)

# Adding labels and title
plt.title('Percentage of All Jobs in Manufacturing by State in 2021')
plt.xlabel('States')
plt.ylabel('Percentage of Manufacturing Jobs (%)')
plt.xticks(rotation=90)
plt.tight_layout()

# Show the plot
plt.show()


# 11. Compare the average wage in the manufacturing sector to the overall average wage per job for each state
manufacturing_avg_wage = data['Manufacturing - Avg Wage per Job 2021']
overall_avg_wage = data['Avg Wage per Job 2021']

# Sorting the data
sorted_states = manufacturing_avg_wage.sort_values().index
sorted_manufacturing_avg_wage = manufacturing_avg_wage.loc[sorted_states]
sorted_overall_avg_wage = overall_avg_wage.loc[sorted_states]

# Creating the plot
fig, ax1 = plt.subplots(figsize=(14, 8))

# Plotting manufacturing average wage on the primary y-axis
ax1.bar(sorted_states, sorted_manufacturing_avg_wage, color='blue', alpha=0.6, label='Manufacturing Avg Wage')
ax1.set_xlabel('States')
ax1.set_ylabel('Manufacturing Avg Wage ($)', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_xticklabels(sorted_states, rotation=90)

# Create a secondary y-axis to plot the overall average wage
ax2 = ax1.twinx()
ax2.plot(sorted_states, sorted_overall_avg_wage, color='green', marker='o', label='Overall Avg Wage')
ax2.set_ylabel('Overall Avg Wage ($)', color='green')
ax2.tick_params(axis='y', labelcolor='green')

# Adding the title and legend
plt.title('Comparison of Manufacturing Avg Wage vs Overall Avg Wage per Job by State in 2021')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.tight_layout()
plt.show()


# 12. Replicate the above analysis for the Healthcare(Social Assist), Finance(and Insurance) and Transportation(and Warehousing) sector
# Extract the relevant data for the different sectors' wages and the overall average wage
healthcare_avg_wage = data['Health Care, Social Assist. - Avg Wage per Job 2021']
finance_avg_wage = data['Finance and Insurance - Avg Wage per Job 2021']
transportation_avg_wage = data['Transportation and Warehousing - Avg Wage per Job 2021']
overall_avg_wage = data['Avg Wage per Job 2021']

# Sorting the data
sorted_states = overall_avg_wage.sort_values().index
sorted_healthcare_avg_wage = healthcare_avg_wage.loc[sorted_states]
sorted_finance_avg_wage = finance_avg_wage.loc[sorted_states]
sorted_transportation_avg_wage = transportation_avg_wage.loc[sorted_states]
sorted_overall_avg_wage = overall_avg_wage.loc[sorted_states]

# Create the dual-axis plots for each sector

# Healthcare Sector
fig, ax1 = plt.subplots(figsize=(14, 8))
ax1.bar(sorted_states, sorted_healthcare_avg_wage, color='purple', alpha=0.6, label='Healthcare Avg Wage')
ax1.set_xlabel('States')
ax1.set_ylabel('Healthcare Avg Wage ($)', color='purple')
ax1.tick_params(axis='y', labelcolor='purple')
ax1.set_xticklabels(sorted_states, rotation=90)

ax2 = ax1.twinx()
ax2.plot(sorted_states, sorted_overall_avg_wage, color='green', marker='o', label='Overall Avg Wage')
ax2.set_ylabel('Overall Avg Wage ($)', color='green')
ax2.tick_params(axis='y', labelcolor='green')

plt.title('Comparison of Healthcare Avg Wage vs Overall Avg Wage per Job by State in 2021')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.tight_layout()
plt.show()

# Finance Sector
fig, ax1 = plt.subplots(figsize=(14, 8))
ax1.bar(sorted_states, sorted_finance_avg_wage, color='orange', alpha=0.6, label='Finance Avg Wage')
ax1.set_xlabel('States')
ax1.set_ylabel('Finance Avg Wage ($)', color='orange')
ax1.tick_params(axis='y', labelcolor='orange')
ax1.set_xticklabels(sorted_states, rotation=90)

ax2 = ax1.twinx()
ax2.plot(sorted_states, sorted_overall_avg_wage, color='green', marker='o', label='Overall Avg Wage')
ax2.set_ylabel('Overall Avg Wage ($)', color='green')
ax2.tick_params(axis='y', labelcolor='green')

plt.title('Comparison of Finance Avg Wage vs Overall Avg Wage per Job by State in 2021')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.tight_layout()
plt.show()

# Transportation Sector
fig, ax1 = plt.subplots(figsize=(14, 8))
ax1.bar(sorted_states, sorted_transportation_avg_wage, color='red', alpha=0.6, label='Transportation Avg Wage')
ax1.set_xlabel('States')
ax1.set_ylabel('Transportation Avg Wage ($)', color='red')
ax1.tick_params(axis='y', labelcolor='red')
ax1.set_xticklabels(sorted_states, rotation=90)

ax2 = ax1.twinx()
ax2.plot(sorted_states, sorted_overall_avg_wage, color='green', marker='o', label='Overall Avg Wage')
ax2.set_ylabel('Overall Avg Wage ($)', color='green')
ax2.tick_params(axis='y', labelcolor='green')

plt.title('Comparison of Transportation Avg Wage vs Overall Avg Wage per Job by State in 2021')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.tight_layout()
plt.show()


#Statistics and Probability
# 1. Analyze the correlation between covered employment in 2021 and average wage per job by state.

# Extract the relevant columns
covered_employment = data['Covered Employment 2021']
average_wage = data['Avg Wage per Job 2021']

# Calculate the correlation coefficient
correlation_coefficient = np.corrcoef(covered_employment, average_wage)[0, 1]
print(f"Correlation Coefficient: {correlation_coefficient:.2f}")

# Visualize the relationship using a scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x=covered_employment, y=average_wage)

plt.title(f'Scatter Plot of Covered Employment vs. Average Wage per Job\nCorrelation Coefficient: {correlation_coefficient:.2f}')
plt.xlabel('Covered Employment 2021')
plt.ylabel('Average Wage per Job 2021')
plt.show()
print(f"Number of states after cleaning 12: {len(data.index)}")
print(data.index.tolist())

# 2. Investigate the relationship between education level and median household income by state

# Extract relevant columns for analysis
high_school_diploma_rate = data['High School Diploma or More - Pct. of Adults 25+ 2020']
bachelors_degree_rate = data['Bachelor\'s Degree or More - Pct. of Adults 25+ 2020']
median_household_income = data['Median Household Income 2020']

# Drop any NaN values and ensure matching lengths
valid_indices = high_school_diploma_rate.dropna().index.intersection(
    bachelors_degree_rate.dropna().index).intersection(
    median_household_income.dropna().index)

# Filter the data to only include valid indices
high_school_diploma_rate = high_school_diploma_rate.loc[valid_indices]
bachelors_degree_rate = bachelors_degree_rate.loc[valid_indices]
median_household_income = median_household_income.loc[valid_indices]

# Print lengths after filtering
print(f"Length after filtering - High School Diploma Rate: {len(high_school_diploma_rate)}")
print(f"Length after filtering - Bachelor's Degree Rate: {len(bachelors_degree_rate)}")
print(f"Length after filtering - Median Household Income: {len(median_household_income)}")

# Calculate correlation coefficients
if len(high_school_diploma_rate) == len(median_household_income):
    correlation_high_school = np.corrcoef(high_school_diploma_rate, median_household_income)[0, 1]
    correlation_bachelors = np.corrcoef(bachelors_degree_rate, median_household_income)[0, 1]

    print(f"Correlation between High School Diploma Rate and Median Household Income: {correlation_high_school:.2f}")
    print(f"Correlation between Bachelor's Degree Rate and Median Household Income: {correlation_bachelors:.2f}")
else:
    print("The lengths of the arrays do not match. Please inspect the data.")

# Plotting the relationships between education levels and median household income
plt.figure(figsize=(14, 6))
print(f"Number of states after cleaning: {len(data.index)}")
print(data.index.tolist())  # This should list all states

# Scatter plot for High School Diploma Rate vs. Median Household Income
plt.subplot(1, 2, 1)
sns.scatterplot(x=high_school_diploma_rate, y=median_household_income)
sns.regplot(x=high_school_diploma_rate, y=median_household_income, scatter=False, color='blue')
plt.title(f'High School Diploma Rate vs. Median Household Income\nCorrelation: {correlation_high_school:.2f}')
plt.xlabel('High School Diploma Rate (%)')
plt.ylabel('Median Household Income ($)')

# Scatter plot for Bachelor's Degree Rate vs. Median Household Income
plt.subplot(1, 2, 2)
sns.scatterplot(x=bachelors_degree_rate, y=median_household_income)
sns.regplot(x=bachelors_degree_rate, y=median_household_income, scatter=False, color='green')
plt.title(f"Bachelor's Degree Rate vs. Median Household Income\nCorrelation: {correlation_bachelors:.2f}")
plt.xlabel("Bachelor's Degree Rate (%)")
plt.ylabel('Median Household Income ($)')

plt.tight_layout()
plt.show()


# 3. In addition to that, compare that relationship for 3 different states

# Specify the states to compare
selected_states = ['California', 'Texas', 'New York']

# Subset the data for these states
subset_data = data.loc[selected_states]

# Specify colors for the selected states
colors = {'California': 'blue', 'Texas': 'orange', 'New York': 'green'}

plt.figure(figsize=(14, 8))

# Plotting High School Diploma Rate vs. Median Household Income for each state
plt.subplot(2, 2, 1)
for state in selected_states:
    hs_diploma_rate = subset_data.loc[state, 'High School Diploma or More - Pct. of Adults 25+ 2020']
    median_income = subset_data.loc[state, 'Median Household Income 2020']  # Directly use the value

    print(f"State: {state}")
    print(f"High School Diploma Rate: {hs_diploma_rate}")
    print(f"Median Household Income: {median_income}")

    plt.scatter(hs_diploma_rate,
                median_income,
                color=colors[state], label=state)
plt.title('High School Diploma Rate vs. Median Household Income')
plt.xlabel('High School Diploma Rate (%)')
plt.ylabel('Median Household Income ($)')
plt.legend()

# Plotting Bachelor's Degree Rate vs. Median Household Income for each state
plt.subplot(2, 2, 2)
for state in selected_states:
    bachelor_rate = subset_data.loc[state, "Bachelor's Degree or More - Pct. of Adults 25+ 2020"]
    median_income = subset_data.loc[state, 'Median Household Income 2020']  # Directly use the value

    print(f"State: {state}")
    print(f"Bachelor's Degree Rate: {bachelor_rate}")
    print(f"Median Household Income: {median_income}")

    plt.scatter(bachelor_rate,
                median_income,
                color=colors[state], label=state)
plt.title("Bachelor's Degree Rate vs. Median Household Income")
plt.xlabel("Bachelor's Degree Rate (%)")
plt.ylabel('Median Household Income ($)')
plt.legend()

plt.tight_layout()
plt.show()


# 4. correlation between the poverty rate in 2020 and the percentage of adults with at least a high school diploma by state.

# Extract relevant columns
high_school_diploma_rate = data['High School Diploma or More - Pct. of Adults 25+ 2020']
poverty_rate = data['Poverty Rate 2020']

# Calculate the correlation coefficient
correlation = np.corrcoef(high_school_diploma_rate, poverty_rate)[0, 1]
print(f"Correlation between High School Diploma Rate and Poverty Rate: {correlation:.2f}")

# Plot the relationship
plt.figure(figsize=(10, 6))
sns.regplot(x=high_school_diploma_rate, y=poverty_rate, scatter_kws={'s':50}, line_kws={'color':'red'})
plt.title(f"High School Diploma Rate vs. Poverty Rate\nCorrelation: {correlation:.2f}")
plt.xlabel("High School Diploma Rate (%)")
plt.ylabel("Poverty Rate (%)")
plt.show()


# 5. relationship between per capita personal income in 2021 and the percentage of adults with a bachelor's degree or more.

bachelors_degree_rate = data["Bachelor's Degree or More - Pct. of Adults 25+ 2020"]
per_capita_income = data['Per Capita Personal Income 2021']

# Calculate the correlation
correlation = np.corrcoef(bachelors_degree_rate, per_capita_income)[0, 1]

# Plotting the scatter plot with regression line
plt.figure(figsize=(10, 6))
sns.regplot(x=bachelors_degree_rate, y=per_capita_income, color='blue')

plt.title(f"Bachelor's Degree Rate vs. Per Capita Personal Income\nCorrelation: {correlation:.2f}")
plt.xlabel("Bachelor's Degree Rate (%)")
plt.ylabel("Per Capita Personal Income ($)")
plt.show()


# 6. Conduct a hypothesis testing to see if observed differences in economic indicators like median income or poverty rates across different regions are statistically significant.

# Define regions with corresponding states
regions = {
    'West': ['California', 'Oregon', 'Washington', 'Nevada', 'Arizona', 'Alaska', 'Hawaii', 'Idaho', 'Montana', 'Utah', 'Wyoming', 'Colorado','New Mexico'],
    'Midwest': ['Illinois', 'Indiana', 'Iowa', 'Michigan', 'Ohio', 'Wisconsin', 'Minnesota', 'Missouri', 'Nebraska', 'North Dakota', 'South Dakota', 'Kansas'],
    'South': ['Alabama', 'Arkansas', 'Florida', 'Georgia', 'Texas', 'Kentucky', 'Louisiana', 'Mississippi', 'North Carolina', 'South Carolina', 'Tennessee', 'Virginia', 'West Virginia', 'Oklahoma'],
    'Northeast': ['New York', 'New Jersey', 'Massachusetts', 'Pennsylvania', 'Connecticut', 'Rhode Island', 'Maine', 'New Hampshire', 'Vermont', 'Delaware', 'Maryland']
}

# Assign regions
data['Region'] = data.index.to_series().apply(
    lambda state: next((region for region, states in regions.items() if state in states), None)
)

# Filter out rows where 'Region' is None
data = data.dropna(subset=['Region'])

# ANOVA
anova_groups = []
region_names = []

for region in regions.keys():
    region_data = data[data['Region'] == region]['Median Household Income 2020'].dropna()
    if len(region_data) > 1:
        anova_groups.append(region_data.values)
        region_names.append(region)
        print(f"Region {region}: Adding data for ANOVA with {len(region_data)} states. Data: {region_data.values}")

if len(anova_groups) > 1:
    anova_result = stats.f_oneway(*anova_groups)
    print(f"ANOVA for Median Household Income across all regions: F-statistic = {anova_result.statistic:.2f}, p-value = {anova_result.pvalue:.4f}")
else:
    print("Not enough groups to perform ANOVA.")


# 7. Test the hypothesis that states with a higher percentage of manufacturing jobs (compared to the national average) have a higher average wage per job in 2021.

# Convert to percentage
if data['Manufacturing - Pct. All Jobs in County 2021'].max() <= 1:
    data['Manufacturing - Pct. All Jobs in County 2021'] *= 100

# Calculate the national average percentage of manufacturing jobs
national_avg_manufacturing_pct = data['Manufacturing - Pct. All Jobs in County 2021'].mean()

# Group 1: States with higher than average percentage of manufacturing jobs
high_manufacturing_states = data[data['Manufacturing - Pct. All Jobs in County 2021'] > national_avg_manufacturing_pct]['Avg Wage per Job 2021']

# Group 2: States with equal to or lower than average percentage of manufacturing jobs
low_manufacturing_states = data[data['Manufacturing - Pct. All Jobs in County 2021'] <= national_avg_manufacturing_pct]['Avg Wage per Job 2021']

# Perform the t-test
t_stat, p_value = stats.ttest_ind(high_manufacturing_states.dropna(), low_manufacturing_states.dropna())

# Output the results
print(f"National Average Manufacturing Job Percentage: {national_avg_manufacturing_pct:.2f}%")
print(f"t-statistic = {t_stat:.2f}, p-value = {p_value:.4f}")

# Interpretation of results
if p_value < 0.05:
    print("We reject the null hypothesis. There is a significant difference in average wage per job between the two groups.")
else:
    print("We fail to reject the null hypothesis. There is no significant difference in average wage per job between the two groups.")


# 8. Evaluate the hypothesis that states with above-average per capita personal income have lower poverty rates than the national average.
# Calculate the national average for per capita personal income and poverty rate
national_avg_income = data['Per Capita Personal Income 2021'].mean()
national_avg_poverty = data['Poverty Rate 2020'].mean()

# Categorize states based on above/below average income and poverty rate
income_category = np.where(data['Per Capita Personal Income 2021'] > national_avg_income, 'Above Average', 'Below Average')
poverty_category = np.where(data['Poverty Rate 2020'] <= national_avg_poverty, 'Below Average', 'Above Average')

# Create a contingency table
contingency_table = pd.crosstab(income_category, poverty_category)

# Perform the chi-square test
chi2, p_value, _, _ = stats.chi2_contingency(contingency_table)

# Output the results
print("Contingency Table:")
print(contingency_table)
print(f"Chi-square statistic = {chi2:.2f}, p-value = {p_value:.4f}")

# Interpretation of results
if p_value < 0.05:
    print("We reject the null hypothesis. There is a significant association between per capita personal income and poverty rates.")
else:
    print("We fail to reject the null hypothesis. There is no significant association between per capita personal income and poverty rates.")


# 9. Perform a regression analysis to understand the impact of increasing the percentage of adults with a bachelor's degree on per capita personal income by state.
# Independent variable: Percentage of adults with a bachelor's degree
X = data["Bachelor's Degree or More - Pct. of Adults 25+ 2020"]

# Dependent variable: Per capita personal income
Y = data['Per Capita Personal Income 2021']

# Perform linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(X, Y)

# Output the results
print(f"Intercept: {intercept:.2f}")
print(f"Slope: {slope:.2f}")
print(f"R-squared: {r_value**2:.2f}")
print(f"p-value: {p_value:.4f}")

# Interpretation of results
if p_value < 0.05:
    print("The relationship between the percentage of adults with a bachelor's degree and per capita personal income is statistically significant.")
else:
    print("The relationship between the percentage of adults with a bachelor's degree and per capita personal income is not statistically significant.")


# 10. Create a multiple regression model to forecast per capita personal income in 2021.

# Define the dependent variable
Y = data['Per Capita Personal Income 2021']

# Define the independent variables
X1 = data['Median Household Income 2020']
X2 = data['Manufacturing - Pct. All Jobs in County 2021']
X3 = data["Bachelor's Degree"] + data["Graduate, Professional or Doctorate Degree"]

X = np.column_stack((X1, X2, X3))

X = np.column_stack((np.ones(X.shape[0]), X))

# Perform the multiple regression
coefficients, residuals, rank, s = np.linalg.lstsq(X, Y, rcond=None)

# Output the coefficients
print("Intercept:", coefficients[0])
print("Coefficient for Median Household Income:", coefficients[1])
print("Coefficient for Percentage of Manufacturing Jobs:", coefficients[2])
print("Coefficient for Education Level (Total University Attended):", coefficients[3])

# Calculate R-squared
y_pred = X @ coefficients
ss_total = np.sum((Y - np.mean(Y)) ** 2)
ss_residual = np.sum((Y - y_pred) ** 2)
r_squared = 1 - (ss_residual / ss_total)

print(f"R-squared: {r_squared:.2f}")


# 11. Use regression analysis to examine the impact of education (both high school and bachelor's degree levels) on the average wage per job in 2021.

# Define the dependent variable
Y = data['Avg Wage per Job 2021']

# Define the independent variables
X1 = data['High School Diploma or More - Pct. of Adults 25+ 2020']
X2 = data["Bachelor's Degree or More - Pct. of Adults 25+ 2020"]

# Create the interaction term between high school education and bachelor's degree
interaction_term = X1 * X2

X = np.column_stack((X1, X2, interaction_term))

X = np.column_stack((np.ones(X.shape[0]), X))

coefficients, residuals, rank, s = np.linalg.lstsq(X, Y, rcond=None)

# Output the coefficients
print("Intercept:", coefficients[0])
print("Coefficient for High School Diploma (%):", coefficients[1])
print("Coefficient for Bachelor's Degree (%):", coefficients[2])
print("Coefficient for Interaction Term:", coefficients[3])

# Calculate R-squared
y_pred = X @ coefficients
ss_total = np.sum((Y - np.mean(Y)) ** 2)
ss_residual = np.sum((Y - y_pred) ** 2)
r_squared = 1 - (ss_residual / ss_total)

print(f"R-squared: {r_squared:.2f}")


#Machine Learning
# 1. Standardize data for total earnings, Personal Contributions for Government Social Insurance, Net Earnings by Place of Residence, and Single Family Permits.

# Extract the relevant columns
columns_to_standardize = [
    'Total Earnings by Place of Work',
    'Personal Contributions for Government Social Insurance',
    'Net Earnings by Place of Residence',
    'Single Family Permits'
]

# Extract the data
data_to_standardize = data[columns_to_standardize]

# Check data types
print("Data Types:\n", data_to_standardize.dtypes)

# Check for missing values
print("\nChecking for missing values:")
print(data_to_standardize.isnull().sum())

# Fill missing values with column mean if any
if data_to_standardize.isnull().any().any():
    data_to_standardize = data_to_standardize.fillna(data_to_standardize.mean())

# Print summary statistics
print("\nSummary Statistics before Standardization:\n", data_to_standardize.describe())

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit the scaler on the data
scaler.fit(data_to_standardize)

# Transform the data
standardized_data = scaler.transform(data_to_standardize)

# Print the standardized data directly after transformation
print("\nScaled Data (Before DataFrame Creation):\n", standardized_data)

# Create a DataFrame with the standardized data
standardized_df = pd.DataFrame(standardized_data, columns=columns_to_standardize, index=data_to_standardize.index)

# Check if the DataFrame contains NaNs after conversion
print("\nChecking for NaNs in the standardized DataFrame:")
print(standardized_df.isnull().sum())

data_standardized = data.copy()
data_standardized[columns_to_standardize] = standardized_df

# Output the standardized data
print("\nStandardized Data:\n", data_standardized[columns_to_standardize].head())


# 2. Perform hierarchical clustering on the standardized features using Ward's method
# Select the socio-economic indicators
features = data[['Per Capita Personal Income 2021',
                 'Median Household Income 2020',
                 'Poverty Rate 2020',
                 'High School Diploma or More - Pct. of Adults 25+ 2020',
                 "Bachelor's Degree or More - Pct. of Adults 25+ 2020"]]

# Verify the number of states included in the clustering
clustering_data = features  # Ensure this includes all 50 states
print(f"States included in clustering: {clustering_data.index.tolist()}")
if len(clustering_data.index) != 50:
    print("Warning: Some states are missing before clustering.")

# Step 2: Standardize the data
scaler = StandardScaler()
standardized_data = scaler.fit_transform(features)

# Perform hierarchical/agglomerative clustering
Z = linkage(standardized_data, method='ward')

# 3. Use a dendrogram to visualize the hierarchical cluster formation.
# Analyze the tree structure to understand the groupings
# Determine the cut-off for the number of clusters.

# Set the maximum distance (height) to cut the dendrogram
max_distance = 5

# Get cluster labels for each state
cluster_labels = fcluster(Z, max_distance, criterion='distance')

# Add cluster labels to the original DataFrame
data['Cluster'] = cluster_labels

# View the resulting clusters
print(data[['Cluster', 'Per Capita Personal Income 2021', 'Poverty Rate 2020']])

# Plot the dendrogram
plt.figure(figsize=(12, 8))
dendrogram(Z, labels=clustering_data.index, leaf_rotation=90, leaf_font_size=10, color_threshold=max_distance)
plt.title('Dendrogram of States Based on Socio-Economic Indicators')
plt.xlabel('State')
plt.ylabel('Distance')
plt.show()


# 4. Analyze the characteristics of each cluster based on the original features
# Understand the commonalities within each group and how they differ from others.
#Cluster 1(Orange): Colorado, Virginia, Minnesota, New Hampshire, Washington, Alaska, Hawaii
#Cluster2(Green) : Maryland, Connecticut, Massachusetts, New Jersey, New York, California
#Cluster 3(Red):Illinois, Delaware, Rhode Island, Vermont, Nebraska, North Dakota, Wyoming, Ohio, Michigan
#Cluster 4(Purple): Wisconsin, Missouri, Iowa, Indiana, Maine
#Cluster 5(Brown): Florida, Georgia, Nevada, Arizona, Texas, South Carolina, Oklahoma, Arkansas, Alabama, Louisiana, Mississippi

# Group the data by 'Cluster' and calculate the mean of each feature within each cluster
#cluster_summary = data.groupby('Cluster').mean()
# Display the cluster characteristics
#print(cluster_summary)''''''



# 5. Map out the geographical patterns in population growth, economic indicators, and educational levels using maps and spatial data, ie US states

# Load the shapefile
shapefile_path = '/Users/lokeshwaripotluri/Downloads/ne_110m_admin_1_states_provinces/ne_110m_admin_1_states_provinces.shp'
states = gpd.read_file(shapefile_path)

data = data.reset_index()
data = data.rename(columns={'index': 'State'})

# Merge the GeoDataFrame with your data based on state names
merged = states.merge(data, how='left', left_on='name', right_on='State')

# Plot the results
fig, ax = plt.subplots(1, 1, figsize=(15, 10))
merged.boundary.plot(ax=ax, linewidth=1)
merged.plot(column='Per Capita Personal Income 2021', ax=ax, legend=True,
            legend_kwds={'label': "Per Capita Personal Income by State",
                         'orientation': "horizontal"})
plt.title("Per Capita Personal Income Across US States")
plt.show()

#Median Household income
fig, ax = plt.subplots(1, 1, figsize=(15, 10))
merged.boundary.plot(ax=ax, linewidth=1)
merged.plot(column='Median Household Income 2020', ax=ax, legend=True,
         legend_kwds={'label': "Median Household Income (2020)", 'orientation': "horizontal"},
         cmap='YlGnBu')
plt.title('US States: Median Household Income (2020)')
plt.show()

#Edcuational levels
fig, ax = plt.subplots(1, 1, figsize=(15, 10))
merged.boundary.plot(ax=ax, linewidth=1)
merged.plot(column="Bachelor's Degree or More - Pct. of Adults 25+ 2020", ax=ax, legend=True,
         legend_kwds={'label': "Pct. of Adults 25+ with Bachelor's Degree (2020)", 'orientation': "horizontal"},
         cmap='Purples')
plt.title("US States: Adults 25+ with Bachelor's Degree (2020)")
plt.show()
