import pandas as pd

# Load the dataset
file_path = 'Week2_challenge_data_source.csv'
df = pd.read_csv(file_path, encoding='ascii')

# Display the first few rows of the dataframe to understand its structure
df.head()

# Identify the top 10 handsets used by customers
# Using value_counts() to list handsets by usage frequency
top_handsets = df['Handset Type'].value_counts().head(10)

print(top_handsets)

# Step 2: Identify the top 3 handset manufacturers
# Group the handsets by manufacturer and count the number of handsets per manufacturer

top_manufacturers = df['Handset Manufacturer'].value_counts().head(3)

print(top_manufacturers)

# Step 3: Identify the top 5 handsets per top 3 manufacturer
# Filter the dataset for each of the top 3 manufacturers and identify the top 5 handsets

# Create a dictionary to store the top 5 handsets for each manufacturer
top_5_handsets_per_manufacturer = {}

# Loop through each of the top 3 manufacturers
top_manufacturers_list = top_manufacturers.index.tolist()

for manufacturer in top_manufacturers_list:
    # Filter the dataset for the current manufacturer
    manufacturer_data = df[df['Handset Manufacturer'] == manufacturer]
    
    # Identify the top 5 handsets for the current manufacturer
    top_5_handsets = manufacturer_data['Handset Type'].value_counts().head(5)
    
    # Store the result in the dictionary
    top_5_handsets_per_manufacturer[manufacturer] = top_5_handsets

# Print the top 5 handsets for each of the top 3 manufacturers
for manufacturer, handsets in top_5_handsets_per_manufacturer.items():
    print('Top 5 handsets for', manufacturer, ':')
    print(handsets)
    print()

# Step 1: Aggregate Information Per User
# A. Compute the following per user:
# 1. Number of xDR sessions
# 2. Session duration
# 3. Total download (DL) and upload (UL) data
# 4. Total data volume per application (DL + UL)

# Group by 'MSISDN/Number' to aggregate data per user
user_aggregates = df.groupby('MSISDN/Number').agg({
    'Bearer Id': 'count',  # Number of xDR sessions
    'Dur. (ms)': 'sum',    # Total session duration
    'Total DL (Bytes)': 'sum',  # Total download data
    'Total UL (Bytes)': 'sum',  # Total upload data
    'Social Media DL (Bytes)': 'sum',
    'Social Media UL (Bytes)': 'sum',
    'Google DL (Bytes)': 'sum',
    'Google UL (Bytes)': 'sum',
    'Email DL (Bytes)': 'sum',
    'Email UL (Bytes)': 'sum',
    'Youtube DL (Bytes)': 'sum',
    'Youtube UL (Bytes)': 'sum',
    'Netflix DL (Bytes)': 'sum',
    'Netflix UL (Bytes)': 'sum',
    'Gaming DL (Bytes)': 'sum',
    'Gaming UL (Bytes)': 'sum',
    'Other DL (Bytes)': 'sum',
    'Other UL (Bytes)': 'sum'
}).reset_index()

# Calculate total data volume per application (DL + UL)
user_aggregates['Total Social Media'] = user_aggregates['Social Media DL (Bytes)'] + user_aggregates['Social Media UL (Bytes)']
user_aggregates['Total Google'] = user_aggregates['Google DL (Bytes)'] + user_aggregates['Google UL (Bytes)']
user_aggregates['Total Email'] = user_aggregates['Email DL (Bytes)'] + user_aggregates['Email UL (Bytes)']
user_aggregates['Total Youtube'] = user_aggregates['Youtube DL (Bytes)'] + user_aggregates['Youtube UL (Bytes)']
user_aggregates['Total Netflix'] = user_aggregates['Netflix DL (Bytes)'] + user_aggregates['Netflix UL (Bytes)']
user_aggregates['Total Gaming'] = user_aggregates['Gaming DL (Bytes)'] + user_aggregates['Gaming UL (Bytes)']
user_aggregates['Total Other'] = user_aggregates['Other DL (Bytes)'] + user_aggregates['Other UL (Bytes)']

# Display the first few rows of the aggregated data
user_aggregates.head()

# Check data types of the columns
df.dtypes

# Select only numeric columns for missing value replacement
numeric_columns = df.select_dtypes(include=[np.number]).columns

# Replace missing values with the mean of the column for numeric columns only
for column in numeric_columns:
    if df[column].isnull().sum() > 0:
        df[column].fillna(df[column].mean(), inplace=True)

# Re-run the box plot and z-score calculation
plt.figure(figsize=(15, 10))
sns.boxplot(data=df[numeric_columns])
plt.xticks(rotation=90)
plt.title('Boxplot for Numerical Features')
plt.show()

# Calculate z-scores to detect outliers
z_scores = np.abs((df[numeric_columns] - df[numeric_columns].mean()) / df[numeric_columns].std())

# Identify outliers (z-score > 3)
outliers = (z_scores > 3).sum()

print('Missing values per column:', df_missing)
print('Outliers per column:', outliers)

# B. Variable Descriptions: Provide descriptions and data types for all relevant variables

# Get data types and a brief description of the first few rows to understand the variables
df_info = df.dtypes

df_head = df.head()

print('Data Types:', df_info)
print('First few rows of the dataframe:', df_head)
