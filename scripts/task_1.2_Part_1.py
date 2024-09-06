import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the data
df = pd.read_csv('Week2_challenge_data_source.csv')

# Convert 'Start' and 'End' columns to datetime
df['Start'] = pd.to_datetime(df['Start'])
df['End'] = pd.to_datetime(df['End'])

# Calculate total session duration in seconds
df['Total_Duration'] = (df['End'] - df['Start']).dt.total_seconds()

# Calculate total data (DL + UL)
df['Total_Data'] = df['Total UL (Bytes)'] + df['Total DL (Bytes)']

print(df.head())
print("\
Dataframe shape:", df.shape)
print("\
Column names:")
print(df.columns)

# Segment users into five decile classes based on total session duration
df['Decile'] = pd.qcut(df['Total_Duration'], 5, labels=False)

# Compute total data (DL + UL) per decile class
total_data_per_decile = df.groupby('Decile')['Total_Data'].sum()

print(total_data_per_decile)

import matplotlib.pyplot as plt
import seaborn as sns

# Basic Metrics Analysis
basic_metrics = df['Total_Data'].describe()
print("Basic Metrics for Total Data:")
print(basic_metrics)

# Non-Graphical Univariate Analysis
dispersion_params = pd.DataFrame({
    'Variance': df.var(),
    'IQR': df.quantile(0.75) - df.quantile(0.25)
})
print("\
Dispersion Parameters:")
print(dispersion_params)

# Graphical Univariate Analysis
plt.figure(figsize=(15, 5))

# Histogram
plt.subplot(131)
sns.histplot(df['Total_Data'], kde=True)
plt.title('Histogram of Total Data')
plt.xlabel('Total Data')

# Box Plot
plt.subplot(132)
sns.boxplot(y=df['Total_Data'])
plt.title('Box Plot of Total Data')
plt.ylabel('Total Data')

# Density Plot
plt.subplot(133)
sns.kdeplot(data=df['Total_Data'])
plt.title('Density Plot of Total Data')
plt.xlabel('Total Data')

plt.tight_layout()
plt.savefig('univariate_analysis.png')
plt.close()

print("\
Univariate analysis plots saved as 'univariate_analysis.png'")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Select only numeric columns
numeric_columns = df.select_dtypes(include=[np.number]).columns

# Basic Metrics Analysis
basic_metrics = df[numeric_columns].describe()
print("Basic Metrics for Numeric Columns:")
print(basic_metrics)

# Non-Graphical Univariate Analysis
dispersion_params = pd.DataFrame({
    'Variance': df[numeric_columns].var(),
    'IQR': df[numeric_columns].quantile(0.75) - df[numeric_columns].quantile(0.25)
})
print("\
Dispersion Parameters for Numeric Columns:")
print(dispersion_params)

# Graphical Univariate Analysis
plt.figure(figsize=(15, 5))

# Histogram
plt.subplot(131)
sns.histplot(df['Total_Data'], kde=True)
plt.title('Histogram of Total Data')
plt.xlabel('Total Data')

# Box Plot
plt.subplot(132)
sns.boxplot(y=df['Total_Data'])
plt.title('Box Plot of Total Data')
plt.ylabel('Total Data')

# Density Plot
plt.subplot(133)
sns.kdeplot(data=df['Total_Data'])
plt.title('Density Plot of Total Data')
plt.xlabel('Total Data')

plt.tight_layout()
plt.savefig('univariate_analysis.png')
plt.close()

print("\
Univariate analysis plots saved as 'univariate_analysis.png'")

