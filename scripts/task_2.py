import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Load the data
df = pd.read_csv('Week2_challenge_data_source.csv')

# Display the first few rows and data info
print(df.head())
print("\
Dataset Info:")
print(df.info())

# Step 1: Engagement Metrics Aggregation
# Aggregate metrics (sessions frequency, session duration, total traffic) per customer

# Group by 'MSISDN/Number' to calculate the required metrics
engagement_metrics = df.groupby('MSISDN/Number').agg({
    'Bearer Id': 'count',  # Sessions frequency
    'Dur. (ms)': 'sum',    # Total session duration
    'Total UL (Bytes)': 'sum',  # Total upload traffic
    'Total DL (Bytes)': 'sum'   # Total download traffic
}).reset_index()

# Calculate total traffic
engagement_metrics['Total Traffic (Bytes)'] = engagement_metrics['Total UL (Bytes)'] + engagement_metrics['Total DL (Bytes)']

# Identify the top 10 customers for each metric
top_10_sessions = engagement_metrics.nlargest(10, 'Bearer Id')
top_10_duration = engagement_metrics.nlargest(10, 'Dur. (ms)')
top_10_traffic = engagement_metrics.nlargest(10, 'Total Traffic (Bytes)')

print("Top 10 Customers by Sessions Frequency:")
print(top_10_sessions)

print("\
Top 10 Customers by Session Duration:")
print(top_10_duration)

print("\
Top 10 Customers by Total Traffic:")
print(top_10_traffic)

# Step 2: Normalize Metrics and Run K-Means Clustering
# Normalize metrics to ensure comparability
scaler = StandardScaler()
normalized_metrics = scaler.fit_transform(engagement_metrics[['Bearer Id', 'Dur. (ms)', 'Total Traffic (Bytes)']])

# Apply k-means clustering with k=3
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(normalized_metrics)

# Add cluster labels to the dataframe
engagement_metrics['Cluster'] = clusters

# Calculate cluster statistics
cluster_stats = engagement_metrics.groupby('Cluster').agg({
    'Bearer Id': ['min', 'max', 'mean', 'sum'],
    'Dur. (ms)': ['min', 'max', 'mean', 'sum'],
    'Total Traffic (Bytes)': ['min', 'max', 'mean', 'sum']
})

print("Cluster Statistics:")
print(cluster_stats)

# Step 3: Visualize Results
# Use bar charts and scatter plots to illustrate engagement insights

# Bar chart for the number of customers in each cluster
plt.figure(figsize=(10, 6))
sns.countplot(x='Cluster', data=engagement_metrics)
plt.title('Number of Customers in Each Cluster')
plt.xlabel('Cluster')
plt.ylabel('Number of Customers')
plt.show()

# Scatter plot for session duration vs. total traffic colored by cluster
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Dur. (ms)', y='Total Traffic (Bytes)', hue='Cluster', data=engagement_metrics, palette='viridis')
plt.title('Session Duration vs. Total Traffic by Cluster')
plt.xlabel('Session Duration (ms)')
plt.ylabel('Total Traffic (Bytes)')
plt.show()

# Step 4: Analyze Engagement by Application
# Aggregate user total traffic per application and identify the top 10 most engaged users

# List of application columns
app_columns = ['Social Media DL (Bytes)', 'Social Media UL (Bytes)', 
               'Google DL (Bytes)', 'Google UL (Bytes)',
               'Email DL (Bytes)', 'Email UL (Bytes)',
               'Youtube DL (Bytes)', 'Youtube UL (Bytes)',
               'Netflix DL (Bytes)', 'Netflix UL (Bytes)',
               'Gaming DL (Bytes)', 'Gaming UL (Bytes)',
               'Other DL (Bytes)', 'Other UL (Bytes)']

# Aggregate total traffic per application for each user
app_engagement = df.groupby('MSISDN/Number')[app_columns].sum().reset_index()

# Calculate total traffic for each application
for app in ['Social Media', 'Google', 'Email', 'Youtube', 'Netflix', 'Gaming', 'Other']:
    app_engagement[f'{app} Total (Bytes)'] = app_engagement[f'{app} DL (Bytes)'] + app_engagement[f'{app} UL (Bytes)']

# Identify top 10 most engaged users overall
top_10_engaged = app_engagement.nlargest(10, app_columns)

print("Top 10 Most Engaged Users by Total Application Traffic:")
print(top_10_engaged[['MSISDN/Number'] + [f'{app} Total (Bytes)' for app in ['Social Media', 'Google', 'Email', 'Youtube', 'Netflix', 'Gaming', 'Other']]])

# Calculate total traffic for each application across all users
total_app_traffic = app_engagement[[f'{app} Total (Bytes)' for app in ['Social Media', 'Google', 'Email', 'Youtube', 'Netflix', 'Gaming', 'Other']]].sum()

# Sort applications by total traffic
top_apps = total_app_traffic.sort_values(ascending=False)

# Plot the top 3 most used applications
plt.figure(figsize=(10, 6))
top_3_apps = top_apps.head(3)
plt.pie(top_3_apps.values, labels=top_3_apps.index, autopct='%1.1f%%')
plt.title('Top 3 Most Used Applications by Total Traffic')
plt.axis('equal')
plt.show()

print("\
Total Traffic per Application:")
print(total_app_traffic)
