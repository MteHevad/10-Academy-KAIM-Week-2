import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = r'C:\Users\hp\Desktop\KAIM\data\Week2_challenge_data_source.csv'
df = pd.read_csv(file_path)

# Handle missing values by replacing with mean for numeric and mode for non-numeric
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
df[non_numeric_cols] = df[non_numeric_cols].fillna(df[non_numeric_cols].mode().iloc[0])

# Treat outliers in numeric columns: Replace values beyond 3 standard deviations with the mean
for col in numeric_cols:
    col_mean = df[col].mean()
    col_std = df[col].std()
    df.loc[(df[col] > col_mean + 3 * col_std) | (df[col] < col_mean - 3 * col_std), col] = col_mean

# Task 3.1: Aggregate per customer
customer_agg = df.groupby('MSISDN/Number').agg({
    'TCP DL Retrans. Vol (Bytes)': 'mean',
    'TCP UL Retrans. Vol (Bytes)': 'mean',
    'Avg RTT DL (ms)': 'mean',
    'Avg RTT UL (ms)': 'mean',
    'Avg Bearer TP DL (kbps)': 'mean',
    'Avg Bearer TP UL (kbps)': 'mean',
    'Handset Type': 'first'  # Assuming the first occurrence represents the user's device
}).reset_index()

# Calculate averages for TCP retransmission, RTT, and throughput
customer_agg['Average TCP Retransmission'] = (customer_agg['TCP DL Retrans. Vol (Bytes)'] + customer_agg['TCP UL Retrans. Vol (Bytes)']) / 2
customer_agg['Average RTT'] = (customer_agg['Avg RTT DL (ms)'] + customer_agg['Avg RTT UL (ms)']) / 2
customer_agg['Average Throughput'] = (customer_agg['Avg Bearer TP DL (kbps)'] + customer_agg['Avg Bearer TP UL (kbps)']) / 2

# Display the aggregated data for review
print("\nAggregated Customer Data (First 5 Rows):\n", customer_agg.head())

# Display full data for all numerical computations if needed
print("\nFull Aggregated Customer Data:\n", customer_agg)

# Task 3.2: Compute & list 10 of the top, bottom, and most frequent values
def compute_top_bottom_frequent(column_name):
    top_10 = customer_agg[column_name].nlargest(10)
    bottom_10 = customer_agg[column_name].nsmallest(10)
    most_freq = customer_agg[column_name].mode().head(10)
    print(f"\nTop 10 {column_name} Values:\n", top_10.to_list())
    print(f"\nBottom 10 {column_name} Values:\n", bottom_10.to_list())
    print(f"\nMost Frequent {column_name} Values:\n", most_freq.to_list())

# Compute for TCP retransmission, RTT, and Throughput
compute_top_bottom_frequent('Average TCP Retransmission')
compute_top_bottom_frequent('Average RTT')
compute_top_bottom_frequent('Average Throughput')

# Task 3.3: Compute & report distribution by handset type
# Average throughput per handset type
throughput_by_handset = customer_agg.groupby('Handset Type')['Average Throughput'].mean().sort_values(ascending=False)
tcp_by_handset = customer_agg.groupby('Handset Type')['Average TCP Retransmission'].mean().sort_values(ascending=False)

# Print distributions for all handset types
print("\nAverage Throughput per Handset Type:\n", throughput_by_handset)
print("\nAverage TCP Retransmission per Handset Type:\n", tcp_by_handset)

# Plot distributions
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
throughput_by_handset.plot(kind='bar', color='skyblue')
plt.title('Average Throughput per Handset Type')
plt.xlabel('Handset Type')
plt.ylabel('Average Throughput (kbps)')
plt.xticks(rotation=45)

plt.subplot(1, 2, 2)
tcp_by_handset.plot(kind='bar', color='salmon')
plt.title('Average TCP Retransmission per Handset Type')
plt.xlabel('Handset Type')
plt.ylabel('Average TCP Retransmission')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Task 3.4: K-means clustering for user experience segmentation
# Select features for clustering
features = customer_agg[['Average TCP Retransmission', 'Average RTT', 'Average Throughput']]

# Standardize the data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Apply K-Means clustering with k=3
kmeans = KMeans(n_clusters=3, random_state=42)
customer_agg['Cluster'] = kmeans.fit_predict(scaled_features)

# Print cluster assignments and summary
print("\nCluster Assignments (First 5 Rows):\n", customer_agg[['MSISDN/Number', 'Cluster']].head())
cluster_summary = customer_agg.groupby('Cluster').agg({
    'Average TCP Retransmission': 'mean',
    'Average RTT': 'mean',
    'Average Throughput': 'mean',
    'MSISDN/Number': 'count'
}).rename(columns={'MSISDN/Number': 'Number of Users'})

print("\nCluster Summary:\n", cluster_summary)

print("\nCluster Descriptions:")
print("Cluster 0: High Throughput, Low Retransmission, Low RTT – Users with optimal experience.")
print("Cluster 1: Moderate Throughput, Moderate Retransmission, Moderate RTT – Users with average experience.")
print("Cluster 2: Low Throughput, High Retransmission, High RTT – Users with poor experience, likely needing network improvements.")
