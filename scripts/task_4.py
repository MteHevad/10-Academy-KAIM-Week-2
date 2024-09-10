import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
import mysql.connector
import matplotlib.pyplot as plt

# Step 0: Load the dataset from the specified file path
file_path = r"C:\Users\hp\Desktop\KAIM\data\Week2_challenge_data_source.csv"
customer_agg = pd.read_csv(file_path)  # Modify this line if the file is in a different format (e.g., Excel)

# Assume 'scaled_features' contains normalized values of necessary features for clustering.
# Here, I'm assuming the file contains relevant features for clustering and scoring

# Step 1: Engagement & Experience Score Calculation (Task 4.1)
# ----------------------------------------------------------

# Assuming the cluster centroids were obtained from prior KMeans clustering (3 clusters)
# If you haven't performed clustering, do it here
kmeans = KMeans(n_clusters=3, random_state=42).fit(scaled_features)
cluster_centroids = kmeans.cluster_centers_

# Assuming cluster 0 is less engaged, and cluster 2 is the worst experience
less_engaged_cluster = cluster_centroids[0]  # Less engaged
worst_experience_cluster = cluster_centroids[2]  # Worst experience

# Calculate Euclidean distances for engagement and experience
customer_agg['Engagement Score'] = euclidean_distances(scaled_features, [less_engaged_cluster]).flatten()
customer_agg['Experience Score'] = euclidean_distances(scaled_features, [worst_experience_cluster]).flatten()

# Step 2: Satisfaction Score Calculation (Task 4.2)
# ------------------------------------------------
customer_agg['Satisfaction Score'] = (customer_agg['Engagement Score'] + customer_agg['Experience Score']) / 2

# Report Top 10 Satisfied Customers
top_10_satisfied = customer_agg[['MSISDN/Number', 'Satisfaction Score']].sort_values(by='Satisfaction Score', ascending=False).head(10)
print("Top 10 Satisfied Customers:\n", top_10_satisfied)

# Step 3: Build a Regression Model to Predict Satisfaction Score (Task 4.3)
# -----------------------------------------------------------------------
# Features: Engagement Score & Experience Score
X = customer_agg[['Engagement Score', 'Experience Score']]
y = customer_agg['Satisfaction Score']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict satisfaction score on test data
y_pred = model.predict(X_test)

# Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error of the model: {mse}")

# Step 4: K-Means Clustering (k=2) on Engagement & Experience Scores (Task 4.4)
# ---------------------------------------------------------------------------
kmeans_2 = KMeans(n_clusters=2, random_state=42)
customer_agg['Engagement-Experience Cluster'] = kmeans_2.fit_predict(customer_agg[['Engagement Score', 'Experience Score']])

# Step 5: Aggregate Satisfaction & Experience Scores per Cluster (Task 4.5)
# ------------------------------------------------------------------------
cluster_agg = customer_agg.groupby('Engagement-Experience Cluster').agg({
    'Satisfaction Score': 'mean',
    'Experience Score': 'mean'
})

print("\nAverage Satisfaction & Experience Scores per Cluster:\n", cluster_agg)

# Step 6: Export Data to MySQL Database (Task 4.6)
# -----------------------------------------------
# Connect to MySQL database (replace with your database credentials)
db_connection = mysql.connector.connect(
    host="localhost",
    user="your_username",
    password="your_password",
    database="your_database"
)

cursor = db_connection.cursor()

# Create table for customer satisfaction data if not already present
create_table_query = '''
CREATE TABLE IF NOT EXISTS customer_satisfaction (
    MSISDN_Number BIGINT PRIMARY KEY,
    Engagement_Score FLOAT,
    Experience_Score FLOAT,
    Satisfaction_Score FLOAT
)
'''
cursor.execute(create_table_query)

# Insert customer data into the MySQL table
insert_query = '''
INSERT INTO customer_satisfaction (MSISDN_Number, Engagement_Score, Experience_Score, Satisfaction_Score)
VALUES (%s, %s, %s, %s)
'''

# Prepare the data to insert
data_to_insert = customer_agg[['MSISDN/Number', 'Engagement Score', 'Experience Score', 'Satisfaction Score']].values.tolist()
cursor.executemany(insert_query, data_to_insert)

# Commit the transaction
db_connection.commit()

# Verify the inserted data by running a select query
cursor.execute("SELECT * FROM customer_satisfaction LIMIT 5")
for row in cursor.fetchall():
    print(row)

# Close the cursor and connection
cursor.close()
db_connection.close()

# Visualization (Optional, if required)
# ------------------------------------
# Scatter plot for Engagement vs Experience Score colored by Clusters
plt.figure(figsize=(8, 6))
plt.scatter(customer_agg['Engagement Score'], customer_agg['Experience Score'], c=customer_agg['Engagement-Experience Cluster'], cmap='viridis')
plt.title('Engagement vs Experience Scores (Clustered)')
plt.xlabel('Engagement Score')
plt.ylabel('Experience Score')
plt.colorbar(label='Cluster')
plt.show()
