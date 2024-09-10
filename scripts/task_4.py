import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from scipy.spatial.distance import euclidean
import mysql.connector

# Load the data
file_path = "C:/Users/hp/Desktop/KAIM/data/Week2_challenge_data_source.csv"
data = pd.read_csv(file_path)

# Task 4.1: Calculate Engagement and Experience Scores
# Assuming 'engagement_features' and 'experience_features' are the feature sets used for clustering

# Use the first clustering results
kmeans_engagement = KMeans(n_clusters=3).fit(data[engagement_features])
engagement_clusters = kmeans_engagement.predict(data[engagement_features])

# Use the second clustering results for experience
kmeans_experience = KMeans(n_clusters=3).fit(data[experience_features])
experience_clusters = kmeans_experience.predict(data[experience_features])

# Get cluster centers for engagement and experience
less_engaged_cluster_center = kmeans_engagement.cluster_centers_[0]  # Assuming 0 is the least engaged cluster
worst_experience_cluster_center = kmeans_experience.cluster_centers_[-1]  # Assuming -1 is the worst experience cluster

# Calculate Engagement and Experience scores using Euclidean distance
data['Engagement_Score'] = data[engagement_features].apply(lambda x: euclidean(x, less_engaged_cluster_center), axis=1)
data['Experience_Score'] = data[experience_features].apply(lambda x: euclidean(x, worst_experience_cluster_center), axis=1)

# Task 4.2: Calculate Satisfaction Score and Report Top 10 Satisfied Customers
data['Satisfaction_Score'] = data[['Engagement_Score', 'Experience_Score']].mean(axis=1)

# Sort and report the top 10 satisfied customers
top_10_satisfied = data.nlargest(10, 'Satisfaction_Score')

# Task 4.3: Build a Regression Model to Predict Satisfaction Score
X = data[['Engagement_Score', 'Experience_Score']]  # Features
y = data['Satisfaction_Score']  # Target

reg_model = LinearRegression().fit(X, y)
data['Predicted_Satisfaction_Score'] = reg_model.predict(X)

# Task 4.4: K-means Clustering (k=2) on Engagement & Experience Scores
kmeans_satisfaction = KMeans(n_clusters=2).fit(data[['Engagement_Score', 'Experience_Score']])
data['Satisfaction_Cluster'] = kmeans_satisfaction.predict(data[['Engagement_Score', 'Experience_Score']])

# Task 4.5: Aggregate Average Satisfaction & Experience per Cluster
cluster_aggregation = data.groupby('Satisfaction_Cluster').agg({
    'Satisfaction_Score': 'mean',
    'Experience_Score': 'mean'
}).reset_index()

# Task 4.6: Export Final Table to MySQL Database

# MySQL connection setup (use your credentials)
connection = mysql.connector.connect(
    host="localhost",
    user="your_username",
    password="your_password",
    database="your_database"
)

cursor = connection.cursor()

# Create table if not exists
cursor.execute("""
CREATE TABLE IF NOT EXISTS satisfaction_analysis (
    user_id VARCHAR(255),
    engagement_score FLOAT,
    experience_score FLOAT,
    satisfaction_score FLOAT
)
""")

# Insert data into MySQL
for index, row in data.iterrows():
    cursor.execute("""
        INSERT INTO satisfaction_analysis (user_id, engagement_score, experience_score, satisfaction_score)
        VALUES (%s, %s, %s, %s)
    """, (row['user_id'], row['Engagement_Score'], row['Experience_Score'], row['Satisfaction_Score']))

# Commit and close connection
connection.commit()
cursor.close()
connection.close()

# Print a screenshot of the database content
# This would typically be done in a SQL client like MySQL Workbench to show the screenshot
