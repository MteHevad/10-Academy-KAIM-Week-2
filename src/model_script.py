import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import euclidean_distances
import mlflow
import mlflow.sklearn
import time

# Start MLflow run
with mlflow.start_run() as run:
    # Log code version, source, start and end time
    mlflow.log_param("code_version", "v1.0")
    mlflow.log_param("source", "Customer Satisfaction Analysis")
    start_time = time.time()
    mlflow.log_param("start_time", start_time)
    
    # Load your data
    data_path = 'C:\\yfinance_data\\your_file.csv'  # Replace with your actual file path
    data = pd.read_csv(data_path)

    # Define features for engagement and experience
    engagement_features = ['engagement_feature1', 'engagement_feature2']  # Replace with actual column names
    experience_features = ['experience_feature1', 'experience_feature2']  # Replace with actual column names

    # Task 4.1: Engagement and Experience Score Calculation
    engagement_data = data[engagement_features]
    experience_data = data[experience_features]

    # KMeans Clustering
    kmeans_engagement = KMeans(n_clusters=2, random_state=0).fit(engagement_data)
    kmeans_experience = KMeans(n_clusters=2, random_state=0).fit(experience_data)

    # Log parameters (e.g., number of clusters)
    mlflow.log_param("kmeans_n_clusters", 2)

    # Compute Scores
    less_engaged_cluster_center = kmeans_engagement.cluster_centers_[np.argmin(kmeans_engagement.cluster_centers_.sum(axis=1))]
    worst_experience_cluster_center = kmeans_experience.cluster_centers_[np.argmin(kmeans_experience.cluster_centers_.sum(axis=1))]

    data['engagement_score'] = euclidean_distances(engagement_data, [less_engaged_cluster_center]).flatten()
    data['experience_score'] = euclidean_distances(experience_data, [worst_experience_cluster_center]).flatten()

    # Task 4.2: Satisfaction Score Calculation
    data['satisfaction_score'] = data[['engagement_score', 'experience_score']].mean(axis=1)
    top_10_satisfied = data.nlargest(10, 'satisfaction_score')

    # Task 4.3: Regression Model
    X = data[['engagement_score', 'experience_score']]
    y = data['satisfaction_score']
    regression_model = LinearRegression()
    regression_model.fit(X, y)

    # Log model metrics
    mlflow.log_metric("regression_score", regression_model.score(X, y))

    # Predict and log satisfaction scores
    predicted_scores = regression_model.predict(X)
    data['predicted_satisfaction_score'] = predicted_scores

    # Task 4.4: K-means Clustering on Scores
    kmeans_scores = KMeans(n_clusters=2, random_state=0).fit(data[['engagement_score', 'experience_score']])
    data['score_cluster'] = kmeans_scores.labels_

    # Task 4.5: Aggregate Scores
    cluster_summary = data.groupby('score_cluster').agg({'satisfaction_score': 'mean', 'experience_score': 'mean'}).reset_index()

    # Log artifacts - save results to CSV and log it
    result_file_path = 'C:\\yfinance_data\\satisfaction_analysis_results.csv'
    data.to_csv(result_file_path, index=False)
    mlflow.log_artifact(result_file_path)

    # Log model
    mlflow.sklearn.log_model(regression_model, "satisfaction_regression_model")

    # Log end time
    end_time = time.time()
    mlflow.log_param("end_time", end_time)
    mlflow.log_metric("run_duration", end_time - start_time)
