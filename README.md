Telecom User Overview and Engagement Analysis
Introduction
This repository provides a comprehensive exploratory analysis of telecom user behavior and engagement using xDR (data session detail records). The primary focus is on understanding user activity patterns, handset preferences, and app-specific engagement metrics. Additionally, the project applies clustering techniques to segment users based on their activity levels, providing insights for targeted marketing and technical improvements.

Dataset
The dataset consists of detailed telecom records that include user sessions, session duration, upload/download data, and application-specific usage metrics (e.g., Social Media, Google, YouTube, etc.). The main features analyzed are user handset types, session metrics, and data traffic.

Task Breakdown
Task 1: User Overview Analysis
Task 1.1: User Behavior Overview
Top 10 Handsets: Identify the top 10 handsets used by customers.
Top 3 Manufacturers: Identify the top 3 handset manufacturers.
Top 5 Handsets per Manufacturer: For each of the top 3 manufacturers, identify the top 5 handsets.
Marketing Recommendations: Provide short recommendations to marketing teams based on handset popularity.
Task 1.2: Application Usage Overview
Aggregate the following metrics per user:
Number of xDR sessions
Session duration
Total download (DL) and upload (UL) data
Total traffic for each application (Social Media, Google, Email, YouTube, Netflix, Gaming, Others)
Exploratory Data Analysis (EDA):
Handle missing values and outliers (replace with mean or appropriate methods).
Perform descriptive statistics and report the relevant metrics.
Segment users into deciles based on session duration and compute total data (DL + UL) for each class.
Conduct both graphical and non-graphical univariate analysis (histograms, boxplots, and descriptive measures).
Task 1.3: Advanced Analysis
Bivariate Analysis: Analyze the relationships between application-specific data usage and total data consumption.
Correlation Matrix: Explore the correlation between application data (Social Media, Google, Email, YouTube, Netflix, Gaming, Others).
Dimensionality Reduction: Use Principal Component Analysis (PCA) to reduce data dimensions and interpret key insights.
Task 2: User Engagement Analysis
Task 2.1: Engagement Metrics Aggregation
Metrics Aggregation:
Aggregate sessions frequency, session duration, and total traffic per customer (MSISDN).
Identify the top 10 customers based on each engagement metric.
Clustering with K-Means:
Normalize the engagement metrics.
Use K-Means clustering to classify users into 3 clusters based on their engagement.
Analyze cluster characteristics (minimum, maximum, mean, and total values).
Visualize the clusters using appropriate charts.
Application Traffic Analysis:
Aggregate total traffic per application for each user and identify the top 10 most engaged users per application.
Plot the top 3 most used applications by traffic.
Optimizing Clusters: Use the elbow method to determine the optimal number of clusters (k) and interpret the findings.


