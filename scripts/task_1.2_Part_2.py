import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Bivariate Analysis
app_columns = ['Social Media DL (Bytes)', 'Social Media UL (Bytes)', 
               'Google DL (Bytes)', 'Google UL (Bytes)', 
               'Email DL (Bytes)', 'Email UL (Bytes)', 
               'Youtube DL (Bytes)', 'Youtube UL (Bytes)', 
               'Netflix DL (Bytes)', 'Netflix UL (Bytes)', 
               'Gaming DL (Bytes)', 'Gaming UL (Bytes)', 
               'Other DL (Bytes)', 'Other UL (Bytes)']

# Scatter plot matrix
sns.pairplot(df[app_columns + ['Total_Data']], diag_kind='kde', plot_kws={'alpha': 0.6})
plt.tight_layout()
plt.savefig('scatter_matrix.png')
plt.close()

# Correlation Analysis
correlation_matrix = df[app_columns + ['Total_Data']].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.close()

print("Scatter matrix and correlation heatmap saved as 'scatter_matrix.png' and 'correlation_heatmap.png'")

# Dimensionality Reduction (PCA)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[app_columns])

pca = PCA()
pca_result = pca.fit_transform(scaled_data)

# Calculate explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_

# Plot cumulative explained variance ratio
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(explained_variance_ratio) + 1), np.cumsum(explained_variance_ratio))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('PCA: Cumulative Explained Variance Ratio')
plt.savefig('pca_explained_variance.png')
plt.close()

print("PCA explained variance plot saved as 'pca_explained_variance.png'")

# Print key findings
print("\
Key findings from PCA:")
print(f"1. Number of components needed to explain 80% of variance: {np.argmax(np.cumsum(explained_variance_ratio) >= 0.8) + 1}")
print(f"2. Explained variance ratio of first component: {explained_variance_ratio[0]:.2f}")
print(f"3. Explained variance ratio of second component: {explained_variance_ratio[1]:.2f}")
print(f"4. Cumulative explained variance ratio of first two components: {np.sum(explained_variance_ratio[:2]):.2f}")

