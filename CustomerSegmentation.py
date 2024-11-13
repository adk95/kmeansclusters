import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

url = 'customer_segmentation.csv' 
data = pd.read_csv(url)


data = data.drop(columns=['ID', 'Dt_Customer', 'Z_CostContact', 'Z_Revenue'])


data = data.dropna()


data['Age'] = 2024 - data['Year_Birth']
data = data.drop(columns=['Year_Birth'])

data = pd.get_dummies(data, columns=['Education', 'Marital_Status'], drop_first=True)


features = data.drop(columns=['Response'])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)


wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)


plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o', color='b')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


k = 4
kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=42)
data['Cluster'] = kmeans.fit_predict(X_scaled)


plt.figure(figsize=(10, 7))
sns.scatterplot(
    x='Income', y='MntWines', 
    hue='Cluster', data=data, palette='Set1', s=100
)
plt.title('Customer Segmentation based on Income and Wine Spending')
plt.xlabel('Annual Income')
plt.ylabel('Wine Spending')
plt.legend(title='Cluster')
plt.show()


silhouette_avg = silhouette_score(X_scaled, data['Cluster'])
print(f'Silhouette Score: {silhouette_avg:.2f}')


summary = data.groupby('Cluster').mean()
print(summary)


data.to_csv('Customer_Segmentation_Results.csv', index=False)
