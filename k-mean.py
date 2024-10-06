import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

# โหลดข้อมูลจากไฟล์ CSV
df = pd.read_csv('weather_classification_data.csv')

# เลือกคอลัมน์ที่เป็น features และ target
x = df.drop('Weather Type', axis=1)
y = df['Weather Type']

# แปลงข้อมูล categorical ให้เป็นตัวเลข
le = LabelEncoder()

# ทำการแปลงคอลัมน์ที่เป็น string ทั้งหมดใน features
for column in x.columns:
    if x[column].dtype == 'object':
        x[column] = le.fit_transform(x[column])

# ทำการแปลง target (ถ้า target เป็น string)
if y.dtype == 'object':
    y = le.fit_transform(y)

# KMeans clustering
kmeans = KMeans(n_clusters=4).fit(x)

# ลดมิติของข้อมูลเหลือ 2 มิติ ด้วย PCA.
pca = PCA(n_components=2)
x_pca = pca.fit_transform(x)

# พล็อตกราฟ
plt.figure(figsize=(10, 7))

# พล็อตข้อมูล โดยใช้ labels ที่ได้จาก KMeans
plt.scatter(x_pca[:, 0], x_pca[:, 1], c=kmeans.labels_, cmap='viridis')

# ใส่ centroids ลงในกราฟ
centroids = pca.transform(kmeans.cluster_centers_)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=300, marker='x', label='Centroids')

plt.title('K-Means Clustering (PCA Reduced)')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend()
plt.show()
