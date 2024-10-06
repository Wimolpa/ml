import pandas as pd
from sklearn.cluster import KMeans ,AgglomerativeClustering
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import adjusted_rand_score, homogeneity_score

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

clustering = AgglomerativeClustering(n_clusters=4).fit(x)


kmeans = KMeans(n_clusters=4).fit(x)


# for i in range(len(y)):
#     print(clustering.labels_[i],"\t",kmeans.labels_[i],"\t" ,y[i])

Z = linkage(x, method='ward')


# ค่า adjusted rand index สำหรับ Agglomerative Clustering
ari_agglomerative = adjusted_rand_score(y, clustering.labels_)
# ค่า adjusted rand index สำหรับ KMeans
ari_kmeans = adjusted_rand_score(y, kmeans.labels_)

# ค่า homogeneity score สำหรับ Agglomerative Clustering
homogeneity_agglomerative = homogeneity_score(y, clustering.labels_)
# ค่า homogeneity score สำหรับ KMeans
homogeneity_kmeans = homogeneity_score(y, kmeans.labels_)

print(f"Agglomerative Clustering ")
print(f"ARI: {ari_agglomerative}")
print(f"Homogeneity: {homogeneity_agglomerative}")
print(f"KMeans Clustering ")
print(f"ARI: {ari_kmeans}")
print(f"Homogeneity: {homogeneity_kmeans}")

plt.figure(figsize=(10, 7))
dendrogram(Z, labels=y, leaf_rotation=90, leaf_font_size=16)
plt.title('Dendrogram for Weather Dataset')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()

