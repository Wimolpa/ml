import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE

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

# แบ่งข้อมูลเป็น train และ test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

# สร้างและฝึก
best_n = 0
best_accuracy = 0
accuracy_scores = []
n_estimators = [1, 2, 3, 4, 5, 10, 100, 1000]  # ช่วงที่ลดลงเพื่อให้รันเร็วขึ้น

# ทดลองค่าต่าง ๆ ของ n_estimators


clf = RandomForestClassifier(n_estimators=1000, max_depth=9)
selector = RFE(clf, n_features_to_select=9, step=1)
selector = selector.fit(x_train, y_train)

y_pred = selector.predict(x_test)
accuracy_score = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy_score:.2f}")

target_names = ['Cloudy', 'Rainy', 'Sunny', 'Snowy']
clr = classification_report(y_test, y_pred, target_names=target_names)
print(clr)

# คำนวณ confusion matrix
cm = confusion_matrix(y_test, y_pred)

# กำหนดชื่อ labels สำหรับแถวและคอลัมน์
labels = le.inverse_transform(sorted(set(y)))  # แปลง labels กลับเป็นชื่อที่อ่านเข้าใจได้

# สร้าง Confusion Matrix Plot
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

print("Selected features:", selector.support_)