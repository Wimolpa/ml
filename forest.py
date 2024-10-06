import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

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
for n in n_estimators:
    clf = RandomForestClassifier(n_estimators=n, max_depth=9)
    clf.fit(x_train, y_train)

    # ทำนายผล
    y_pred = clf.predict(x_test)

    # คำนวณค่า accuracy
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append((n, accuracy))  # เก็บข้อมูล n และ accuracy

    # เช็คว่าค่า accuracy สูงสุดหรือไม่
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_n = n

print(f"ค่าที่ดีที่สุดของ n_estimators: {best_n}")
print(f"Accuracy สูงสุด: {best_accuracy:.2f}")

target_names = ['Cloudy', 'Rainy', 'Sunny', 'Snowy']
clr = classification_report(y_test, y_pred, target_names=target_names)
print(clr)

# คำนวณ confusion matrix
cm = confusion_matrix(y_test, y_pred)

# กำหนดชื่อ labels สำหรับแถวและคอลัมน์
labels = le.inverse_transform(sorted(set(y)))  # แปลง labels กลับเป็นชื่อที่อ่านเข้าใจได้

# สร้าง Confusion Matrix Plot
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

