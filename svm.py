import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn import svm

# โหลดข้อมูลจากไฟล์ CSV
df = pd.read_csv('weather_classification_data.csv')

# เลือกคอลัมน์ที่เป็น features และ target
x = df.drop('Weather Type', axis=1)
y = df['Weather Type']


# แปลงข้อมูล categorical ให้เป็นตัวเลข
le = LabelEncoder()

# ทำการแปลงคอลัมน์ที่เป็น string ทั้งหมดใน features
for column in x.columns:
    if x[column].dtype == 'object':  # ตรวจสอบว่าเป็น string หรือไม่
        x[column] = le.fit_transform(x[column])

# ทำการแปลง target (ถ้า target เป็น string)
if y.dtype == 'object':
    y = le.fit_transform(y)

# แบ่งข้อมูลเป็น train และ test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

# สร้างและฝึก
clf = svm.SVC(kernel='rbf')
clf = clf.fit(x_train, y_train)

# ทำนายผล
y_pred = clf.predict(x_test)

# คำนวณคะแนน
accuracy = accuracy_score(y_test, y_pred)
macro_precision = precision_score(y_test, y_pred, average='macro')
macro_recall = recall_score(y_test, y_pred, average='macro')
macro_f1 = f1_score(y_test, y_pred, average='macro')


# แสดงผลลัพธ์
print(f"Accuracy: {accuracy:.2f}")
print(f"Macro-average F1-score: {macro_f1:.2f}")

target_names = ['Cloudy', 'Rainy', 'Sunny', 'Snowy']
clr = classification_report(y_test, y_pred, target_names=target_names)
print(clr)

# คำนวณ confusion matrix
cm = confusion_matrix(y_test, y_pred)

# กำหนดชื่อ labels สำหรับแถวและคอลัมน์
labels = le.inverse_transform(sorted(set(y_test)))  # แปลง labels กลับเป็นชื่อที่อ่านเข้าใจได้

print(df.isnull().sum())

# สร้าง Confusion Matrix Plot
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()



