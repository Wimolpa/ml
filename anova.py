import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import seaborn as sns

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

# เก็บค่า accuracy ที่ดีที่สุดและค่า k ที่เหมาะสม
best_k = 0
best_accuracy = 0

# ลูปค่า k ตั้งแต่ 1 ถึงจำนวนคุณลักษณะทั้งหมด
for k in range(1, x_train.shape[1] + 1):
    # ใช้ ANOVA F-test เพื่อเลือก k คุณลักษณะที่ดีที่สุด
    selector = SelectKBest(f_classif, k=k)  # เปลี่ยนจาก best_k เป็น k
    clf = RandomForestClassifier(n_estimators=1000, max_depth=9)
    anova_RF = make_pipeline(selector, clf)
    anova_RF.fit(x_train, y_train)

    y_pred = anova_RF.predict(x_test)

    # คำนวณค่า accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # แสดงผลค่า k และ accuracy
    print(f"k={k}, Accuracy={accuracy}")

    # เช็คว่าค่า accuracy นี้สูงสุดหรือไม่
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_k = k

# แสดงผลลัพธ์ที่ดีที่สุด
print(f'\nBest k = {best_k} \nAccuracy = {best_accuracy}\n')

# ใช้ค่า k ที่ดีที่สุดในการแสดงผล Confusion Matrix และ Classification Report

selector = SelectKBest(f_classif, k=best_k)
clf = RandomForestClassifier(n_estimators=1000, max_depth=9)
anova_RF = make_pipeline(selector ,clf)
anova_RF.fit(x_train,y_train)

y_pred = anova_RF.predict(x_test)
target_names = ['Cloudy', 'Rainy', 'Sunny', 'Snowy']
print(classification_report(y_test,y_pred,target_names=target_names))

# สร้าง Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# แสดง Confusion Matrix ด้วย heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


