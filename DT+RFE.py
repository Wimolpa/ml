from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

data = load_breast_cancer()
x = data.data
y = data.target

clf = DecisionTreeClassifier()

n_features_to_select = 25
rfe = RFE(estimator=clf, n_features_to_select=n_features_to_select)
x_rfe = rfe.fit_transform(x, y)

x_train, x_test, y_train, y_test = train_test_split(x_rfe, y, test_size=0.20, random_state=42)

clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")


print(f"Support (ฟีเจอร์ที่ถูกเลือก): {rfe.support_}")
print(f"Ranking (อันดับของฟีเจอร์): {rfe.ranking_}")
