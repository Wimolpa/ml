from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
    classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import make_pipeline

from dt import selector

data = load_breast_cancer()
x = data.data
y = data.target

feature_names = data.feature_names
selected_features = feature_names[selector.get_support()]
print(f"Selected Features using ANOVA: {selected_features}")

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

anova_filter = SelectKBest(score_func=f_classif, k=20)
clf = DecisionTreeClassifier()
anova_dt = make_pipeline(anova_filter,clf)
anova_dt.fit(x_train, y_train)


y_pred = anova_dt.predict(x_test)

print(classification_report(y_test,y_pred))


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)


print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")


