
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load preprocessed data
X_train = pd.read_csv('esrb_rating_dataset_preprocessing/X_train.csv')
X_test = pd.read_csv('esrb_rating_dataset_preprocessing/X_test.csv')
y_train = pd.read_csv('esrb_rating_dataset_preprocessing/y_train.csv').values.ravel()
y_test = pd.read_csv('esrb_rating_dataset_preprocessing/y_test.csv').values.ravel()

# Enable MLflow autologging
mlflow.sklearn.autolog()

with mlflow.start_run():
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc}")
