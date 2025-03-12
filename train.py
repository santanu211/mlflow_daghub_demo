import mlflow
import mlflow.sklearn

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import dagshub
dagshub.init(repo_owner='santanu211', repo_name='mlflow_daghub_demo', mlflow=True)


mlflow.set_tracking_uri("https://github.com/santanu211/mlflow_daghub_demo.git")



# Load dataset
wine = load_wine()
X, y = wine.data, wine.target  # Features and target labels

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#parameter
n_estimators=100
random_state=2
mlflow.set_experiment("wine_rf")

with mlflow.start_run(run_name="santanu"):
    # Train Random Forest model
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy",accuracy)
    mlflow.log_param("n_estimators",n_estimators)
    mlflow.log_param("random_state",random_state)
    print(f'Accuracy: {accuracy:.2f}')
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    # save the plot as artifact
    plt.savefig("confusion_matrix.png")

    # log it
    mlflow.log_artifact("confusion_matrix.png")
    mlflow.log_artifact(__file__)
    
    mlflow.sklearn.log_model(model, "random_forest")
    mlflow.set_tag("author","santanu")



