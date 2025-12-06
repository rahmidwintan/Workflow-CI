import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import mlflow
import mlflow.sklearn

# ==============================
# Load dataset
# ==============================
df = pd.read_csv("C:\Rahmi\kuliah\SEMESTER 7\ASAH\Eksperimen_SML_Rahmi-Dwi-Intan\Membangun_Model\StudentsPerformance_preprocessing\dataset_preprocessed.csv")

# Pisahkan fitur & label
X = df.drop("math score", axis=1)
y = df["math score"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# MLflow Tracking
# ==============================
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("students-performance-basic")

with mlflow.start_run():
    mlflow.sklearn.autolog()

    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)

    print("R2 Score:", r2)
    print("MAE:", mae)
