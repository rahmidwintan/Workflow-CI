import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import argparse
import mlflow

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv("MLProject/StudentsPerformance_preprocessing/dataset_preprocessed.csv")

    X = df.drop(columns="math score", axis=1)
    y = df["math score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)

    mlflow.log_metric("r2", score)
    mlflow.sklearn.log_model(model, "model")

    print("Training complete. R2:", score)


if __name__ == "__main__":
    main()


