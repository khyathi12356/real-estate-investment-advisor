import joblib
from src.utils import load_data
from src.feature_engineering import create_features
from src.pipelines import build_preprocessor, build_regressor
from src.evaluate import evaluate_regression


def train_regressor():

    df = load_data("data/india_housing_prices.csv")
    df = create_features(df)

    X = df.drop(columns=["Good_Investment", "Future_Price"])
    y = df["Future_Price"]

    preprocessor = build_preprocessor(X)
    model = build_regressor(preprocessor)

    model.fit(X, y)

    preds = model.predict(X)

    print(evaluate_regression(y, preds))

    joblib.dump(model, "models/regressor.pkl")


if __name__ == "__main__":
    train_regressor()