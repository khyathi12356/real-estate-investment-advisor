import joblib
from src.utils import load_data
from src.feature_engineering import create_features
from src.pipelines import build_preprocessor, build_classifier
from src.evaluate import evaluate_classification


def train_classifier():

    df = load_data("data/india_housing_prices.csv")
    df = create_features(df)

    X = df.drop(columns=["Good_Investment", "Future_Price"])
    y = df["Good_Investment"]

    preprocessor = build_preprocessor(X)
    model = build_classifier(preprocessor)

    model.fit(X, y)

    preds = model.predict(X)

    print(evaluate_classification(y, preds))

    joblib.dump(model, "models/classifier.pkl")


if __name__ == "__main__":
    train_classifier()