from src.train_classification import train_classifier
from src.train_regression import train_regressor

def menu():
    print("1. Train Classification")
    print("2. Train Regression")
    print("3. Train All")

    choice = input("Enter choice: ")

    if choice == "1":
        train_classifier()
    elif choice == "2":
        train_regressor()
    else:
        train_classifier()
        train_regressor()

if __name__ == "__main__":
    menu()