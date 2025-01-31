from pandas import DataFrame
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import repository


TARGET_COLUMN: str = "Survived"
IGNORED_COLUMNS: list[str] = ["PassengerId",
                              "Name", "Ticket", "Cabin", "Embarked"]


def input_transform(input: DataFrame) -> DataFrame:
    input.drop(columns=IGNORED_COLUMNS, inplace=True)
    input["Sex"] = input["Sex"].apply(
        lambda sex_str: 1 if sex_str == "male" else 0)

    return input


def main():
    # Data loading
    data_train: DataFrame = repository.read_training_data()
    data_test: DataFrame = repository.read_test_data()

    # Data cleaning
    data_train = input_transform(data_train)
    x: DataFrame = data_train.drop(columns=[TARGET_COLUMN])
    y: DataFrame = data_train[TARGET_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    # Prediction
    rf = RandomForestClassifier()
    rf.fit(x_train, y_train)
    y_pred = rf.predict(x_test)

    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)

    print(f"# Accuracy: {accuracy}")


if __name__ == "__main__":
    main()
