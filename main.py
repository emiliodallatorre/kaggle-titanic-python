from pandas import DataFrame
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import re
import repository
import brain


TARGET_COLUMN: str = "Survived"
IDENTITY_COLUMN: str = "PassengerId"
IGNORED_COLUMNS: list[str] = ["Name", "Cabin"]
MAPPABLE_COLUMNS: list[str] = ["Embarked", "Sex", "Ticket"]


def input_transform(input: DataFrame) -> DataFrame:
    input.drop(columns=IGNORED_COLUMNS, inplace=True)

    TICKET_SPLIT_REGEX: str = r"(.*?) (\d+)"
    input["Ticket"] = input["Ticket"].apply(
        lambda ticket_code: re.match(TICKET_SPLIT_REGEX, ticket_code).group(
            1) if re.match(TICKET_SPLIT_REGEX, ticket_code) else None
    )
    print(input["Ticket"])

    # Map labels to the DataFrame
    for column in MAPPABLE_COLUMNS:
        input[column] = input[column].map(
            {val: idx for idx, val in enumerate(sorted(input[column].dropna().unique()))})

    return input


def main():
    # Data loading
    data_test: DataFrame = repository.read_test_data()
    data_train: DataFrame = repository.read_training_data()

    # Data cleaning
    data_test = input_transform(data_test)

    data_train = input_transform(data_train)
    data_train.drop(columns=[IDENTITY_COLUMN], inplace=True)
    x: DataFrame = data_train.drop(columns=[TARGET_COLUMN])
    y: DataFrame = data_train[TARGET_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    # Prediction
    model = brain.prepare_model(x_train, y_train, x_test, y_test)
    y_pred = model.predict(x_test)

    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    print(f"# Accuracy: {accuracy}")

    # Output
    output_data = DataFrame(
        {"PassengerId": data_test["PassengerId"], TARGET_COLUMN: model.predict(data_test.drop(columns=[IDENTITY_COLUMN]))})
    repository.write_output(output_data)


if __name__ == "__main__":
    main()
