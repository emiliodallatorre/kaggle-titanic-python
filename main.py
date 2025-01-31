import random
from pandas import DataFrame
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from fstpso import FuzzyPSO
import re
import repository


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


def prepare_model(x_train: DataFrame, y_train: DataFrame, x_test: DataFrame, y_test: DataFrame) -> ClassifierMixin:
    available_hyperparams: dict[str, list] = {
        "n_estimators": [100, 1000],
        "max_depth": [3, 30],
        "min_samples_split": [2, 20],
        "min_samples_leaf": [1, 10],
    }

    def evaluate_hyperparameters(particle):
        hyperparams = {key: particle[idx] for idx, key in enumerate(
            available_hyperparams.keys())}

        rf.set_params(**hyperparams)
        rf.fit(x_train, y_train)

        y_pred = rf.predict(x_test)
        return accuracy_score(y_test, y_pred)

    rf = RandomForestClassifier()
    rf.fit(x_train, y_train)
    print(f"Accuracy with default hyperparameters: {accuracy_score(y_test, rf.predict(x_test))}")

    search_space = list(map(lambda boundaries: list(range(
        boundaries[0], boundaries[1])), available_hyperparams.values()))

    FP = FuzzyPSO()
    FP.set_search_space_discrete(search_space)
    FP.set_fitness(evaluate_hyperparameters)
    result = FP.solve_with_fstpso()

    print("Best solution:", result[0])
    print("Whose fitness is:", result[1])

    rf.fit(x_train, y_train)

    return rf


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
    model = prepare_model(x_train, y_train, x_test, y_test)
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
