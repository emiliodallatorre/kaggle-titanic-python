from fstpso import FuzzyPSO
from pandas import DataFrame
from sklearn.base import ClassifierMixin, accuracy_score
from sklearn.ensemble import RandomForestClassifier


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
    print(f"Accuracy with default hyperparameters: {
          accuracy_score(y_test, rf.predict(x_test))}")

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
