from pandas import DataFrame, read_csv


def read_training_data() -> DataFrame:
    return read_csv("data/train.csv")


def read_test_data() -> DataFrame:
    return read_csv("data/test.csv")


def write_output(output: DataFrame) -> None:
    output.to_csv("output.csv")
