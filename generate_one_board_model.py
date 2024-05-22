"""
File to create the chess engine model. For just to or from board
"""

# Train Size : 3428414
# Test Size : 979546
# Validate Size : 489774

# Imports

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from typing import List
from ast import literal_eval

import pandas as pd
import tensorflow as tf
import numpy as np


def parse_boards(boards: List[int]) -> np.ndarray:
    """Convert the format that the data is stored in, to the format that is useful for the machine leanring"""

    board_array = np.zeros((len(boards), 8, 8), dtype=np.int32)

    for i, board in enumerate(boards):

        for piece in board:

            board_array[i, piece // 8, piece % 8] = 1

    return board_array.flatten()


def get_feature_tensor(boards: List[int], elo: int, white_to_move: bool) -> tf.Tensor:
    """Generate the feature tensors from a list of boards, the elo, and whose turn it is"""

    boards_array = parse_boards(boards)
    feature_array = np.concatenate(
        (
            boards_array,
            np.array([elo], dtype=np.int32),
            np.array([1] if white_to_move else [0], dtype=np.int32),
        )
    )

    return tf.constant(feature_array, dtype=tf.int32)


def get_target_tensor(boards: List[int], to_board: bool) -> tf.Tensor:
    """Generate the feature tensors from a list of boards, the from board and the to board"""

    if to_board:
        return tf.constant(parse_boards([boards[0]]), dtype=np.int32)
    return tf.constant(parse_boards([boards[1]]), dtype=np.int32)


def move_generator(filename: str):
    """Yields a feature target pair"""

    df = pd.read_csv(
        str(filename, encoding="utf-8"),
        index_col=0,
    )

    boards = list(df["Board"])
    elos = list(df["Elo"])
    white_to_moves = list(df["WhiteToMove"])
    moves = list(df["move"])

    for board_string, elo, white_to_move, move_string in zip(
        boards, elos, white_to_moves, moves
    ):

        boards_list = literal_eval(board_string)
        move_list = literal_eval(move_string)

        feature_tensor = get_feature_tensor(boards_list, elo, white_to_move)

        # modify to get to or from board !
        target_tensor = get_target_tensor(move_list, to_board=True)

        yield feature_tensor, target_tensor


def main():
    """Main function"""

    train_dataset = tf.data.Dataset.from_generator(
        move_generator,
        args=["learning_data/train_data.csv"],
        output_signature=(
            tf.TensorSpec(shape=(770,), dtype=tf.int32),
            tf.TensorSpec(shape=(64,), dtype=tf.int32),
        ),
    ).batch(32)

    test_dataset = tf.data.Dataset.from_generator(
        move_generator,
        args=["learning_data/test_data.csv"],
        output_signature=(
            tf.TensorSpec(shape=(770,), dtype=tf.int32),
            tf.TensorSpec(shape=(64,), dtype=tf.int32),
        ),
    ).batch(32)

    validate_dataset = tf.data.Dataset.from_generator(
        move_generator,
        args=["learning_data/validate_data.csv"],
        output_signature=(
            tf.TensorSpec(shape=(770,), dtype=tf.int32),
            tf.TensorSpec(shape=(64,), dtype=tf.int32),
        ),
    ).batch(32)

    for parameter in [4, 8, 16, 32]:

        layer_array = []

        layers = [
            (512, 4),
            (256, 4),
            (128, 4)
        ]

        for layer_tuple in layers:
            for _ in range(layer_tuple[1]):
                layer_array.append(tf.keras.layers.Dense(layer_tuple[0], activation="relu"))

        layer_array.append(tf.keras.layers.Dense(64, activation="sigmoid"))

        model = tf.keras.Sequential(layer_array)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=[tf.keras.metrics.CategoricalAccuracy()],
        )

        # Test Run
        model.fit(
            train_dataset.repeat(),
            steps_per_epoch=100000,
        )

        print(f"Parameter = {parameter}")

        model.evaluate(validate_dataset, steps=10000)

        # Full Run
        # model.fit(
        #     train_dataset.repeat(),
        #     validation_data=test_dataset,
        #     epochs=10,
        #     steps_per_epoch=100000,
        #     validation_steps=10000,
        # )

        # model.evaluate(validate_dataset, steps=100000)


if __name__ == "__main__":
    main()
