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

    move_turn_filter = range(0,10)

    boards = list(df["Board"])
    elos = list(df["Elo"])
    white_to_moves = list(df["WhiteToMove"])
    moves = list(df["move"])
    turn_numbers = list(df["TurnNumber"])

    for board_string, elo, white_to_move, move_string, turn_number_string in zip(
        boards, elos, white_to_moves, moves, turn_numbers
    ):

        turn_number = int(turn_number_string)

        if turn_number in move_turn_filter:

            boards_list = literal_eval(board_string)
            move_list = literal_eval(move_string)

            feature_tensor = get_feature_tensor(boards_list, elo, white_to_move)

            # modify to get to or from board !
            target_tensor = get_target_tensor(move_list, to_board=True)

            yield feature_tensor, target_tensor

def save_model(model, model_name):
    """Saves model"""


    file_name = "./weights/" + model_name

    while os.path.exists(file_name):
        file_name = file_name + "1"

    model.save_weights(file_name)


def main():
    """Main function"""

    full_run = True

    train_dataset = tf.data.Dataset.from_generator(
        move_generator,
        args=["learning_data/train_data.csv"],
        output_signature=(
            tf.TensorSpec(shape=(770,), dtype=tf.int32),
            tf.TensorSpec(shape=(64,), dtype=tf.int32),
        ),
    ).batch(32)

    if full_run:

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

    # for parameter in [0.99, 0.999, 1]:

    layer_array = []

    layers = [
        (512, 2), # Tested
        (256, 2), # Tested
        (128, 3),  # Tested
    ]

    for layer_tuple in layers:
        for _ in range(layer_tuple[1]):
            layer_array.append(tf.keras.layers.Dense(layer_tuple[0], activation="relu"))

    layer_array.append(tf.keras.layers.Dense(64, activation="softmax"))

    model = tf.keras.Sequential(layer_array)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=2e-4,  # Tested
            beta_1=0.6,          # Tested
            beta_2=0.999,        # Tested
            ),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.CategoricalAccuracy()],
    )

    if full_run:

        model.fit(
            train_dataset.repeat(),
            validation_data=test_dataset,
            epochs=10,
            steps_per_epoch=100000,
            validation_steps=10000,
        )

        model.evaluate(validate_dataset, steps=100000)

        save_model(model, "to_early_model")

    else:

        model.fit(
            train_dataset.repeat(),
            steps_per_epoch=10000,
        )

        # print(f"Parameter = {parameter}")

        model.evaluate(validate_dataset, steps=1000)



if __name__ == "__main__":
    main()
