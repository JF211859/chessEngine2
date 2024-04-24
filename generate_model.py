"""
File to create the chess engine model.  
"""

# Imports
from ast import literal_eval
from typing import List
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
import tensorflow as tf
import numpy as np


def parse_boards(boards : List[int]) -> np.ndarray:
    """Convert the format that the data is stored in, to the format that is useful for the machine leanring"""

    board_array = np.zeros((len(boards), 8, 8), dtype=np.int32)

    for i, board in enumerate(boards):

        for piece in board:

            board_array[i, piece // 8, piece % 8] = 1
    
    return board_array.flatten()


def get_feature_tensor(boards : List[int], elo : int, white_to_move : bool) -> tf.Tensor:
    """Generate the feature tensors from a list of boards, the elo, and whose turn it is"""

    boards_array = parse_boards(boards)
    feature_array = np.concatenate((
        boards_array,
        np.array([elo], dtype=np.int32),
        np.array([1] if white_to_move else [0], dtype=np.int32),
    ))

    return tf.constant(feature_array, dtype = tf.int32)

def get_target_tensor(boards : List[int]) -> tf.Tensor:
    """Generate the feature tensors from a list of boards, the from board and the to board"""

    return tf.constant(parse_boards(boards), dtype = np.int32)


def move_generator(filename : str):
    """Yields a feature target pair"""

    df = pd.read_csv(
        str(filename, encoding="utf-8"),
        index_col=0,
        # nrows=10000,
    )

    boards = list(df["Board"])
    elos = list(df["Elo"])
    white_to_moves = list(df["WhiteToMove"])
    moves = list(df["move"])

    for board_string, elo, white_to_move, move_string in zip(
        boards, elos, white_to_moves, moves):

        boards_list = literal_eval(board_string)
        move_list = literal_eval(move_string)

        feature_tensor = get_feature_tensor(boards_list, elo, white_to_move)

        target_tensor = get_target_tensor(move_list)

        yield feature_tensor, target_tensor


class CustomLossFunction(tf.keras.losses.Loss):
    """Custom loss function"""

    def __init__(self):
        """Boilerplate"""
        super().__init__()

    def call(self, y_true, y_pred):
        """Splits both tensors in half, then does cross entropy on each halves, and summs the losses."""

        from_y_true, to_y_true = tf.split(y_true, [64, 64], 1)
        from_y_pred, to_y_pred = tf.split(y_pred, [64, 64], 1)

        return tf.keras.losses.categorical_crossentropy(from_y_true, from_y_pred) + tf.keras.losses.categorical_crossentropy(to_y_true, to_y_pred)
    

class CustomAccuracyFunction(tf.keras.metrics.Metric):
    """Custom accuracy function"""

    def __init__(self):
        """Boilerplate"""
        super().__init__()
        self.correct = self.add_weight(
            shape=(),
            initializer='zeros',
            name='correct'
        )
        self.total = self.add_weight(
            shape=(),
            initializer='zeros',
            name='total'
        )

    # https://stackoverflow.com/questions/36530944/how-to-get-the-count-of-an-element-in-a-tensor-in-tensorflow
    def _tf_count(self, t, val):
        elements_equal_to_value = tf.equal(t, val)
        as_ints = tf.cast(elements_equal_to_value, tf.int32)
        count = tf.reduce_sum(as_ints)
        return count

    @tf.autograph.experimental.do_not_convert
    def update_state(self, y_true, y_pred, sample_weight=None):
        """Splits both tensors in half, then does cross entropy on each halves, and summs the losses."""

        y_pred = tf.cast(y_pred, dtype=tf.float32)
        y_true = tf.cast(y_true, dtype=tf.float32)

        total = tf.shape(y_true)[0]

        from_y_true, to_y_true = tf.split(y_true, [64, 64], 1)
        from_y_pred, to_y_pred = tf.split(y_pred, [64, 64], 1)

        from_correct = tf.keras.metrics.categorical_accuracy(from_y_true, from_y_pred)
        to_correct = tf.keras.metrics.categorical_accuracy(to_y_true, to_y_pred)

        num_correct = self._tf_count(tf.math.add(from_correct, to_correct), 2)

        num_correct = tf.cast(num_correct, dtype=tf.float32)

        self.correct.assign(self.correct + num_correct)

        total = tf.cast(total, dtype=tf.float32)

        self.total.assign(self.total + total)


    def result(self):
        return self.correct / self.total



def main():
    """Main function"""

    train_dataset = tf.data.Dataset.from_generator(
        move_generator,
        args=["learning_data/train_data.csv"],
        output_signature=(
            tf.TensorSpec(shape=(770,), dtype=tf.int32),
            tf.TensorSpec(shape=(128,), dtype=tf.int32),
        )
    ).shuffle(5000).batch(32)

    test_dataset = tf.data.Dataset.from_generator(
        move_generator,
        args=["learning_data/test_data.csv"],
        output_signature=(
            tf.TensorSpec(shape=(770,), dtype=tf.int32),
            tf.TensorSpec(shape=(128,), dtype=tf.int32),
        )
    ).batch(32)

    validate_dataset = tf.data.Dataset.from_generator(
        move_generator,
        args=["learning_data/validate_data.csv"],
        output_signature=(
            tf.TensorSpec(shape=(770,), dtype=tf.int32),
            tf.TensorSpec(shape=(128,), dtype=tf.int32),
        )
    ).batch(32)

    model = tf.keras.Sequential([
        # tf.keras.Input(shape=(770,)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=CustomLossFunction(),
        # loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[
            CustomAccuracyFunction(),
            # tf.keras.metrics.CategoricalAccuracy()
        ],
    )


    model.fit(
        train_dataset.repeat(),
        validation_data=test_dataset,
        epochs=10,
        steps_per_epoch=100000,
        # steps_per_epoch=100,
        # validation_steps=100,
    )

    model.evaluate(
        validate_dataset
    )

    model.save_weights('./weights/custom_loss')



if __name__ == "__main__":
    main()
