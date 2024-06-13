"""Predict a move using two pre-made models"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from typing import List, Tuple
from ast import literal_eval

import chess
import chess.svg
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import numpy as np
import pandas as pd

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

def parse_fen(fen: str) -> List[List[int]]:
    """
    Converts a fen string to bytes that represent fourteen boards.  Where the first
    64 bits is a bit mask of the chess board, where a 1 indicates that a white pawn
    exists there.  There should be fourteen different bit boards for all the different
    pieces, ordered as follows:
    * White Pawns
    * White Rooks
    * White Knights
    * White Bishops
    * White Queens
    * White Kings
    * Black Pawns
    * Black Rooks
    * Black Knights
    * Black Bishops
    * Black Queens
    * Black Kings
    """

    piece_types = [
        (chess.PAWN, chess.WHITE),
        (chess.ROOK, chess.WHITE),
        (chess.KNIGHT, chess.WHITE),
        (chess.BISHOP, chess.WHITE),
        (chess.QUEEN, chess.WHITE),
        (chess.KING, chess.WHITE),
        (chess.PAWN, chess.BLACK),
        (chess.ROOK, chess.BLACK),
        (chess.KNIGHT, chess.BLACK),
        (chess.BISHOP, chess.BLACK),
        (chess.QUEEN, chess.BLACK),
        (chess.KING, chess.BLACK),
    ]

    bit_boards = []

    board = chess.Board(fen=fen)

    for piece_type in piece_types:

        bit_boards.append(list(board.pieces(*piece_type)))

    return bit_boards

def load_model(filepath) -> tf.keras.Sequential:
    """Function call to return a model from disk"""

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

    model.load_weights(filepath)

    return model


def predict_move(
        features: tf.Tensor, 
        to_model: tf.keras.Sequential, 
        from_model : tf.keras.Sequential,
        board: chess.Board,
    ) -> chess.Move:
    """Predicts the next move to play"""

    features = tf.expand_dims(features, axis=0)

    to_logits = to_model.predict(features)[0]
    from_logits = from_model.predict(features)[0]

    to_squares = tf.argsort(to_logits).numpy().tolist()
    from_squares = tf.argsort(from_logits).numpy().tolist()

    move_list = []

    for i, to_square in enumerate(to_squares):
        for j, from_square in enumerate(from_squares):
            move_list.append((from_square, to_square, i+j))

    move_list = sorted(move_list, key=lambda x:x[2])

    for move in move_list:
        if chess.Move(move[0], move[1]) in board.legal_moves:
            return chess.Move(move[0], move[1])

def main():
    """Main function"""

    fen = chess.STARTING_FEN

    board = chess.Board(fen = fen)

    elo = 1000

    to_model = load_model("./weights/to_model")
    from_model = load_model("./weights/from_model")

    for _ in range(10):

        features = get_feature_tensor(parse_fen(board.fen()), elo, board.turn)

        move = predict_move(features, to_model, from_model, board)

        board.push(move)

    chess.svg.board(board)



if __name__ == "__main__":
    main()
