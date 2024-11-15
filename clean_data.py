"""
This module creates the csv that can be placed used to train the neural network

The features should be 14 concatenated bit board, for each different piece.

The label is 2 bit boards, the from square and to square.
"""

# Imports
from typing import List, Tuple
from math import floor
import re


import chess # type: ignore #pylint: disable=import-error
import pandas as pd
import numpy as np
from tqdm import tqdm


class BoardState:
    """
    Class that represents a data point in the machine learning model.
    """

    def __init__(self, moves_string: str, white_elo: int, black_elo: int):

        self.white_elo: int = white_elo
        self.black_elo: int = black_elo
        self.moves: List[str] = self.__list_of_moves(moves_string)
        self.white_to_move: bool = True
        self.board: chess.Board = chess.Board()
        self.turn_number = 0

    def __peek_move(self) -> chess.Move:
        """Get next move"""

        return self.board.parse_san(self.moves[0])

    def __pop_move(self) -> chess.Move:
        """Get next move"""

        next_move = self.board.parse_san(self.moves[0])
        self.moves.pop(0)
        return next_move

    def make_next_move(self) -> None:
        """Makes the next move on the board"""

        move = self.__pop_move()
        self.board.push(move)
        self.white_to_move = not self.white_to_move
        if self.white_to_move:
            self.turn_number += 1

    def flip_board(self, boards : List[List[int]]) -> List[List[int]]:
        """Flips a board"""

        # Flips positions
        boards = [[63 - square for square in board] for board in boards]

        # # Flips colors
        temp = boards[:6]
        boards[:6] = boards[6:]
        boards[6:] = temp

        return boards

    def get_features(self) -> Tuple[List[List[int]], int, bool]:
        """Gets a tuple of features"""

        # TODO, change how the features work to always be from the perspective of white
        # Flip the board around if it is black to move, and report the board from blacks perspective
        # Easier way of doing this might be to throw out all black / white moves in training.

        boards = self.__parse_current_board()

        if not self.white_to_move:

            boards = self.flip_board(boards)

        return (
            boards,
            self.white_elo if self.white_to_move else self.black_elo,
        )

    def get_target(self) -> List[List[int]]:
        """Returns move boards"""

        move = self.__parse_move(self.__peek_move())

        if not self.white_to_move:
            move = self.flip_board(move)

        return move

    def no_more_moves(self) -> bool:
        """Returns a boolean that is true if no more moves are present"""

        return self.moves == []

    def __parse_current_board(self) -> List[List[int]]:
        """
        Converts a fen string to bytes that represent twelve boards.  Where the first
        64 bits is a bit mask of the chess board, where a 1 indicates that a white pawn
        exists there.  There should be twelve different bit boards for all the different
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

        for piece_type in piece_types:

            bit_boards.append(list(self.board.pieces(*piece_type)))

        return bit_boards

    def __parse_move(self, move: chess.Move) -> List[List[int]]:
        """
        Converts a move in standard algebraic notation to a series of bytes that
        represent two bit boards, where the first bit board is the square that the
        piece moves from, and the second bit board is the square that the piece moves to.
        """

        from_square = move.from_square
        to_square = move.to_square

        squares = [[from_square], [to_square]]

        return squares

    def __list_of_moves(self, moves: str) -> List[str]:
        """
        Given the string format from the dataset,
        return a list of moves in standard algebraic notation.
        """

        moves = re.sub(r"\{[^}]*\}", "", moves)
        moves = re.sub(r"[?!]", "", moves)

        return moves.split()


def main() -> None:
    """Main Function"""

    DEBUG = False # pylint: disable=invalid-name

    games = pd.read_csv(
        "games_metadata_profile.csv",
        usecols=[
            "WhiteElo",
            "BlackElo",
            "Moves",
        ],
        dtype={
            "WhiteElo": int,
            "BlackElo": int,
            "Moves": str,
        },
    )

    boards = []
    elos = []
    # white_to_moves = []
    targets = []
    turn_numbers = []

    for white_elo, black_elo, moves in tqdm(
        zip(games["WhiteElo"], games["BlackElo"], games["Moves"])
    ):

        # For each game initialize a new board

        board_state = BoardState(moves, white_elo, black_elo)

        while not board_state.no_more_moves():
            # For each board save data to lists

            features = board_state.get_features()
            boards.append(features[0])
            elos.append(features[1])
            # white_to_moves.append(features[2])
            turn_numbers.append(board_state.turn_number)
            targets.append(board_state.get_target())

            board_state.make_next_move()

        if DEBUG:
            break

        if len(boards) > 1000000:
            break

    if DEBUG:
        print(boards)
        print(elos)
        # print(white_to_moves)
        print(targets)
        print(turn_numbers)

    if not DEBUG:

        # Calculate indexes for train test validate csvs

        train_split = 0.7
        test_split = 0.2

        idxs = np.arange(len(boards))

        train_idxs = np.random.choice(
            idxs,
            size=floor(len(idxs) * train_split),
            replace=False,
        )

        test_and_validate_idxs = np.delete(idxs, train_idxs)

        test_idxs = np.random.choice(
            test_and_validate_idxs,
            size=floor(len(test_and_validate_idxs) * test_split / (1 - train_split)),
            replace=False,
        )

        validate_idxs = np.delete(idxs, np.append(train_idxs, test_idxs))

        # create dataframes with the data and indexes

        train_data = pd.DataFrame(
            {
                "Board": [boards[i] for i in train_idxs],
                "Elo": [elos[i] for i in train_idxs],
                # "WhiteToMove": [white_to_moves[i] for i in train_idxs],
                "move": [targets[i] for i in train_idxs],
                "TurnNumber": [turn_numbers[i] for i in train_idxs],
            }
        )

        test_data = pd.DataFrame(
            {
                "Board": [boards[i] for i in test_idxs],
                "Elo": [elos[i] for i in test_idxs],
                # "WhiteToMove": [white_to_moves[i] for i in test_idxs],
                "move": [targets[i] for i in test_idxs],
                "TurnNumber": [turn_numbers[i] for i in test_idxs],
            }
        )

        validate_data = pd.DataFrame(
            {
                "Board": [boards[i] for i in validate_idxs],
                "Elo": [elos[i] for i in validate_idxs],
                # "WhiteToMove": [white_to_moves[i] for i in validate_idxs],
                "move": [targets[i] for i in validate_idxs],
                "TurnNumber": [turn_numbers[i] for i in validate_idxs],
            }
        )

        # Save dataframes

        train_data.to_csv("learning_data/train_data.csv")
        test_data.to_csv("learning_data/test_data.csv")
        validate_data.to_csv("learning_data/validate_data.csv")


if __name__ == "__main__":

    main()
