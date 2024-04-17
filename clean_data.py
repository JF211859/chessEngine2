"""
This module creates the csv that can be placed used to train the nueral network

The features should be 14 concatenated bitboards, for each different piece.

The label is 2 bitbords, the from square and to square.
"""

# Imports
from typing import List
import re


import pandas as pd
import chess


def bit_board_to_bytes(on_bits: List[List[int]]) -> List[bytes]:
    """
    Convert the chess libraries interpretation of squares to bytes.
    Expects a list of integers and converts it to bytes.
    """

    return_bytes : List[bytes] = []

    for board_bits in on_bits:

        total = 0

        for number in board_bits:

            total += 1 << number

        return_bytes.append(total.to_bytes(8, 'big'))

    return return_bytes


def fen_to_bytes(board: chess.Board) -> bytes:
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

    return None


def move_to_bytes(board: chess.Board, san: str) -> bytes:
    """
    Converts a move in standard algebraic notation to a series of bytes that
    represent two bit boards, where the first bit board is the sqaure that the
    piece moves from, and the second bit board is the square that the piece moves to.
    """

    move = board.parse_san(san)

    from_square = move.from_square
    to_square = move.to_square

    on_bits = [[from_square], [to_square]]

    print(on_bits)
    print(bit_board_to_bytes(on_bits))

    return bit_board_to_bytes(on_bits)


def next_board(current_board: chess.Board, move: str) -> chess.Board:
    """
    Given a fen and a move in standard notation, creates a new fen representing the state
    of the board after move is played.
    """

    current_board.push_san(move)

    return current_board


def list_of_moves(moves: str) -> List[str]:
    """
    Given the string format from the dataset, return a list of moves in standard algebraic notation.
    """

    moves = re.sub(r"\{[^}]*\}", "", moves)
    moves = re.sub(r"[?!]", "", moves)

    return moves.split()


def main() -> None:
    """Main Function"""

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

    for white_elo, black_elo, moves in zip(
        games["WhiteElo"], games["BlackElo"], games["Moves"]
    ):

        moves = list_of_moves(moves)
        board = chess.Board()
        position_bytes : List[bytes] = []
        move_bytes : List[bytes] = []
        white_to_move = True

        for move in moves:

            # position_bytes.append(move_to_bytes(board, move))
            move_bytes.append(move_to_bytes(board, move))

            board = next_board(board, move)

            white_to_move = not white_to_move

        break


if __name__ == "__main__":

    main()
