# Chess Engine

## Summary

This is a chess engine that predicts the next move from the state of the board and a rating.  The engine determines the most likely move that a human player at that rating will play.

## Data

The data was gotten from Kaggles [Analyzed Chess Games Dataset](https://www.kaggle.com/datasets/shkarupylomaxim/chess-games-dataset-lichess-2017-may) dataset.  The data is cleaned by converting the original csv into a preprocessed csv file, where the board is represented by 12 bit boards, and the played move is represented by two bitboards.  Each bitboard is made of 64 bits, corresponding to the 64 squares of the chess board.  The twelve bit boards that represent the position are used to represent the position of the 
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

respectfully.  The move that the user plays is represented by two bitboards, also of 64 bits, where the first bit board is the square that the piece was moved from, and the second bit board is the square that the piece was moved to.