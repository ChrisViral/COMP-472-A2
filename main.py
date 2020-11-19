import sys
from board import Board
from typing import List


def main(args: List[str]) -> None:
    """
    Program entry point
    :param args: Program arguments
    """

    # Get filename from args or set a default
    filename = args[1] if len(args) > 1 else "samplePuzzles.txt"
    with open(filename, "r") as f:
        for line in f:
            # Load board
            data = list(map(int, line.split(' ')))
            shape = (2, 4)
            board = Board.from_list(data, shape, None)
            board = Board.from_list(data, shape, Board.h1)
            board = Board.from_list(data, shape, Board.h2)


if __name__ == '__main__':
    main(sys.argv)
