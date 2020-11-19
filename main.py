import sys
import random
from dataclasses import dataclass
from pathlib import Path
from search import search
from board import Board
from typing import List


@dataclass
class Stats:
    total_length_solution = 0
    total_length_search = 0
    total_fail = 0
    total_cost = 0
    total_time = 0.0


def main(args: List[str]) -> None:
    """
    Program entry point
    :param args: Program arguments
    """

    # Get filename from args or set a default
    results_directory = "results/"
    Path(results_directory).mkdir(parents=True, exist_ok=True)

    filename = args[1] if len(args) > 1 else "samplePuzzles.txt"
    shape = (2, 4)
    with open(filename, "r") as f:
        for i, line in enumerate(f):
            # Load board
            data = list(map(int, line.split(' ')))

            # Uniform Cost Search
            board = Board.from_list(data, shape)
            write_results(board, f"{results_directory}{i}_ucs", show_h=False)

            # GBFS h1
            board = Board.from_list(data, shape, Board.h1, sort_g=False)
            write_results(board, f"{results_directory}{i}_gbfs-h1", show_f=False, show_g=False)

            # GBFS h2
            board = Board.from_list(data, shape, Board.h2, sort_g=False)
            write_results(board, f"{results_directory}{i}_gbfs-h2", show_f=False, show_g=False)

            # A* h1
            board = Board.from_list(data, shape, Board.h1)
            write_results(board, f"{results_directory}{i}_astar-h1")

            # A* h1
            board = Board.from_list(data, shape, Board.h2)
            write_results(board, f"{results_directory}{i}_astar-h2")

    ucs = Stats()
    gbfs_h1 = Stats()
    gbfs_h2 = Stats()
    astar_h1 = Stats()
    astar_h2 = Stats()
    with open(results_directory + "randomPuzzles.txt", "r") as f:
        for i, line in enumerate(f):
            print(f"Calculating random puzzle {i}")
            data = list(map(int, line.split(' ')))

            # Uniform Cost Search
            board = Board.from_list(data, shape)
            calculate_stats(board, ucs)

            # GBFS h1
            board = Board.from_list(data, shape, Board.h1, sort_g=False)
            calculate_stats(board, gbfs_h1)

            # GBFS h2
            board = Board.from_list(data, shape, Board.h2, sort_g=False)
            calculate_stats(board, gbfs_h2)

            # A* h1
            board = Board.from_list(data, shape, Board.h1)
            calculate_stats(board, astar_h1)

            # A* h2
            board = Board.from_list(data, shape, Board.h2)
            calculate_stats(board, astar_h2)

    write_stats(results_directory + "ucs_random_stats.txt", ucs)
    write_stats(results_directory + "gbfs_h1_random_stats.txt", gbfs_h1)
    write_stats(results_directory + "gbfs_h2_random_stats.txt", gbfs_h2)
    write_stats(results_directory + "astar_h1_random_stats.txt", astar_h1)
    write_stats(results_directory + "astar_h2_random_stats.txt", astar_h2)

    with open(results_directory + "largePuzzles.txt", "w") as f:
        print("Writing largePuzzles.txt")
        for i, shape in enumerate(((3, 4), (4, 4), (4, 5))):
            ucs = Stats()
            gbfs_h1 = Stats()
            gbfs_h2 = Stats()
            astar_h1 = Stats()
            astar_h2 = Stats()

            data = list(range(shape[0] * shape[1]))
            random.shuffle(data)
            f.write(" ".join(map(str, data)) + "\n")

            # Uniform Cost Search
            board = Board.from_list(data, shape)
            print(f"Calculating ucs_large{i}_stats.txt")
            calculate_stats(board, ucs, timeout=180.0)

            # GBFS h1
            board = Board.from_list(data, shape, Board.h1, sort_g=False)
            print(f"Calculating gbfs_h1_large{i}_stats.txt")
            calculate_stats(board, gbfs_h1, timeout=180.0)

            # GBFS h2
            board = Board.from_list(data, shape, Board.h2, sort_g=False)
            print(f"Calculating gbfs_h2_large{i}_stats.txt")
            calculate_stats(board, gbfs_h2, timeout=180.0)

            # A* h1
            board = Board.from_list(data, shape, Board.h1)
            print(f"Calculating astar_h1_large{i}_stats.txt")
            calculate_stats(board, astar_h1, timeout=180.0)

            # A* h2
            board = Board.from_list(data, shape, Board.h2)
            print(f"Calculating astar_h2_large{i}_stats.txt")
            calculate_stats(board, astar_h2, timeout=180.0)

            write_stats(results_directory + f"ucs_large{i}_stats.txt", ucs, averages=False)
            write_stats(results_directory + f"gbfs_h1_large{i}_stats.txt", gbfs_h1, averages=False)
            write_stats(results_directory + f"gbfs_h2_large{i}_stats.txt", gbfs_h2, averages=False)
            write_stats(results_directory + f"astar_h1_large{i}_stats.txt", astar_h1, averages=False)
            write_stats(results_directory + f"astar_h2_large{i}_stats.txt", astar_h2, averages=False)


def write_results(board: Board, filename: str, show_f: bool = True, show_g: bool = True, show_h: bool = True) -> None:
    time, result = search(board)
    if time is None:
        return

    with open(filename + "_solution.txt", "w") as f:
        print(f"Writing {filename}_solution.txt")
        if result is None:
            f.write("no solution")
        else:
            results = result[0]
            prev_target = 0
            prev_cost = 0
            while len(results.path) > 0:
                s = results.path.pop()
                f.write(f"{prev_target} {prev_cost} {str(s.board)}\n")
                prev_target = s.target
                prev_cost = s.cost

            f.write(f"{results.cost} {time}")

    with open(filename + "_search.txt", "w") as f:
        print(f"Writing {filename}_search.txt")
        if result is None:
            f.write("no solution")
        else:
            path = result[1]

            for state in path:
                f.write(f"{state.f if show_f else 0} {state.g if show_g else 0} {state.h if show_h else 0} {state.state}\n")


def calculate_stats(board: Board, stats: Stats, timeout: float = 60.0) -> None:
    time, result = search(board, timeout)
    if time is None:
        return

    if result is None:
        stats.total_fail += 1
        stats.total_time += timeout
    else:
        stats.total_time += time
        results = result[0]
        stats.total_length_solution += len(results.path)
        stats.total_cost += results.cost
        stats.total_length_search += len(result[1])


def write_stats(filename: str, stats: Stats, averages: bool = True) -> None:
    with open(filename, "w") as f:
        print("Writing " + filename)
        if averages:
            valid = 50 - stats.total_fail

        f.write(f"Total solution length: {stats.total_length_solution}\n")
        if averages:
            f.write(f"Average solution length: {stats.total_length_solution / valid}\n")

        f.write(f"Total search length: {stats.total_length_search}\n")
        if averages:
            f.write(f"Average search length: {stats.total_length_search / valid}\n")

        f.write(f"Total failures: {stats.total_fail}\n")
        if averages:
            f.write(f"Average failures: {stats.total_fail / 50}\n")

        f.write(f"Total cost: {stats.total_cost}\n")
        if averages:
            f.write(f"Average cost: {stats.total_cost / valid}\n")

        f.write(f"Total time: {stats.total_time}\n")
        if averages:
            f.write(f"Average time: {stats.total_time / 50}\n")


if __name__ == '__main__':
    main(sys.argv)
