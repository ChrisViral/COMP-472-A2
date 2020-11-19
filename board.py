from __future__ import annotations
import numpy as np
from numpy import ndarray as Array
from typing import List, Dict, Tuple, Iterable, NamedTuple, Union, Optional


class Point(NamedTuple):
    """
    Position tuple
    """
    x: int = 0
    y: int = 0

    def __add__(self, other: Union[Point, Tuple[int, int]]) -> Point:
        """
        Component wise add onto the point
        :param other: Other point to add
        :return: The resulting point
        """

        return Point(self.x + other[0], self.y + other[1])

    def __sub__(self, other: Union[Point, Tuple[int, int]]) -> Point:
        """
        Component wise subtract onto the point
        :param other: Other point to subtract
        :return: The resulting point
        """

        return Point(self.x - other[0], self.y - other[1])

    def __mod__(self, other: Union[Point, Tuple[int, int]]) -> Point:
        """
        Component wise modulo onto the point
        :param other: Other point to mod by
        :return: The resulting point
        """

        return Point(self.x % other[0], self.y % other[1])

    @staticmethod
    def manhattan_distance(a: Point, b: Point) -> Point:
        """
        Manhattan distance between two points
        :param a: First point
        :param b: Second point
        :return: The distance between the points
        """

        return Point(abs(a.x - b.x), abs(a.y - b.y))


class State(NamedTuple):
    """
    Board state, consisting of cost and associated board
    """

    cost: int
    board: Board


class Board:
    """
    Chi-Puzzle board
    """

    def __init__(self, array: Array, zero: Point, goals: Tuple[Array, ...]) -> None:
        """
        Creates a new board with the specified parameters
        :param array: Board array
        :param zero:  Position of the zero
        :param goal:  Tuple of goal states
        """

        self.height: int
        self.width: int
        self.height, self.width = array.shape
        self.array = array
        self.zero = zero
        self.goals = goals

    @property
    def is_goal(self) -> bool:
        """
        Checks if this board is in a goal state
        :return: True if this is a goal state, False otherwise
        """
        return any(np.array_equal(self.array, goal) for goal in self.goals)

    # region Move generation
    def generate_moves(self) -> Iterable[State]:
        """
        Generates all possible moves from the current board state
        :return: Iterable of all the possible moves
        """

        targets: Dict[Point, State] = {}

        # Check cardinal direction moves
        self._check_cardinal(self.zero + (1, 0), targets)
        self._check_cardinal(self.zero - (1, 0), targets)
        self._check_cardinal(self.zero + (0, 1), targets)
        self._check_cardinal(self.zero - (0, 1), targets)

        # Check the corner moves
        max_x = self.height - 1
        max_y = self.width - 1
        if self.zero in ((0, 0), (max_x, max_y)):
            # Top left/bottom right corners
            self._check_corner(self.zero + (1, 1), targets)
            self._check_corner(self.zero - (1, 1), targets)
        elif self.zero in ((max_x, 0), (0, max_y)):
            # Bottom left/top right corners
            self._check_corner(self.zero + (1, -1), targets)
            self._check_corner(self.zero - (1, -1), targets)

        return targets.values()

    def _check_cardinal(self, target: Point, targets: Dict[Point, State]) -> None:
        """
        Checks for wrapping on a cardinal move and adjusts the position and cost
        :param target:  Target move
        :param targets: Known targets so far
        """

        # Check if we're wrapping with this move
        if target.x in (-1, self.height) or target.y in (-1, self.width):
            cost = 2
            target %= self.array.shape
        else:
            cost = 1
        self._check_target(target, cost, targets)

    def _check_corner(self, target: Point, targets: Dict[Point, State]) -> None:
        """
        Adjusts the wrapping on corner moves and sets the correct cost
        :param target:  Target move
        :param targets: Known targets so far
        """

        # Adjust wrapping bounds
        target %= self.array.shape
        self._check_target(target, 3, targets)

    def _check_target(self, target: Point, cost: int, targets: Dict[Point, State]) -> None:
        """
        Validates the target move and adds it to or adjusts known targets if possible
        :param target:  Target move
        :param cost:    Move cost
        :param targets: Known targets so far
        """

        # Check if not in targets
        if target not in targets:
            # Copy array, then apply the move
            a = self.array.copy()
            a[self.zero], a[target] = a[target], a[self.zero]
            board = Board(a, target, self.goals)
        else:
            # Check if we have a lower cost
            state = targets[target]
            if cost >= state.cost:
                # If not do nothing
                return

            # Reuse the same board if possible
            board = state.board

        # Create state
        targets[target] = State(cost, board)
    # endregion

    # region Heuristics
    @staticmethod
    def h0(self: Board) -> int:
        """
        Heuristic 0 - A naive heuristic base one the position of 0
        :param self: The board to calculate the heuristic for
        :return: The value of the heuristic
        """
        return 0 if self.array[(self.height - 1, self.width - 1)] == 0 else 1

    @staticmethod
    def h1(self: Board) -> int:
        """
        Heuristic 1 - Hamming Distance
        :param self: The board to calculate the heuristic for
        :return: The value of the heuristic
        """

        # Find the lowest heuristic over all goal states, return 0 if a goal state
        return min(map(self._heuristic_hamming, self.goals)) if not self.is_goal else 0

    @staticmethod
    def h2(self: Board) -> int:
        """
        Heuristic 2 - Wrapped Manhattan Distance
        We are using regular Manhattan Distance, and accounting for wraps.
        If a wrap is the shorter path, one is also added to account for the more expensive move.
        :param self: The board to calculate the heuristic for
        :return: The value of the heuristic
        """

        # Find the lowest heuristic over all goal states, return 0 if a goal state
        return min(map(self._heuristic_manhattan, self.goals)) if not self.is_goal else 0

    def _heuristic_hamming(self, goal: Array) -> int:
        """
        Hamming Distance heuristic
        :param goal: Goal state the calculate the heuristic from
        :return: The Hamming Distance from the given goal state
        """

        # Running total
        total = 0
        for index in np.ndindex(goal.shape):
            i = goal[index]
            if i == 0:
                # Skip zero since it's out "empty" position
                continue

            # If the spots do not match, add one
            if i != self.array[index]:
                total += 1
        return total

    def _heuristic_manhattan(self, goal: Array) -> int:
        """
        Manhattan Distance heuristic
        :param goal: Goal state the calculate the heuristic from
        :return: The Manhattan Distance from the given goal state
        """

        # Running total
        total = 0
        for index in np.ndindex(goal.shape):
            i = goal[index]
            if i == 0:
                # Skip zero since it's out "empty" position
                continue

            g = Point(index)
            t = self._find_point(self.array, i)
            x, y = Point.manhattan_distance(g, t)
            # Take care of wrapping
            wraps = 0
            if x > self.height // 2:
                x = self.height - x
                wraps = 1
            if y > self.width // 2:
                y = self.width - y
                wraps = 1
            # Make sure we don't add two wrapping penalties
            total += x + y + wraps
        return total
    # endregion

    # region Static methods
    @staticmethod
    def _find_point(array: Array, value: int) -> Point:
        return Point(*np.asarray(np.where(array == value)).T[0])

    @staticmethod
    def from_list(data: List[int], shape: Tuple[int, int], dtype: Optional[object] = np.int16) -> Board:
        """
        Creates a new board from a list and a specified size
        :param data:  List to create the board from
        :param shape: Shape of the board (height, width)
        :param dtype: Type used within the Numpy arrays
        :return: The created board
        """

        # Create the board array
        array: Array = np.array(data, dtype=dtype).reshape(shape)
        # Find the location of the zero
        zero = Board._find_point(array, 0)
        # Create both solution boards
        g1: Array = np.roll(np.arange(array.size, dtype=dtype), -1).reshape(shape)
        g2: Array = g1.T.reshape(shape, order='F')
        return Board(array, zero, (g1, g2))
    # endregion
