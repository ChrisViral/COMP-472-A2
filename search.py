import time
from queue import PriorityQueue
from board import Board, State
from typing import List, Tuple, Dict, NamedTuple, Optional


class SearchResult(NamedTuple):
    cost: int
    path: List[State]


class SearchTrace(NamedTuple):
    f: int
    g: int
    h: int
    state: str


def search(board: Board, timeout: float = 60.0) -> Tuple[Optional[float], Optional[Tuple[SearchResult, List[SearchTrace]]]]:
    """
    Search through the board using a PriorityQueue as an open list
    :param board:   Starting state board
    :param timeout: If the search times out after 60 seconds
    :return: A tuple containing
    """

    start = time.time()
    opened: PriorityQueue[Board] = PriorityQueue()
    closed: Dict[Board, Board] = {}
    path: List[SearchTrace] = []

    opened.put(board)

    while not opened.empty():
        if timeout and time.time() - start > timeout:
            return time.time() - start, None

        board = opened.get()
        path.append(SearchTrace(board.f, board.g, board.h, str(board)))

        # When goal found
        if board.is_goal:
            return time.time() - start, (get_answer(board), path)

        # Skip if already visited at lower cost
        if board in closed and closed[board].g < board.g:
            # This is to skip duplicates in the open list since we cannot remove from it
            continue

        # Put into closed list
        closed[board] = board

        for _, _, child in board.generate_moves():
            if child in closed:
                if closed[child].g < child.g:
                    # If in closed list at lower cost, ignore
                    continue
                else:
                    # Otherwise remove from closed list
                    del closed[child]

            # Add to open list
            opened.put(child)

    return None, None


def get_answer(board: Board) -> SearchResult:
    cost = board.g
    path: List[State] = [State(0, 0, board)]
    while board.parent is not None:
        path.append(board.parent)
        board = board.parent.board

    return SearchResult(cost, path)
