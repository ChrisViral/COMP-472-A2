import sys
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


def search(board: Board) -> Optional[Tuple[SearchResult, List[SearchTrace]]]:
    """
    Search through the board using a PriorityQueue as an open list
    :param board: Starting state board
    :return: A tuple containing
    """
    opened: PriorityQueue[Board] = PriorityQueue()
    closed: Dict[Board, Board] = {}
    path: List[SearchTrace] = []

    opened.put(board)

    while not opened.empty():
        board = opened.get()
        path.append(SearchTrace(board.f, board.g, board.h, str(board)))

        # When goal found
        if board.is_goal:
            return get_answer(board), path

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

    return None


def search_gbfs(board: Board) -> Optional[Tuple[SearchResult, List[SearchTrace]]]:

    path: List[SearchTrace] = []
    while not board.is_goal:
        path.append(SearchTrace(0, 0, board.h, str(board)))
        best = sys.maxsize
        chosen: Optional[Board] = None
        for _, _, child in board.generate_moves():
            if child.h < best:
                best = child.h
                chosen = child

        board = chosen

    return get_answer(board), path


def get_answer(board: Board) -> SearchResult:
    cost = board.g
    path: List[State] = []
    while board.parent is not None:
        path.append(board.parent)
        board = board.parent.board

    return SearchResult(cost, path)
