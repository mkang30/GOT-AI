from collections import namedtuple
from random import choice
from trainer import MCTS, Node
from functools import reduce
from operator import concat
from adversarialsearchproblem import AdversarialSearchProblem, GameState
from boardprinter import BoardPrinter
from GoT_types import CellType
import random
import numpy as np

_GB = namedtuple("GameBoard", "tup ptm winner terminal height width player_locs ww_locs prev_cell_ty")

def get_next_player(ptm):
    return 1 - ptm

def toTuple(array):
    flatlist = reduce(concat,array)
    return tuple(flatlist)

def toList(tup,rows,cols):
    newlist = [[0 for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            newlist[i][j] = tup[cols*i+j]
    return newlist


class GameBoard(_GB, Node):
    def find_children(board):
        if board.terminal:  # If the game is finished then no moves can be made
            return set()

    def make_move(board, action):
        pass

def main():
    a = GameBoard(("0","1","0"),0,None, False, 13,13, [5,5],[[1,3],[5,7]],"1")
    b = GameBoard(("0","1","0"),0,None, False, 13,13, [5,5],[[1,3],[5,7]],"1")
    if a==b:
        print(list(a))

if __name__ == "__main__":
    main()
