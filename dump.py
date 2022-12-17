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

_GB = namedtuple("GameBoard", "tup ptm")
def move(loc, direction):
    """
    Produces the location attained by going in the given direction
    from the given location.
    loc will be a (<row>, <column>) double, and direction will be
    U, L, D, or R.
    """
    r0, c0 = loc
    if direction == U:
        return (r0 - 1, c0)
    elif direction == D:
        return (r0 + 1, c0)
    elif direction == L:
        return (r0, c0 - 1)
    elif direction == R:
        return (r0, c0 + 1)
    else:
        raise ValueError("The input direction is not valid.")

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
def transition(node, action):

    # prepare parts of result state
    ptm = node.ptm
    startBoard = toList(toList(node.tup,height,width))
    board = [[elt for elt in row] for row in startBoard]
    rows, cols = len(board), len(board[0])
    player_locs = [loc for loc in node.player_locs]
    ww_locs = [loc for loc in node.ww_locs]
    prev_cell_type = [cell for cell in node.prev_cell_type]
    next_ptm = get_next_player(ptm)

    r0, c0 = node.player_locs[ptm]

    # get target location after moving
    r1, c1 = GoTProblem.move((r0, c0), action)

    # End of Game: Hit wall or own trail, the current player loses
    if board[r1][c1] == CellType.WALL:
        player_locs[ptm] = None         # end of game
        self.mprint("Player " + self.get_player_head(ptm) + " hit the wall and crashed!")
    elif self._check_hit_own_trail(board, [r1, c1], ptm):
        player_locs[ptm] = None         # end of game
        self.mprint("Player " + self.get_player_head(ptm) + " hit its own tail and crashed!")
    # End of Game: Attack the other player, the current player wins
    elif self._check_hit_other_trail(board, [r1, c1], ptm):
        player_locs[next_ptm] = None
        self.mprint("Player " + self.get_player_head(ptm) +
              " killed Player" + self.get_player_head(next_ptm) + "'s tail!")

    # Enters SPACE area
    elif board[r1][c1] == CellType.SPACE:
        if prev_cell_type[ptm] == self.players[ptm]['PERM']:
            board[r0][c0] = self.players[ptm]['PERM']
        else:
            board[r0][c0] = self.players[ptm]['TEMP']

        GoTProblem._move_player_and_update(board, ptm, player_locs, r1, c1)
        prev_cell_type[ptm] = self.players[ptm]['TEMP']

    # Enters opponent's perm area
    elif board[r1][c1] == self.players[next_ptm]['PERM']:
        # board[r0][c0] = self.players[ptm]['TEMP'] ######OLD
        if prev_cell_type[ptm] == self.players[ptm]['PERM']:
            board[r0][c0] = self.players[ptm]['PERM']
        else:
            board[r0][c0] = self.players[ptm]['TEMP']


        GoTProblem._move_player_and_update(board, ptm, player_locs, r1, c1)
        prev_cell_type[ptm] = self.players[ptm]['TEMP']

    # Enters perm area of current player
    else:
        if prev_cell_type[ptm] == self.players[ptm]['TEMP']:
        # The current player returns to PERM from TEMP, trigger claiming and filling
        # The temporary location is in the same fully connected component
        # The player has to return to the base; only reaching the last step of trail is not 'close'
            board[r0][c0] = self.players[ptm]['TEMP']
            enclose_space = self._detect_space_inside(board, ptm)
            capture_ww_list, capture_other_player_bool  = GoTProblem._capture_others(
                board, enclose_space, ptm, player_locs)

            if capture_other_player_bool:
                board[player_locs[next_ptm][0]][player_locs[next_ptm][1]] = CellType.DEATH
                player_locs[next_ptm] = None
                self.mprint("Player " + self.get_player_head(ptm) + " captured and killed Player" +
                      self.get_player_head(next_ptm) + "!")
                return GoTState(board, player_locs, next_ptm, prev_cell_type, ww_locs)

            elif capture_ww_list:
                self.mprint("ww(s) captured at location(s):" + str(*capture_ww_list))
                for itm_loc in capture_ww_list:
                    assert len(ww_locs)
                    if not GoTProblem._is_same_loc(ww_locs[-1], itm_loc):
                        this_pos = ww_locs.index(itm_loc)
                        ww_locs[this_pos] = ww_locs[-1].copy()
                    ww_locs = ww_locs[:-1]

            space_to_fill = [[i, j] for j in range(cols) for i in range(rows) \
                if enclose_space[i][j] or board[i][j] == self.players[ptm]["TEMP"]]

            self.fill_board(board, space_to_fill, ptm)
            prev_cell_type[ptm] = self.players[ptm]['PERM']

        else:
            board[r0][c0] = self.players[ptm]['PERM']
            prev_cell_type[ptm] = self.players[ptm]['PERM']

        GoTProblem._move_player_and_update(board, ptm, player_locs, r1, c1)
        # only calculate the space once one claims some new spaces (otherwise won't increase)
        player_spaces = GoTProblem._count_space_players(board, prev_cell_type, self.players)
        space_ptm = player_spaces[ptm]
        if space_ptm >= self.half_cells:
            player_locs[next_ptm] = None
            self.mprint("Player " + self.get_player_head(ptm) + " won by claiming over half space!")

    return GoTState(board, player_locs, next_ptm, prev_cell_type, ww_locs)




class GameBoard(_GB, Node):
    def find_children(board, asp):
        if board.terminal:  # If the game is finished then no moves can be made
            return set()

    def make_move(board, action, asp):
        pass

def main():
    a = GameBoard(("0","1","0"),0,None, False, 13,13, [5,5],[[1,3],[5,7]],"1")
    b = GameBoard(("0","1","0"),0,None, False, 13,13, [5,5],[[1,3],[5,7]],"0")
    if a==b:
        print("yay")

if __name__ == "__main__":
    main()
