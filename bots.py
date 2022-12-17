#!/usr/bin/python
import math
import numpy as np
from GoT_problem import *
from GoT_types import CellType
import random

from sys import platform
if "linux" in platform:
    import getch
else:
    import msvcrt


# Throughout this file, ASP means adversarial search problem.
class StudentBot:
    def __init__(self):
        self.ptm = None
        self.perm_cell = None
        self.temp_cell = None
        self.prev_cell_type = None
        self.last_move = None
        self.opp_perm_cell = None
        self.opp_temp_cell = None

    def cleanup(self):
        """
        Input: None
        Output: None
        This function will be called in between
        games during grading. You can use it
        to reset any variables your bot uses during the game
        (for example, you could use this function to reset a
        turns_elapsed counter to zero). If you don't need it,
        feel free to leave it as "pass"
        """
        self.prev_cell_type = None
        self.last_move = None
        self.ptm = None
        self.perm_cell = None
        self.temp_cell = None
        self.opp_perm_cell = None
        self.opp_temp_cell = None

    def miniboard_area_safe(self, mini_board):
        """
        Takes in a mini-board and returns true if the mini board does not contain opp or WW
        """
        opp = (self.ptm + 1) % 2
        res = np.count_nonzero(mini_board == str(opp + 1)) == 0 and \
               np.count_nonzero(mini_board == CellType.WHITE_WALKER) == 0
        return res

    def num_open_trails(self, board):
        """
        Counts the number of open trails of ptm
        """
        return np.count_nonzero(board == self.temp_cell)

    def bfs(self, board, loc):
        """
        Does BFS starting loc and returns the manhattan distance to the closest ptm perm space
        """
        queue = [(loc, 0)]
        while queue:
            c, depth = queue.pop(0)
            possibilities = list(GoTProblem.get_safe_actions(board, c, self.ptm))
            if depth == 3:
                return math.inf
            if board[c] == self.perm_cell:
                return depth
            for move in possibilities:
                next_loc = GoTProblem.move(c, move)
                if board[next_loc] == self.perm_cell:
                    return depth + 1
                if board[next_loc] in {CellType.SPACE, CellType.TWO_PERM}:
                    queue.append((next_loc, depth + 1))

    def get_mini_board(self, board, loc, size):
        """
        Gets the size x size mini board from the loc
        """
        return board[(loc[0] - 1):(loc[0] - 1)+size, (loc[1] - 1):(loc[1] - 1)+size]

    def decide(self, asp):
        """
        Input: asp, a GoTProblem
        Output: A direction in {'U','D','L','R'}
        To get started, you can get the current
        state by calling asp.get_start_state()
        """
        state = asp.get_start_state()
        locs = state.player_locs
        board = np.array(state.board)
        ptm = state.ptm
        opp = (ptm + 1) % 2  # 0 if ptm == 1 and 1 if ptm == 0

        # both player locations
        loc = locs[ptm]
        opp_loc = locs[opp]

        if self.ptm is None:  # initialize ptm if not yet initialized
            self.ptm = ptm
            if ptm == 0:
                self.perm_cell = CellType.ONE_PERM
                self.temp_cell = CellType.ONE_TEMP
                self.opp_perm_cell = CellType.TWO_PERM
                self.opp_temp_cell = CellType.TWO_TEMP
            else:
                self.perm_cell = CellType.TWO_PERM
                self.temp_cell = CellType.TWO_TEMP
                self.opp_perm_cell = CellType.ONE_PERM
                self.opp_temp_cell = CellType.ONE_TEMP

        # possible next moves for ptm
        possibilities = list(GoTProblem.get_safe_actions(board, loc, ptm))
        if not possibilities:  # if ptm cannot make a move that avoids collision
            return "U"  # default move

        if self.prev_cell_type is None:  # meaning ptm is at their start position
            self.prev_cell_type = self.temp_cell  # temp_cell will be their next move, so save it as prev move for next run
            this_move = possibilities[0]  # could be U, D, L, R
            self.last_move = this_move  # save it for next run
            return this_move

        # areas both players captured
        # ptm_area = np.count_nonzero(board == self.perm_cell)
        # opp_area = np.count_nonzero(board == self.opp_perm_cell)

        # each element holds (move, next_loc, what)
        possible_outcomes = [(move, GoTProblem.move(loc, move), board[GoTProblem.move(loc, move)]) for move in
                             possibilities]

        # check if player needs to return to perm area (to leave no trail vulnerable)
        if self.prev_cell_type == self.temp_cell:  # i.e. has left a trail area (ptm is outside perm area)
            # if there's a clear winning move, take it
            clear_winning_outcomes = list(filter(lambda x: x[2] in {self.opp_temp_cell, str(opp + 1)}, possible_outcomes))
            if clear_winning_outcomes:
                return clear_winning_outcomes[0][0]  # returns the move

            mini_board = self.get_mini_board(board, [loc[0] - 3, loc[1] - 3],
                                             7)  # gets the 7x7 board around ptm's location

            # the 5x5 board around the ptm loc is safe, make trail
            if self.miniboard_area_safe(mini_board) and self.num_open_trails(mini_board) < 3:
                area_expanding_outcomes = list(filter(lambda x: x[2] in {self.opp_perm_cell, CellType.SPACE}, possible_outcomes))
                close_to_perm_area_outcomes = list(filter(lambda x: self.perm_cell in [board[GoTProblem.move(x[1], move)] for move in
                                                    list(GoTProblem.get_safe_actions(board, x[1], ptm))], area_expanding_outcomes))
                if close_to_perm_area_outcomes:
                    this_move = close_to_perm_area_outcomes[0][0]
                    next_loc = close_to_perm_area_outcomes[0][1]
                    self.prev_cell_type = self.perm_cell if board[next_loc] == self.perm_cell else self.temp_cell  # set to permCell as this_move takes you back to perm area
                    self.last_move = this_move
                    return self.last_move

            # else, do bfs to find the move that takes it closer to the nearest perm space
            this_move = None
            next_loc = None
            shortest_dist = math.inf
            for move, next_loc, _ in possible_outcomes:
                if board[next_loc] == self.perm_cell:
                    this_move = move
                    next_loc = next_loc
                    break
                dist = self.bfs(board, next_loc)
                if dist and dist < shortest_dist:
                    shortest_dist = dist
                    this_move = move
                    next_loc = next_loc

            self.prev_cell_type = self.perm_cell if board[next_loc] == self.perm_cell else self.temp_cell
            self.last_move = this_move
            return self.last_move

        # Beyond this implies ptm already in perm area
        valid_next_move = None

        # if there's a clear winning move, take it
        clear_winning_outcomes = list(filter(lambda x: x[2] in {self.opp_temp_cell, str(opp + 1)}, possible_outcomes))
        if clear_winning_outcomes:
            return clear_winning_outcomes[0][0]  # returns the move

        # if surrounding is safe, prioritize moves that are expanding ptm area (and get them closer to opp)
        area_expanding_outcomes = list(filter(lambda x: x[2] in {self.opp_perm_cell, CellType.SPACE}, possible_outcomes))
        if area_expanding_outcomes:
            shortest = math.inf
            for outcome in area_expanding_outcomes:
                mini_board = self.get_mini_board(board, outcome[1], 3)  # gets the 3x3 board around ptm location
                if not self.miniboard_area_safe(mini_board):  # if opponent or WW is in surrounding area
                    continue
                ptm_loc = outcome[1]
                # ww_locs = GoTProblem._ww_locs_from_board(board)
                dists = []
                # for ww_loc in ww_locs:
                #     dists.append(math.sqrt(math.pow(ptm_loc[0] - ww_loc[0], 2) + math.pow(ptm_loc[1] - ww_loc[1], 2)))
                dist_to_opp = math.sqrt(math.pow(ptm_loc[0] - opp_loc[0], 2) + math.pow(ptm_loc[1] - opp_loc[1], 2))
                dists.append(dist_to_opp)
                if min(dists) < shortest:
                    shortest = min(dists)
                    valid_next_move = outcome[0]

        # if no valid_next_move found yet and there are some moves that remain in the same area, then pick the move
        # that gets them closer to opp or opp area
        area_safe_outcomes = list(filter(lambda x: x[2] == self.perm_cell, possible_outcomes))
        if not valid_next_move and area_safe_outcomes:
            shortest = math.inf
            for outcome in area_safe_outcomes:
                ptm_loc = outcome[1]
                # ww_locs = GoTProblem._ww_locs_from_board(board)
                dists = []
                # for ww_loc in ww_locs:
                #     dists.append(
                #         math.sqrt(math.pow(ptm_loc[0] - ww_loc[0], 2) + math.pow(ptm_loc[1] - ww_loc[1], 2)))
                dist_to_opp = math.sqrt(math.pow(ptm_loc[0] - opp_loc[0], 2) + math.pow(ptm_loc[1] - opp_loc[1], 2))
                dists.append(dist_to_opp)
                if min(dists) < shortest:
                    shortest = min(dists)
                    valid_next_move = outcome[0]

        if not valid_next_move:  # if in case the var is None, pick a random move from possibilities
            random.shuffle(possibilities)
            valid_next_move = possibilities[0]

        # determine the prev cell type
        next_loc = GoTProblem.move(loc, valid_next_move)
        if board[next_loc[0], next_loc[1]] == self.perm_cell:  # i.e. the new loc ptm ends up in is in its own area
            self.prev_cell_type = self.perm_cell
        else:
            self.prev_cell_type = self.temp_cell

        self.last_move = valid_next_move

        return self.last_move


class RandBot:
    """Moves in a random (safe) direction"""

    def decide(self, asp):
        """
        Input: asp, a GoTProblem
        Output: A direction in {'U','D','L','R'}
        """
        state = asp.get_start_state()
        locs = state.player_locs
        board = state.board
        ptm = state.ptm
        loc = locs[ptm]
        possibilities = list(GoTProblem.get_safe_actions(board, loc, ptm))
        if possibilities:
            return random.choice(possibilities)
        return "U"

    def cleanup(self):
        pass


class ManualBot:
    """Bot which can be manually controlled using W, A, S, D"""

    def decide(self, asp: GoTProblem):
        # maps keyboard input to {U, D, L, R}
        dir_map = {'A': 'L', 'W': 'U',
                   'a': 'L', 'w': 'U',
                   'S': 'D', 'D': 'R',
                   's': 'D', 'd': 'R'}
        if "linux" in platform:
            # Command for mac/unix:
            direction = getch.getch()
        else:
            # Command for Windows:
            direction = msvcrt.getch().decode('ASCII')
        return dir_map[direction]

    def cleanup(self):
        pass


class AttackBot:
    """Aggressive bot which attacks opposing player when possible"""

    def __init__(self):
        self.prev_cell_type = None
        self.last_move = None
        self.ptm = None
        self.perm_cell = None
        self.temp_cell = None
        self.opp_temp_cell = None

    def cleanup(self):
        self.prev_cell_type = None
        self.last_move = None
        self.ptm = None
        self.perm_cell = None
        self.temp_cell = None
        self.opp_temp_cell = None

    def dist_from_opp(self, opp_loc, ptm_loc):
        dist = 0
        for i in range(len(opp_loc)):
            dist += abs(opp_loc[i] - ptm_loc[i])
        return dist

    def min_dist_to_temp(self, board, ptm_loc):
        locs = self.temp_barrier_locs_from_board(board)
        min_dist = math.inf
        for loc in locs:
            this_dist = self.dist_from_opp(loc, ptm_loc)
            if this_dist < min_dist:
                min_dist = this_dist
        return min_dist

    def temp_barrier_locs_from_board(self, board):
        if self.opp_temp_cell is None:
            return []
        loc_dict = {}
        num_temp = 0
        for r in range(len(board)):
            for c in range(len(board[r])):
                char = board[r][c]
                if char == self.opp_temp_cell:
                    loc_dict[num_temp] = (r, c)
                    num_temp += 1
        loc_list = []
        for index in range(num_temp):
            loc_list.append(loc_dict[index])
        return loc_list

    def decide(self, asp):
        """
        Input: asp, a GoTProblem
        Output: A direction in {'U','D','L','R'}
        """
        state = asp.get_start_state()
        locs = state.player_locs
        board = state.board
        ptm = state.ptm
        loc = locs[ptm]
        possibilities = list(GoTProblem.get_safe_actions(board, loc, ptm))
        opp_loc = locs[(ptm + 1) % 2]  # 0 if ptm == 1 and 1 if ptm == 0
        if self.ptm is None:  # initialize ptm if not yet initialized
            self.ptm = ptm
            self.perm_cell = CellType.TWO_PERM
            self.temp_cell = CellType.TWO_TEMP
            self.opp_temp_cell = CellType.ONE_TEMP
            if ptm == 0:
                self.perm_cell = CellType.ONE_PERM
                self.temp_cell = CellType.ONE_TEMP
                self.opp_temp_cell = CellType.TWO_TEMP

        if not possibilities:  # if ptm cannot make a move that avoids collision
            return "U"  # default move

        if self.prev_cell_type is None:  # meaning ptm is at their start position
            "Attack bot starting"
            self.prev_cell_type = self.temp_cell  # temp_cell will be their next move, so save it as prev move for next run
            this_move = possibilities[0]  # could be U, D, L, R
            self.last_move = this_move  # save it for next run
            return this_move

        # if player needs to return to perm area (to keep its trail safe)
        must_return_to_perm = False
        if self.prev_cell_type == self.temp_cell:
            # this means ptm left 1 trail, so the attack bot tries to close it by going back to perm area
            this_move = None
            if self.last_move == "U":
                this_move = "D"
            elif self.last_move == "D":
                this_move = "U"
            elif self.last_move == "R":
                this_move = "L"
            elif self.last_move == "L":
                this_move = "R"
            else:
                raise Exception

            self.prev_cell_type = self.perm_cell
            self.last_move = this_move
            must_return_to_perm = True

        # else, player is (potentially) leaving perm area - i.e. ptm is currently in its area
        # "potentially" because, the ptm may be still inside the area after next move decision
        min_dist = math.inf  # dist from opponent
        min_dist_to_temp = math.inf  # dist from opponent's trail
        go_for_temp = False  # decide if to go towards trail or opponent
        decision = possibilities[0]  # default decision
        min_next_loc = [None] * 2
        for move in possibilities:  # for every move see which is closer - dist from opp or dist from opp trail
            next_loc = GoTProblem.move(loc, move)  # new loc if this move possibility was going to be taken
            dist_from_opponent = self.dist_from_opp(next_loc, opp_loc)
            this_dist_to_temp = self.min_dist_to_temp(board, next_loc)
            if this_dist_to_temp == 0:  # basically implies ptm can hit opp's trail (and career) using this possible move
                return move  # this move will be obv prioritized over retreating back to ptm's prem area

            if not must_return_to_perm:
                # If we are close to temp barrier
                if this_dist_to_temp <= 5 or go_for_temp:
                    go_for_temp = True
                    if this_dist_to_temp < min_dist_to_temp:
                        min_dist_to_temp = this_dist_to_temp
                        decision = move
                        min_next_loc = next_loc

                elif dist_from_opponent < min_dist:
                    min_dist = dist_from_opponent
                    decision = move
                    min_next_loc = next_loc
                    min_dist_to_temp = this_dist_to_temp

                elif dist_from_opponent == min_dist:
                    if this_dist_to_temp < min_dist_to_temp:
                        min_dist_to_temp = this_dist_to_temp
                        decision = move
                        min_next_loc = next_loc

        if not must_return_to_perm:
            # find the min_next_loc move on board to move and set prev cell type
            if board[min_next_loc[0]][min_next_loc[1]] == self.perm_cell:
                self.prev_cell_type = self.perm_cell
            else:
                self.prev_cell_type = self.temp_cell
            self.last_move = decision
        return self.last_move


class SafeBot:
    """Bot that plays safe and takes area"""

    def __init__(self):
        self.prev_move = None
        self.to_empty = []
        self.algo_path = []
        self.path = []
        self.calc_empty = False
        self.order = {"U": ("L", "R"),
                      "D": ("L", "R"),
                      "L": ("U", "D"),
                      "R": ("U", "D")}

    def cleanup(self):
        self.prev_move = None
        self.to_empty = []
        self.algo_path = []
        self.path = []
        self.calc_empty = False
        self.order = {"U": ("L", "R"),
                      "D": ("L", "R"),
                      "L": ("U", "D"),
                      "R": ("U", "D")}

    def get_safe_neighbors_wall(self, board, loc):
        neighbors = [
            ((loc[0] + 1, loc[1]), D),
            ((loc[0] - 1, loc[1]), U),
            ((loc[0], loc[1] + 1), R),
            ((loc[0], loc[1] - 1), L),
        ]
        return list(filter(lambda m: board[m[0][0]][m[0][1]] != CellType.WALL, neighbors))

    def get_safe_neighbors_no_wall(self, board, loc, wall):
        neighbors = [
            ((loc[0] + 1, loc[1]), D),
            ((loc[0] - 1, loc[1]), U),
            ((loc[0], loc[1] + 1), R),
            ((loc[0], loc[1] - 1), L),
        ]
        return list(
            filter(lambda m: board[m[0][0]][m[0][1]] != CellType.WALL and board[m[0][0]][m[0][1]] != wall, neighbors))

    def decide(self, asp: GoTProblem):
        state = asp.get_start_state()
        if not self.path:
            if self.calc_empty:
                self.gen_path_to_empty(state)
                self.path += self.to_empty
                self.to_empty = []
                self.calc_empty = False
            else:
                self.gen_space_grab(state)
                self.path += self.algo_path
                self.algo_path = []
                self.calc_empty = True
        move = self.path.pop(0)
        self.prev_move = move
        return move

    def gen_space_grab(self, state: GoTState):
        board = state.board
        loc = state.player_locs[state.ptm]
        if state.ptm == 0:
            player_wall = CellType.ONE_PERM
        else:
            player_wall = CellType.TWO_PERM
        avail_actions = {U, D, L, R}
        prev = self.prev_move
        if prev:
            avail_actions.remove(prev)
        else:
            safe_actions = self.get_safe_neighbors_wall(board, loc)
            random.shuffle(safe_actions)
            loc, move = safe_actions[0]
            self.algo_path.append(move)
            avail_actions.remove(move)
            prev = move
        while avail_actions:
            safe_moves = self.get_safe_neighbors_no_wall(board, loc, player_wall)
            safe_moves_wall = self.get_safe_neighbors_wall(board, loc)
            if not safe_moves and not safe_moves_wall:
                self.algo_path.append(U)
                return
            random.shuffle(safe_moves)
            random.shuffle(safe_moves_wall)
            use_wall = True
            for loc, move in safe_moves:
                board_val = board[loc[0]][loc[1]]
                if move in self.order[prev] and move in avail_actions and board_val != player_wall:
                    self.algo_path.append(move)
                    avail_actions.remove(move)
                    prev = move
                    use_wall = False
                    break
            if use_wall:
                for loc, move in safe_moves_wall:
                    board_val = board[loc[0]][loc[1]]
                    if move in self.order[prev] and move in avail_actions:
                        self.algo_path.append(move)
                        avail_actions.remove(move)
                        prev = move
                        use_wall = False
                        break
        return

    def gen_path_to_empty(self, state: GoTState):
        board = state.board
        player_loc = state.player_locs[state.ptm]
        to_check = [(player_loc, None)]
        checked = {(player_loc, None): None}
        while to_check:
            loc, m = to_check.pop(0)
            neighbors = [
                ((loc[0] + 1, loc[1]), D),
                ((loc[0] - 1, loc[1]), U),
                ((loc[0], loc[1] + 1), R),
                ((loc[0], loc[1] - 1), L),
            ]
            random.shuffle(neighbors)
            for move in neighbors:
                x, y = move[0][0], move[0][1]
                board_val = board[x][y]
                if move not in checked and board_val != CellType.WALL:
                    checked[move] = (loc, m)
                    if board_val == ' ':
                        path = []
                        while move[1] is not None:
                            path.append(move[1])
                            move = checked[move]
                        self.to_empty += path
                        return
                    else:
                        to_check.append(move)
        self.to_empty += [U]
        return
