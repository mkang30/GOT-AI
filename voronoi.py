import math
import numpy as np
from GoT_problem import *
from GoT_types import CellType

def voronoi(state,board, player):
    locs = state.player_locs
    board = state.board
    enemy = abs(player-1)
    player_area = 0
    enemy_area = 0
    for i in range(len(board)):
        for j in range(len(board[0])):
            player_dist = math.sqrt(math.pow(locs[player][0]-i,2),math.pow(locs[player][1]-j,2))
            enemy_dist = math.sqrt(math.pow(locs[enemy][0]-i,2),math.pow(locs[enemy][1]-j,2))
            if player_dist>enemy_dist:
                player_area += 1
            elif enemy_dist> player_dist:
                enemy_area += 1
    if player_area+enemy_area == 0:
        return 0
    return 2*player_area/(player_area+enemy_area)-1
