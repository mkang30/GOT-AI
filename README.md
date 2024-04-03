# Game Of Thrones AI-solver
GOT is a 2-player board game with the conquer-the-land rules. Given the mxn matrix-board each player starts at opposing corners of the board. At each turn a player can move in any direction and leave the trace, which upon enclosing an area conquers the land within. Collision with another player or/and trace or/and white walker loses the game.
# Approach
Our solution to this board game was to implement an unsupervised learning on top of restricting policies implemented using Voronoi diagrams and manually weights configured. By combining the two techniques we were able to reach ~70% winrate on 8x8 board.
