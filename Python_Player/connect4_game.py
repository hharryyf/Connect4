import numpy as np
from colorama import Fore
from colorama import Style
import os

os.system("color")

class Board(object):
    # initialize the board
    def __init__(self, nrow=6,ncol=7):
        self.nrow = nrow
        self.ncol = ncol
        # a move dictionary key = (i, j) coordinate, value = player 
        self.movedict = {}
        self.col = list(np.zeros(ncol, dtype=int))
        self.piece = list(np.zeros((nrow, ncol), dtype=int))
        self.lastmove = -1
        # player 1 always moves first
        self.current_player = 1 
        self.status = 2

    def reset(self):
        # a move dictionary key = (i, j) coordinate, value = player 
        self.movedict = {}
        self.col = list(np.zeros(self.ncol, dtype=int))
        self.piece = list(np.zeros((self.nrow, self.ncol), dtype=int))
        self.lastmove = -1
        # player 1 always moves first
        self.current_player = 1 
        self.status = 2

    # check if a player can move at column "move"
    def can_move(self, move):
        if move < self.ncol and move >= 0 and self.col[move] <= 5:
            return True
        return False

    def available(self):
        ret = []
        for i in range(0, self.ncol):
            if self.can_move(i):
                ret.append(i)
        return ret

    # see if there's a winner on the board
    def has_winner(self):
        if self.status != 2:
            return True, self.status
        return False, 2

    # see if the game ends, either someone wins the game or 
    # the game ends up in a draw
    def game_end(self):
        return self.status != 2

    # the current player takes a move at column "move"
    # precondition, the game status is not 'end'
    def do_move(self, move):
        assert (self.game_end() == False)
        move = int(move)
        def ok(tx, ty):
            return tx >= 0 and tx <= self.nrow - 1 and ty >= 0 and ty <= self.ncol - 1
        
        if self.can_move(move):
            self.lastmove = move
            self.movedict[(self.col[move], move)] = self.current_player
            self.piece[self.col[move]][move] = self.current_player
            # update the status of the board
            x, y = self.col[move], move

            for i in range(-3, 1):
                if ok(x+i,y) and ok(x+i+1,y) and ok(x+i+2,y) and ok(x+i+3,y):
                    v = self.piece[x+i][y] + self.piece[x+i+1][y] + self.piece[x+i+2][y] + self.piece[x+i+3][y]
                    if v == 4:
                        self.status = 1
                    elif v == -4:
                        self.status = -1

                if ok(x,y+i) and ok(x,y+i+1) and ok(x,y+i+2) and ok(x,y+i+3):
                    v = self.piece[x][y+i] + self.piece[x][y+i+1] + self.piece[x][y+i+2] + self.piece[x][y+i+3]
                    if v == 4:
                        self.status = 1
                    elif v == -4:
                        self.status = -1
                
                if ok(x+i,y+i) and ok(x+i+1,y+i+1) and ok(x+i+2,y+i+2) and ok(x+i+3,y+i+3):
                    v = self.piece[x+i][y+i] + self.piece[x+i+1][y+i+1] + self.piece[x+i+2][y+i+2] + self.piece[x+i+3][y+i+3]
                    if v == 4:
                        self.status = 1
                    elif v == -4:
                        self.status = -1

                if ok(x+i,y-i) and ok(x+i+1,y-i-1) and ok(x+i+2,y-i-2) and ok(x+i+3,y-i-3):
                    v = self.piece[x+i][y-i] + self.piece[x+i+1][y-i-1] + self.piece[x+i+2][y-i-2] + self.piece[x+i+3][y-i-3]
                    if v == 4:
                        self.status = 1
                    elif v == -4:
                        self.status = -1

            if self.status == 2 and len(self.movedict) == self.nrow * self.ncol:
                self.status = 0
            # increment the total number of moves in the column
            self.col[move] += 1
            # the current player changes
            self.current_player *= -1
            return True
        return False
    
    '''
        get the board state used for the policy-value network
        dimension[0] all the pieces played by the current player
        dimension[1] all the pieces played by the other player
        dimension[2] the player doing the current move, if it is the maximizer (i.e. if it is player 1), 
        then the final dimension is 1, otherwise 0
        precondition: this method can only be called before current_player actually takes the move!!!!
        return type: a numpy array
    '''
    def get_board_state(self):
        board_state = np.zeros((3, self.nrow, self.ncol))
        pieces = np.array(self.piece)
        if len(self.movedict) > 0:
            # positions of the current moves
            board_state[0] = 1.0 * (pieces == self.current_player)
            # positions of the moves by the other player
            board_state[1] = 1.0 * (pieces == -self.current_player)
            
        if len(self.movedict) % 2 == 0:
            board_state[2][:,:] = 1.0
        return board_state

    def display(self):
        for i in range(self.nrow - 1, -1, -1):
            for j in range(0, self.ncol):
                if self.piece[i][j] == 0:
                    print(".", end='')
                elif self.piece[i][j] == 1:
                    print(f"{Fore.RED}X{Style.RESET_ALL}", end='')
                else:
                    print(f"{Fore.BLUE}O{Style.RESET_ALL}", end='')
                if j != self.ncol - 1:
                    print("|", end='')
                else:
                    print("\n", end='')
            if i != 0:
                for j in range(0, self.ncol):
                    print("_", end='')
                    if j != self.ncol - 1:
                        print(" ", end='')
                    else:
                        print("\n", end='')
