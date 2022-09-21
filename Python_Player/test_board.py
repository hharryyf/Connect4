from random import randint
import numpy as np
from connect4_game import Board
from colorama import Fore
from colorama import Style


def test_player_update_simple():
    board = Board(6, 7)
    assert (board.current_player == 1)
    # take a move at position 4
    board.do_move(4)
    assert (len(board.movedict) == 1)
    assert (board.current_player == -1)
    assert (board.lastmove == 4)
    assert (board.col[4] == 1)
    # take a move at position 2
    board.do_move(2)
    assert (len(board.movedict) == 2)
    assert (board.current_player == 1)
    assert (board.lastmove == 2)
    assert (board.col[4] == 1)
    assert (board.col[2] == 1)
    assert (board.movedict.get((0, 4)) == 1)
    assert (board.movedict.get((0, 2)) == -1)
    
def check_can_move():
    board = Board(6, 7)
    assert (board.can_move(4))
    board.do_move(4)
    assert (board.can_move(4))
    board.do_move(4)
    assert (board.can_move(4))
    board.do_move(4)
    assert (board.can_move(4))
    board.do_move(4)
    assert (board.can_move(4))
    board.do_move(4)
    assert (board.can_move(4))
    board.do_move(4)
    assert (not board.can_move(4))

def test_board_state_pre_maximizer():
    board = Board(6, 7)
    board.do_move(4)
    board.do_move(1)
    brd_rep = board.get_board_state()
    target = np.zeros((4, 6, 7))
    target[0, 0, 4] = 1
    target[1, 0, 1] = 1
    target[2, 0, 1] = 1
    target[3] = np.ones((6, 7))
    assert ((target == brd_rep).all())
    board.display()
    
def test_board_state_pre_minimizer():
    board = Board(6, 7)
    board.do_move(4)
    board.do_move(1)
    board.do_move(4)
    brd_rep = board.get_board_state()
    target = np.zeros((4, 6, 7))
    target[1, 0, 4] = 1
    target[0, 0, 1] = 1
    target[1, 1, 4] = 1
    target[2, 1, 4] = 1
    assert ((target == brd_rep).all())

def test_player_update_end_game_col():
    def check_end(b):
        def ok(x, y):
            return x >= 0 and x <= 5 and y >= 0 and y <= 6
        
        found = False
        for i in range(0, 6):
            for j in range(0, 7):
                if b[i][j] == 0:
                    found = True                
                if ok(i, j) and ok(i, j+1) and ok(i, j+2) and ok(i, j+3):
                    if b[i][j] + b[i][j+1] + b[i][j+2] + b[i][j+3] == 4:
                        return True, 1
                    
                    if b[i][j] + b[i][j+1] + b[i][j+2] + b[i][j+3] == -4:
                        return True, -1
                if ok(i, j) and ok(i+1, j) and ok(i+2, j) and ok(i+3, j):
                    if b[i][j] + b[i+1][j] + b[i+2][j] + b[i+3][j] == 4:
                        return True, 1
                    
                    if b[i][j] + b[i+1][j] + b[i+2][j] + b[i+3][j] == -4:
                        return True, -1
                
                if ok(i, j) and ok(i+1, j+1) and ok(i+2, j+2) and ok(i+3, j+3):
                    if b[i][j] + b[i+1][j+1] + b[i+2][j+2] + b[i+3][j+3] == 4:
                        return True, 1
                    
                    if b[i][j] + b[i+1][j+1] + b[i+2][j+2] + b[i+3][j+3] == -4:
                        return True, -1
                
                if ok(i, j) and ok(i+1, j-1) and ok(i+2, j-2) and ok(i+3, j-3):
                    if b[i][j] + b[i+1][j-1] + b[i+2][j-2] + b[i+3][j-3] == 4:
                        return True, 1
                    
                    if b[i][j] + b[i+1][j-1] + b[i+2][j-2] + b[i+3][j-3] == -4:
                        return True, -1

        if found:
            return False, 2
        return True, 0

    # we run 10000 random games
    correct = True
    for i in range(1, 10001):
        board = Board(6, 7)
        cnt = 0
        while True:
            rnd = randint(0, 6)
            if board.can_move(rnd):
                board.do_move(rnd)    
                cnt += 1
            my_state = board.has_winner()
            state = check_end(board.piece)

            if my_state[0] != state[0] or my_state[1] != state[1]:
                print("--------------- Test Case", i, f"(moves={cnt})---------------")
                print(f"                {Fore.RED}FAIL{Style.RESET_ALL}")
                print("---------- Test Details -------------")
                print("move dictionary")
                print(board.movedict)
                print(board.piece)
                print(board.col)
                print("printed status", my_state)
                board.display()

            correct = correct and my_state[0] == state[0] and my_state[1] == state[1]
    
            if state[0] or not correct:
                break
        
        if not correct:
            break
        print("--------------- Test Case", i, f"(moves={cnt})---------------")
        print(f"                {Fore.BLUE}OK{Style.RESET_ALL}")
    assert (correct == True)