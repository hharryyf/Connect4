from connect4_game import Board

class Human(object):
    def get_action(self, board: Board):
        board.display()
        print("please insert action: ", end='')
        mv = int(input())
        return mv

    def __str__(self):
        return "Human Player"