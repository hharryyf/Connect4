import numpy as np
import copy 
from operator import itemgetter
from connect4_game import Board


def random_rollout(board: Board):
    action_probability = np.random.rand(len(board.available()))
    return zip(board.available(), action_probability)


def policy_value_function(board: Board):
    action_probability = np.ones(len(board.available())) / len(board.available())
    return zip(board.available(), action_probability), 0

class PureMCTSNode(object):
    def __init__(self, parent, prior):
        self.parent = parent
        self.children = {}
        self.N, self.Q, self.U = 0, 0, 0
        self.P = prior
    
    def expansion(self, action_prior):
        for action, prob in action_prior:
            if action not in self.children:
                self.children[action] = PureMCTSNode(self, prob)


    def selection(self, c_puct):
        return max(self.children.items(), key=lambda action: action[1].get_value(c_puct))

    # we use the c_puct algorithm
    def get_value(self, c_puct):
        self.U = c_puct * self.P * np.sqrt(self.parent.N) / (1 + self.N)
        return self.Q + self.U

    # val is the new value of the state
    def update(self, val):
        self.N += 1
        # (Q_previous * N + val) / (N + 1) - Q_previous
        # delta = -Q_previous / (N + 1) + val / (N + 1)
        # check page 26 of Mastering the Game of Go without Human Knowledge about why this is true
        self.Q += (val - self.Q) / self.N

    def update_recursive(self, val):
        if self.parent != None:
            self.parent.update_recursive(-val)
        self.update(val)

    def leaf(self):
        return len(self.children) == 0

class PureMCTS(object):
    def __init__(self, policy_value_function, c_puct=5, n_playout=10000):
        self.root = PureMCTSNode(None, 1.0)
        self.policy_value_function = policy_value_function
        self.c_puct = c_puct
        self.max_playout = n_playout

    def playout(self, board: Board):
        curr = self.root
        while(curr != None):
            if curr.leaf():
                break
            action, node = curr.selection(self.c_puct)
            board.do_move(action)
            curr = node

        action_probability, _ = self.policy_value_function(board)
        game_over, _ = board.has_winner()
        if not game_over:
            curr.expansion(action_probability)
        
        leaf_value = self.evaluate_rollout(board)
        curr.update_recursive(-leaf_value)

    def evaluate_rollout(self, board: Board, limit=1000):
        player = board.current_player
        for i in range(limit):
            end, winner = board.has_winner()
            if end:
                break
            action_probability = random_rollout(board)
            best_action = max(action_probability, key=itemgetter(1))[0]
            board.do_move(best_action)
        
        if winner == 0:  
            return 0
        else:
            # note that in the connect4 situation, the return should always be -1
            # but in games like Go, the return can be 1, because the final move might not be
            # made by the winner
            return 1 if winner == player else -1

    def get_move(self, board: Board):
        for n in range(self.max_playout):
            board_copy = copy.deepcopy(board)
            self.playout(board_copy)
        return max(self.root.children.items(), key=lambda act_node: act_node[1].N)[0]

    def update_with_move(self, last_move):
        if last_move in self.root.children:
            self.root = self.root.children[last_move]
            self.root.parent = None
        else:
            self.root = PureMCTSNode(None, 1.0) 

class PureMCTSPlayer(object):
    def __init__(self, c_puct=5, n_playout=2000):
        self.mcts = PureMCTS(policy_value_function, c_puct, n_playout)

    def set_player(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, board: Board):
        if not board.game_end():
            move = self.mcts.get_move(board)
            self.mcts.update_with_move(move)
            return move, 0
        else:
            AssertionError("Cannot move when board is at terminal state")
