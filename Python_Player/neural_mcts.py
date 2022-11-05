import numpy as np
import copy 
from connect4_game import Board

def softmax(probability):
    probability = np.exp(probability - np.max(probability))
    return probability / np.sum(probability)

class AlphaMCTSNode(object):
    def __init__(self, parent, prior):
        self.parent = parent
        self.children = {}
        self.N, self.Q, self.U = 0, 0, 0
        self.P = prior
    
    def expansion(self, action_prior):
        for action, prob in action_prior:
            if action not in self.children:
                self.children[action] = AlphaMCTSNode(self, prob)


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
            self.parent.update_recirsive(-val)
        self.update(val)

    def leaf(self):
        return len(self.children) == 0

class MCTSZero(object):
    def __init__(self, policy_value_function, c_puct=5, n_playout=10000):
        self.root = AlphaMCTSNode(None, 1.0)
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

        action_probability, leaf_value = self.policy_value_function(board)
        game_over, winner = board.has_winner()
        if not game_over:
            curr.expansion(action_probability)
        else:
            if winner == 0:
                leaf_value = 0.0
            else:
                leaf_value = 1.0 if winner == board.current_player else -1.0
        
        curr.update_recursive(-leaf_value)

    def get_move_probability(self, board: Board, temp=1e-3):
        for n in range(self.max_playout):
            board_copy = copy.deepcopy(board)
            self.playout(board_copy)
        
        action_visits = [(action, c.N) for action, c in self.root.children.items()]
        action, visits = zip(*action_visits)
        action_probability = softmax(1.0/temp * np.log(np.array(visits) + 1e-10))

        return action, action_probability

    def update_with_move(self, last_move):
        if last_move in self.root.children:
            self.root = self.root.children[last_move]
            self.root.parent = None
        else:
            self.root = AlphaMCTSNode(None, 1.0) 

class MCTSDQNPlayer(object):
    def __init__(self, policy_value_function, c_puct=5, n_playout=2000, self_play_mode=False):
        self.mcts = MCTSZero(policy_value_function, c_puct, n_playout)
        self.self_play_mode = self_play_mode

    def set_player(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, board: Board, temp=1e-3):
        move_probability = np.zeros(board.ncol)
        if not board.game_end():
            action, action_probability = self.mcts.get_move_probability(board, temp)
            move_probability[list(action)] = action_probability
            if self.self_play_mode:
                move = np.random.choice(action,p=0.75*action_probability + 0.25*np.random.dirichlet(0.3*np.ones(len(action_probability))))
                self.mcts.update_with_move(move)
            else:
                move = np.random.choice(action, action_probability)
                self.mcts.update_with_move(move)
            return move, move_probability
        else:
            AssertionError("Cannot move when board is at terminal state")

    

class GamePipeLine(object):
    def __init__(self, board: Board):
        self.board = board
    
    def self_play(self, player: MCTSDQNPlayer, temp=1e-3):
        self.board.reset()
        board_state, mcts_probability, current_player = [], [], []
        while True:
            move, move_probs = player.get_action(self.board, temp=temp)
            # store the data
            board_state.append(self.board.get_board_state())
            mcts_probability.append(move_probs)
            current_player.append(self.board.current_player)
            # perform a move
            self.board.do_move(move)
            end, winner = self.board.has_winner()
            if end:
                # winner from the perspective of the current player of each state
                winners = np.zeros(len(current_player))
                if winner != 0:
                    winners[np.array(current_player) == winner] = 1.0
                    winners[np.array(current_player) != winner] = -1.0
                # reset MCTS root node
                player.reset_player()
                if winner != 0:
                    print("Game end. Winner is player:", winner)
                else:
                    print("Game end. Tie")
                return winner, zip(board_state, mcts_probability, winners)
    
    def play_game(self, player1, player2):
        self.board.reset()
        i = 0
        while True:
            move = -1
            if i % 2 == 0:
                move = player1.get_move(self.board)
            else:
                move = player2.get_move(self.board)
            self.board.do_move(move)
            end, winner = self.board.has_winner()
            if end:
                return winner
        