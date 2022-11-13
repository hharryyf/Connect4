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
        # get the children with the maximum "weighted" U + Q based on the c_puct value
        return max(self.children.items(), key=lambda action: action[1].get_value(c_puct))

    # we use the c_puct algorithm
    # note that it is quite obvious that the visit count of the parent node is the sum of the visit value of all the children
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
            # the negative sign because the reward is based on the current player's point of view
            # if we move up by 1 level, we should negate it
            self.parent.update_recursive(-val)
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
            # selection
            action, node = curr.selection(self.c_puct)
            board.do_move(action)
            curr = node

        # determine the value of the current state
        # very important: action probability's length is equal to the number of cols that are not filled
        action_probability, leaf_value = self.policy_value_function(board)
        game_over, winner = board.has_winner()
        # if the game does not end
        if not game_over:
            # expand the leaf with the new prior probability
            curr.expansion(action_probability)
        else:
            if winner == 0:
                # leaf value is 0 when the game is draw
                leaf_value = 0.0
            else:
                # otherwise, the leaf value is 1/-1
                # for connect4, this line always return -1.0
                leaf_value = 1.0 if winner == board.current_player else -1.0
        
        # check the 2012 paper Monte-Carlo Graph Search for AlphaZero https://arxiv.org/pdf/2012.11045.pdf
        # on why this negative sign is necessary
        curr.update_recursive(-leaf_value)

    # we want to get the move probability
    def get_move_probability(self, board: Board, temp=1e-3):
        for n in range(self.max_playout):
            # each time we do a playout
            board_copy = copy.deepcopy(board)
            self.playout(board_copy)
        
        # get the probability of the root node, note that this root node is frequently updated
        action_visits = [(action, c.N) for action, c in self.root.children.items()]
        # get the actions and the visit count
        action, visits = zip(*action_visits)
        # action probability encode with the softmax, only contain valid actions
        action_probability = softmax(1.0/temp * np.log(np.array(visits) + 1e-10))

        # return the simulated actions and their probability
        return action, action_probability

    def update_with_move(self, last_move):
        # update the move, when the move has been simulated, replace the root with the subtree's root
        if last_move in self.root.children:
            self.root = self.root.children[last_move]
            self.root.parent = None
        else:
            # otherwise, the root is a new one
            self.root = AlphaMCTSNode(None, 1.0) 

class MCTSDQNPlayer(object):
    def __init__(self, policy_value_function, c_puct=5, n_playout=2000, self_play_mode=False):
        # init
        self.mcts = MCTSZero(policy_value_function, c_puct, n_playout)
        # see if the current mode is self play
        self.self_play_mode = self_play_mode

    def set_player(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, board: Board, temp=1e-3):
        move_probability = np.zeros(board.ncol)
        if not board.game_end():
            action, action_probability = self.mcts.get_move_probability(board, temp)
            # set the action with action probability
            # note that len(action) might not equal to board.ncol
            move_probability[list(action)] = action_probability
            if self.self_play_mode:
                # add some dirichlet noise, see the AlphaGo paper
                # force exploration
                move = np.random.choice(action,p=0.75*action_probability + 0.25*np.random.dirichlet(0.3*np.ones(len(action_probability))))
                self.mcts.update_with_move(move)
            else:
                # almost equivalent to get the move with the highest probability
                # print(action, action_probability)
                move = np.random.choice(action, p=action_probability)
                self.mcts.update_with_move(-1)
            # [3.97563671e-263, 6.82409694e-078, 5.53077680e-254, 4.98458246e-245, 5.53077680e-254, 1.00000000e+000, 4.98458246e-245]
            return move, move_probability
        else:
            AssertionError("Cannot move when board is at terminal state")

    def __str__(self):
        return "MCTSDQNPlayer"

    

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
                move = player1.get_action(self.board)
                print("player", player1, move[0])
            else:
                move = player2.get_action(self.board)
                print("player", player2, move[0])
            self.board.do_move(move[0])
            end, winner = self.board.has_winner()
            if end:
                if winner == 0:
                    print("Draw!")
                elif i % 2 == 0:
                    print("Winner is ", player1)
                else:
                    print("Winner is ", player2)
                return winner
            i += 1