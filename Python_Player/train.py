import random
import numpy as np 
from connect4_game import Board
from neural_mcts import MCTSDQNPlayer
from neural_mcts import GamePipeLine
from pure_mcts import PureMCTSPlayer
from valuenet import ValueNet
from collections import defaultdict, deque

class TrainingPipeLine(object):
    def __init__(self, gpu=False, init_model=None):
        self.nrow, self.ncol = 6, 7
        self.board = Board(self.nrow, self.ncol)
        self.game = GamePipeLine(self.board)
        self.lr = 2e-3
        self.playout = 400
        self.c_puct = 5
        self.buffer_size = 10000
        self.batch_size = 512  # mini-batch size for training
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.epochs = 5  # num of train_steps for each update
        self.kl_targ = 0.02
        self.check_freq = 50
        self.game_batch_num = 1500
        self.best_win_ratio = 0.0
        # num of simulations used for the pure mcts, which is used as
        # the opponent to evaluate the trained policy
        self.pure_mcts_playout_num = 1000
        if init_model:
            self.policy_value_net = ValueNet(gpu,init_model)
        else:
            # start training from a new policy-value net
            self.policy_value_net = ValueNet(gpu, None)
        self.mcts_player = MCTSDQNPlayer(self.policy_value_net.evaluate_position, 5, self.playout, True)

    def policy_evaluate(self, n_games=10):
        """
        Evaluate the trained policy by playing against the pure MCTS player
        Note: this is only for monitoring the progress of training
        """
        dqn_player = MCTSDQNPlayer(self.policy_value_net.evaluate_position,
                                         c_puct=self.c_puct,
                                         n_playout=self.playout)
        pure_mcts_player = PureMCTSPlayer(c_puct=5,
                                     n_playout=self.pure_mcts_playout_num)
        
        win, draw, loss = 0, 0, 0
        for i in range(n_games):
            if i % 2 == 0:
                winner = self.game.play_game(dqn_player, pure_mcts_player)
                if winner == 1:
                    win += 1
                elif winner == 0:
                    draw += 1
                else:
                    loss += 1
            else:
                winner = self.game.play_game(pure_mcts_player, dqn_player)
                if winner == -1:
                    win += 1
                elif winner == 0:
                    draw += 1
                else:
                    loss += 1

        win_rate = 1.0 * (win + 0.5 * draw) / n_games
        print("Pure MCTS #playouts:{}, win: {}, lose: {}, draw:{}".format(self.pure_mcts_playout_num,win, loss, draw))
        
        return win_rate

    def run(self):
        pass 


if __name__ == '__main__':
    pipeline = TrainingPipeLine(False, None)
    pipeline.run()