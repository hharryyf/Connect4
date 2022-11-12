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
        self.lr_multiplier = 1.0
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

    def collect_selfplay_data(self, n_games=1):
        """collect self-play data for training"""
        for i in range(n_games):
            _, play_data = self.game.self_play(self.mcts_player)
            play_data = list(play_data)[:]
            print(play_data)
            self.episode_len = len(play_data)
            self.data_buffer.extend(play_data)

            # do data augmentation, horizontal flip extend with a new set of data
            extended = []
            for state, prob, winner in play_data:
                eqi_state = np.array([np.fliplr(s) for s in state])
                prob = np.fliplr(prob.reshape(1, len(prob))).flatten()
                extended.append((eqi_state, prob, winner))

            self.data_buffer.extend(extended)

            #print(play_data)
            #print("----------------------------")
            #print(extended)

    def policy_update(self):
        """update the policy-value net"""
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        old_probs, old_v = self.policy_value_net.evaluate_batches(state_batch)
        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(
                    state_batch,
                    mcts_probs_batch,
                    winner_batch, self.lr * self.lr_multiplier)
            new_probs, new_v = self.policy_value_net.evaluate_batches(state_batch)
            kl = np.mean(np.sum(old_probs * (
                    np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                    axis=1)
            )
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
        
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        print(("kl:{:.5f},"
               "loss:{},"
               "entropy:{}"
               ).format(kl,
                        loss,
                        entropy))
        return loss, entropy

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
        try:
            for i in range(self.game_batch_num):
                self.collect_selfplay_data(self.play_batch_size)
                print("batch i:{}, episode_len:{}".format(
                        i+1, self.episode_len))
                if len(self.data_buffer) > self.batch_size:
                    loss, entropy = self.policy_update()
                # check the performance of the current model,
                # and save the model params
                if (i+1) % self.check_freq == 0:
                    print("current self-play batch: {}".format(i+1))
                    win_ratio = self.policy_evaluate()
                    self.policy_value_net.save_model('./current_policy.model')
                    if win_ratio > self.best_win_ratio:
                        print("New best policy!!!!!!!!")
                        self.best_win_ratio = win_ratio
                        # update the best_policy
                        self.policy_value_net.save_model('./best_policy.model')
                        if (self.best_win_ratio == 1.0 and
                                self.pure_mcts_playout_num < 5000):
                            self.pure_mcts_playout_num += 1000
                            self.best_win_ratio = 0.0
        except KeyboardInterrupt:
            print('\n\rquit')


if __name__ == '__main__':
    pipeline = TrainingPipeLine(False, 'best_policy.model')
    pipeline.run()
