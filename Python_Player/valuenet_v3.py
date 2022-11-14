import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from connect4_game import Board

# heavily referenced the implementation https://github.com/junxiaosong/AlphaZero_Gomoku/blob/master/policy_value_net_pytorch.py
# for learning purpose only

def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class Net(nn.Module):
    def __init__(self):
        super().__init__() 
        # common layer
        self.common_layer = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        self.transition = nn.Conv2d(512, 3, kernel_size=1)


        self.policy_linear_layer = nn.Sequential(
            nn.Linear(3 * 6 * 7, 3 * 6 * 7),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(3 * 6 * 7, 3 * 6 * 7),
            nn.Dropout(0.2),
            nn.ReLU(inplace=True),
            nn.Linear(3 * 6 * 7, 7),     
        )

        
        self.value_linear_layer = nn.Sequential(
            nn.Linear(3 * 6 * 7, 2 * 6 * 7),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(2 * 6 * 7, 2 * 6 * 7),
            nn.Dropout(0.2),
            nn.ReLU(inplace=True),
            nn.Linear(2 * 6 * 7, 1),     
        )

    # return the action probability (not necessary from 0 to 1) and the position score
    # action probability size = [1][7], position score size = [1][1]
    def forward(self, state_input):
        #x_in = state_input.clone()
        x = self.common_layer(state_input)
        x = self.transition(x)
        #x = x + x_in
        x = torch.flatten(x, start_dim=1)
        x_action = F.log_softmax(self.policy_linear_layer(x), dim=1)
        x_value = torch.tanh(self.value_linear_layer(x))
        # print(x_action, x_value)
        return x_action, x_value

class ValueNet(object):
    def __init__(self,gpu=False, trained_model=None):
        # if we have gpu
        self.gpu = gpu
        if gpu:
            # if so we use the cuda option
            self.value_net = Net().cuda()
        else:
            # otherwise, just the cpu
            self.value_net = Net()
        
        # we use the Adam optimizer
        self.optimizer = optim.Adam(self.value_net.parameters(), lr=0.002, weight_decay=0.0001)
        
        # if we have a trained model, we need to load the trained model from the file of the given path
        if trained_model:
            if gpu == False:
                params = torch.load(trained_model, map_location=torch.device('cpu'))
            else:
                params = torch.load(trained_model)
            self.value_net.load_state_dict(params)
    
    # evaluate a single position
    def evaluate_position(self, board: Board):
        # get all the available positions in a 1-d list
        valid_position = board.available()
        current_state = np.ascontiguousarray(board.get_board_state().reshape(
                -1, 3, 6, 7))
        if self.gpu:
            # the log probability of positions and the position reward
            log_probability, position_score = self.value_net(Variable(torch.from_numpy(current_state).cuda().float()))
            # get the exponential of these log probability
            move_probability = np.exp(log_probability.data.cpu().numpy().flatten())
        else:
            # same as the gpu one, just don't push it to cpu anymore
            log_probability, position_score = self.value_net(Variable(torch.from_numpy(current_state).float()))
            move_probability = np.exp(log_probability.data.numpy().flatten())

        # get the move probability, a zip of valid positions and the move probability of these positions
        move_probability = zip(valid_position, move_probability[valid_position])
        # the position score, a floating point number
        position_score = position_score.data[0][0]
        return move_probability, position_score

    # evaluate a batch
    def evaluate_batches(self, batch):
        if self.gpu:
            batch = Variable(torch.FloatTensor(batch).cuda())
            log_probability_batch, position_score_batch = self.value_net(batch)
            move_probability = np.exp(log_probability_batch.data.cpu().numpy())
            return move_probability, position_score_batch.data.cpu().numpy()
        else:
            batch = Variable(torch.FloatTensor(batch))
            log_probability_batch, position_score_batch = self.value_net(batch)
            move_probability = np.exp(log_probability_batch.data.numpy())
            return move_probability, position_score_batch.data.numpy()

    # a single training step, given a batch of state, mcts probability
    def train_step(self, batch, mcts_probability, winner, lr=0.002):
        
        if self.gpu:
            batch = Variable(torch.FloatTensor(batch).cuda())
            mcts_probability = Variable(torch.FloatTensor(mcts_probability).cuda())
            winner = Variable(torch.FloatTensor(winner).cuda())
        else:
            batch = Variable(torch.FloatTensor(batch))
            mcts_probability = Variable(torch.FloatTensor(mcts_probability))
            winner = Variable(torch.FloatTensor(winner))

        self.optimizer.zero_grad()
                
        set_learning_rate(self.optimizer, lr)

        # feed forward the state batch with the policy value net
        log_probability_batch, position_score_batch = self.value_net(batch)
        # define the loss = (z - v)^2 - pi^T * log(p) + c||theta||^2
        # Note: the L2 penalty is incorporated in optimizer
        value_loss = F.mse_loss(position_score_batch.view(-1), winner)
        policy_loss = -torch.mean(torch.sum(mcts_probability * log_probability_batch, 1))
        # the loss is the sum of the value loss and the policy loss
        loss = value_loss + policy_loss
        
        # backpropagation
        loss.backward()
        self.optimizer.step()
        
        # return the loss item and the policy item for monitor only
        entropy = -torch.mean(torch.sum(torch.exp(log_probability_batch) * log_probability_batch, 1))
        return loss.item(), entropy.item()


    # save the trained model
    def save_model(self, model_file):
        torch.save(self.value_net.state_dict(), model_file)
    