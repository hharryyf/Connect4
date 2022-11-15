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

class ConvBlock(nn.Module):
    def __init__(self):
        super(ConvBlock, self).__init__()
        self.action_size = 7
        self.conv1 = nn.Conv2d(3, 128, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(128)

    def forward(self, s):
        s = s.view(-1, 3, 6, 7)  # batch_size x channels x board_x x board_y
        s = F.relu(self.bn1(self.conv1(s)))
        return s

class ResBlock(nn.Module):
    def __init__(self, inplanes=128, planes=128, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = F.relu(out)
        return out
    
class OutBlock(nn.Module):
    def __init__(self):
        super(OutBlock, self).__init__()
        self.conv = nn.Conv2d(128, 3, kernel_size=1) # value head
        self.bn = nn.BatchNorm2d(3)
        self.fc1 = nn.Linear(3*6*7, 32)
        self.fc2 = nn.Linear(32, 1)
        
        self.conv1 = nn.Conv2d(128, 32, kernel_size=1) # policy head
        self.bn1 = nn.BatchNorm2d(32)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.fc = nn.Linear(6*7*32, 7)
    
    def forward(self,s):
        v = F.relu(self.bn(self.conv(s))) # value head
        v = v.view(-1, 3*6*7)  # batch_size X channel X height X width
        v = F.relu(self.fc1(v))
        v = torch.tanh(self.fc2(v))
        
        p = F.relu(self.bn1(self.conv1(s))) # policy head
        p = p.view(-1, 6*7*32)
        p = self.fc(p)
        p = self.logsoftmax(p)
        return p, v

class Net(nn.Module):
    # after many failure, I believe a ResNet is important, try the ResNet from this repo
    # https://github.com/ThePrincipalComponent/AlphaZeroConnect4/blob/main/Part%205/model.py
    def __init__(self, res_block_num=5):
        super().__init__()
        self.conv = ConvBlock()
        self.res_block_num = res_block_num
        for block in range(self.res_block_num):
            setattr(self, "res_%i" % block,ResBlock())
        self.outblock = OutBlock()
    
    def forward(self,s):
        s = self.conv(s)
        for block in range(self.res_block_num):
            s = getattr(self, "res_%i" % block)(s)
        s = self.outblock(s)
        return s

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
        with torch.autograd.set_detect_anomaly(True):
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
    