import math
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

import snake


class SnakeState(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, snake, dir):
        #Move snake
        action = snake.action_dict[torch.argmax(dir).item()]
        snake.turn(action)
        snake.step()
        output = torch.from_numpy(snake.state.flatten()).unsqueeze(0).float()
        output.requires_grad = True
        return output
        
        

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        
        self.hidden_size = hidden_size

        self.lstm = nn.LSTMCell(input_size, hidden_size)
        self.out = nn.Linear(self.hidden_size, output_size)
        self.snakeState = SnakeState()



    def forward(self, input, hidden, cell, snake):
        hidden, cell = self.lstm(input, (hidden, cell))
        output = self.snakeState(snake, self.out(hidden))
        
        return output, hidden, cell


    def initHidden(self):
        return torch.zeros(1, self.hidden_size)      
      
    def initCell(self):
        return torch.zeros(1, self.hidden_size)

def move_snake(snake, action):
    snake.turn(action)
    snake.step()

n_hidden = 128
n_moves = 4
snake_board_size = 10
input_size = snake_board_size*snake_board_size
learning_rate = 0.005 # If you set this too high, it might explode. If too low, it might not learn

lstm = LSTM(input_size, n_hidden, n_moves)

criterion = nn.L1Loss()


def train():

    import snake
    snake = snake.Snake(snake_board_size)
    
    hidden = lstm.initHidden()
    cell = lstm.initCell()
    lstm.zero_grad()
    output = torch.from_numpy(snake.state.flatten()).unsqueeze(0).float()


    while snake.game_over == False:
        output, hidden, cell = lstm(output, hidden, cell, snake)
        


    output.requires_grad = True
    loss = criterion(output, torch.zeros(1, 1))
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in lstm.parameters():
        if p.grad:
            print(1)
            p.data.add_(-learning_rate, p.grad.data)


    return snake.score, loss




n_iters = 10000
print_every = 50
current_loss = 0
avg_score = 0

for iter in range(1, n_iters + 1):
    
    output, loss = train()
    current_loss += loss
    avg_score += output

    # Print iter number, loss, name and guess
    if iter % print_every == 0:
        print("Average score:",avg_score/print_every)
        avg_score = 0
        

