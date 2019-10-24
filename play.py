import snake
import qlearning
import numpy as np
import pandas as pd
import math

from keras.models import Sequential
from keras.layers import Dense
import keras.metrics

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


#Using QBot to make training data for Neural Network
#################################################################

train_in = []
train_out = []

def get_input_state(snake):
    new_input = []
    new_output = []

    head = snake.body[0]

    #Get board state in 4 directions
    moves = [np.array((1,0)),np.array((-1,0)),np.array((0,1)),np.array((0,-1))]
    for move in moves:
        #Get possible next moves for input
        coord = head + move
        if not((coord>=0).all() and (coord<snake.state.shape[0]).all()): #If out of bounds
            new = 1.0
        else:
            new = snake.state[coord[0],coord[1]]
        new_input.append(new)

        #Get value of possible moves for output
        if new == 1:
            new_output.append(-1) #-1 if move will result in crash
        elif new == 3:
             new_output.append(1)
        else:
            if np.linalg.norm(coord-snake.fruit_location) <= np.linalg.norm(head-snake.fruit_location):
                new_output.append(1) #1 if move get closer to fruit
            else:
                new_output.append(0) #0 if not closer to fruit but not dead
        
    #Append coord diff between head and fruit
    food = head - snake.fruit_location
    new_input.append(food[0])
    new_input.append(food[1])

    return new_input, new_output

def extract_train_data(snake):

    new_input, new_output = get_input_state(snake)
    train_in.append(new_input)
    train_out.append(new_output)
    
    

#################################################################



class Play_Snake:

    def __init__(self, player=None):
        self.snake = snake.Snake(10)
        self.player = player
        self.graphics = snake.Draw(self.snake)
        self.score = []

        #Bind keyboard events
        self.graphics.master_tk.bind("<Up>",self.up)
        self.graphics.master_tk.bind("<Down>",self.down)
        self.graphics.master_tk.bind("<Right>",self.right)
        self.graphics.master_tk.bind("<Left>",self.left)
        self.play()
        self.graphics.master_tk.mainloop()


    def play(self):
        if isinstance(self.player,qlearning.QLearning):
            while True:
            #for x in range(10000):
                self.player.step(self.snake)
                if self.snake.game_over:
                    from statistics import mean
                    self.score.append(self.snake.score)
                    print(self.score)
                    if len(self.score) == 20:
                        print(mean(self.score))
                        self.score = []

                ##############################################
                #extract_train_data(self.snake)  #Temporary to collect NN training data
                ##############################################

                self.graphics.update(self.snake)
                self.graphics.master_tk.after(100)
            
            #np.savetxt("C:/Users/Kilby/Code/Waterloo/CS680/Project/train_in.csv", np.asarray(train_in), delimiter=",")
            #np.savetxt("C:/Users/Kilby/Code/Waterloo/CS680/Project/train_out.csv", np.asarray(train_out), delimiter=",")
   
        
        if isinstance(self.player, keras.engine.sequential.Sequential):
            while True:
                state = self.snake.get_basic_state2()[:6]
                step = self.player.predict(np.asarray([state])).flatten()

                #Note: different order than qbot's action_dict!
                #May need to reverse up and down
                action_dict = {
                0:np.array((1,0)), #down
                1:np.array((-1,0)),  #up
                2:np.array((0,1)),  #right
                3:np.array((0,-1))  #left
                }

                action = action_dict[np.argmax(step)]

                #Move snake
                self.snake.turn(action)

                self.snake.step()

                if self.snake.game_over:
                    from statistics import mean
                    self.score.append(self.snake.score)
                    print(self.score)
                    if len(self.score) == 20:
                        print(mean(self.score))
                        score = []




                self.graphics.update(self.snake)
                self.graphics.master_tk.after(100)
        
        if isinstance(self.player, GeneticSnake):
            steps = 0
            while True:
                steps += 1
                input = torch.tensor(self.snake.get_basic_state2()).type('torch.FloatTensor').view(1,-1)
                output = self.player(input).detach().numpy()[0]
                action = self.snake.action_dict[np.argmax(output)]
                
                #Perform action
                self.snake.turn(action)
                self.snake.step()

                if self.snake.game_over:
                    from statistics import mean
                    self.score.append(self.snake.score)
                    print(self.score)
                    if len(self.score) == 20:
                        print(mean(self.score))
                        score = []
                    print(steps)
                    steps = 0

                self.graphics.update(self.snake)
                self.graphics.master_tk.after(100)
        
        if self.player == None:
            while True:
                
                self.graphics.update(self.snake)
                self.human_priors(self.snake.dir)
                self.graphics.master_tk.after(100)
                self.snake.step()


    def human_priors(self, action):

        new_action = []
        v = action[0]
        h = action[1]

        if v == -1:
            actions.append([0,1,0,0])
        elif v == 1:
            actions.append([1,0,0,0])
        elif h == 1:
            actions.append([0,0,1,0])
        elif h == -1:
            actions.append([0,0,0,1])

        states.append(self.snake.get_basic_state2()[:6])

        if len(states) % 20 == 0:
            np.savetxt("C:/Users/Kilby/Code/Waterloo/CS680/Project/human-data/FINAL_s.csv", states, delimiter=",")
            np.savetxt("C:/Users/Kilby/Code/Waterloo/CS680/Project/human-data/FINAL_a.csv", np.asarray(actions), delimiter=",")

    #Keyboard events
    def up(self, event):
        self.snake.turn(np.array((-1,0)))
        #self.snake.step()
        #self.graphics.update(self.snake)
        #self.human_priors([-1,0])

    def down(self, event):
        self.snake.turn(np.array((1,0)))
        #self.snake.step()
        #self.graphics.update(self.snake)
        #self.human_priors([1,0])
    def right(self, event):
        self.snake.turn(np.array((0,1)))
        #self.snake.step()
        #self.graphics.update(self.snake)
        #self.human_priors([0,1])
    def left(self, event):
        self.snake.turn(np.array((0,-1)))
        #self.snake.step()
        #self.graphics.update(self.snake)
        #self.human_priors([0,-1])

#temporary!!!    
class GeneticSnake(nn.Module):
        def __init__(self, in_size, out_size):
            super().__init__()
            self.model = nn.Sequential(
                        nn.Linear(in_size,40, bias=True),
                        nn.ReLU(),
                        nn.Linear(40,out_size, bias=True),
                        nn.Softmax(dim=1)
                        )

                
        def forward(self, inputs):
            x = self.model(inputs)
            return x


print(1)
states = []
actions = []


qbot = qlearning.QLearning(10, 0.00)

nnbot = keras.models.load_model("C:/Users/Kilby/Code/Waterloo/CS680/Project/NN1.h5")

gabot = GeneticSnake(6, 4)
gabot.load_state_dict(torch.load("C:/Users/Kilby/Code/Waterloo/CS680/Project/genetic_champs/winners1/champ29.pt"))
gabot.eval()

#Training data:
#Import test data and training data


play = Play_Snake(qbot)


