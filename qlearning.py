import numpy as np
import random
import math
import pandas as pd
import time
import os



class QLearning:

    def __init__(self, size, rand):
        self.board_size = size
        self.learning_rate = 0.4 #arbitrary
        self.rand = rand #from paper
        self.discount_factor = 0.8 #arbitrary
        
        self.action_dict = {
            0:np.array((-1,0)), #up
            1:np.array((1,0)),  #down
            2:np.array((0,1)),  #right
            3:np.array((0,-1))  #left
        }

        #Get Q-Table
        try:
            self.states = pd.read_csv("C:/Users/Kilby/Code/Waterloo/CS680/Project/Qdata/qtable_states"+str(self.board_size)+".csv",header=None).to_numpy()
            self.actions = pd.read_csv("C:/Users/Kilby/Code/Waterloo/CS680/Project/Qdata/qtable_actions"+str(self.board_size)+".csv",header=None).to_numpy()

        #Make new qtable with 2 entries
        except FileNotFoundError:
            self.states = np.vstack([np.zeros(9), np.zeros(9)])
            self.actions = np.vstack([np.zeros(4),np.zeros(4)])

            self.update_csv()

    def get_state(self, snake):
        return snake.get_basic_state2()
    
    def step(self, snake):

        #Find state in qtable
        s = np.where((self.states == self.get_state(snake)).all(axis=1))[0]
        if len(s) == 0:
            self.states = np.vstack([self.states, self.get_state(snake)])
            self.actions = np.vstack([self.actions,np.zeros(4)])
            row_index = self.states.shape[0]-1
        else:
            row_index = s[0]
        
        #Determine action
        action_index = np.argmax(self.actions[row_index])
        if random.uniform(0,1) < self.rand:  #choose a random action a fraction of the time
            action_index = random.randint(0,3)
        action = self.action_dict[action_index]

        #Move snake
        old_score = snake.score
        snake.turn(action)
        snake.step()
        reward = snake.score - old_score

        #Find optimal next move
        optimal_future = 0
        s = np.where((self.states == self.get_state(snake)).all(axis=1))[0]
        if len(s) > 0:
            optimal_future = np.amax(self.actions[s[0]])

        #Update actions score for most recent action
        new_value = self.actions[row_index][action_index] + self.learning_rate*(reward + self.discount_factor*optimal_future - self.actions[row_index][action_index])
        self.actions[row_index][action_index] = new_value

        return action
    
    def train(self, num_steps, snake):

        scores = []
        for s in range(num_steps):
            self.step(snake)

            #Reset if game over
            if snake.game_over:
                scores.append(snake.score)
                if len(scores) == 100:
                    from statistics import mean
                    print("mean:",mean(scores))
                    scores = []
                snake.reset(self.board_size)

            #Print training progress
            if s%1000 == 0:
                print(s,self.states.shape)
                with open("qlearn_num_states.txt", "a") as f:
                    f.write(str(self.states.shape[0])+"\n")
            #if s%100000 == 0:
                #self.update_csv()
    
        #self.update_csv()
    
    def update_csv(self):
        np.savetxt("C:/Users/Kilby/Code/Waterloo/CS680/Project/qtable_states"+str(self.board_size)+".csv", self.states, delimiter=",")
        np.savetxt("C:/Users/Kilby/Code/Waterloo/CS680/Project/qtable_actions"+str(self.board_size)+".csv", self.actions, delimiter=",")

# import snake
# qbot = QLearning(0, 0.01)
# qbot.train(0, snake.Snake(10))

# #Working directory
# dir = "C:/Users/Kilby/Code/Waterloo/CS680/Project/"
# os.chdir(dir)
# avgs = []
# with open(dir+"qlearn_num_states.txt", encoding="utf8") as f:
#     for line in f.readlines():
#         avgs.append(int(line))
# np.savetxt("n_qstates.csv", np.asarray(avgs), delimiter=",")