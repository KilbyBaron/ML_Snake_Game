
#Reference: https://github.com/paraschopra/deepneuroevolution/blob/master/openai-gym-cartpole-neuroevolution.ipynb

import snake
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from statistics import mean
import random
import copy
import os
import sys

class GeneticSnake(nn.Module):
        def __init__(self, in_size, out_size, hidden_units):
            super().__init__()
            self.model = nn.Sequential(
                        nn.Linear(in_size,hidden_units, bias=True),
                        nn.ReLU(),
                        nn.Linear(hidden_units,out_size, bias=True),
                        nn.Softmax(dim=1)
                        )

                
        def forward(self, inputs):
            x = self.model(inputs)
            return x


#Initialize weights (found this here: https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch)
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.00)

def test_agents(agents, max_steps):
    results = []
    agent_number = 0
    game = snake.Snake(10)

    for agent in agents:
        print("evaluating agent number:",agent_number+1,"/",len(agents), end='\r')
        agent.eval() #put in eval mode

        scores = []
        for x in range(1):#Test each agent 5 times to get average score

            score = 0
            for _ in range(max_steps):#Let the agent play until it dies or reaches max_steps

                #Generate an action
                input = torch.tensor(game.get_basic_state2()).type('torch.FloatTensor').view(1,-1)
                output = agent(input).detach().numpy()[0]
                action = game.action_dict[np.argmax(output)]
                
                #Perform action
                game.turn(action)
                game.step()

                if game.game_over:
                    score += game.score
                    game.reset(10)

            #Save final score and reset game
            scores.append(score)
            game.reset(10)
        
        results.append([round(mean(scores),2),agent_number])
        agent_number += 1
        
    return results

def evolve(stock, num_agents, gen_cutoff):
    new_generation = []

    #Add champions to new generation
    for s in stock:
        new_generation.append(s)

    #Mutate champions to fill up rest of generation
    for i in range(num_agents-gen_cutoff):
        print("Creating mutant number:",i+gen_cutoff+1,"/",num_agents, end='\r')
        mutant = mutate(random.choice(stock))
        new_generation.append(mutant)

    return new_generation

def mutate(agent):
    child = copy.deepcopy(agent)
    mutation_power = 0.02

    for param in child.parameters():

        if len(param.shape) == 2:
            for i0 in range(param.shape[0]):
                for i1 in range(param.shape[1]):
                    param[i0][i1] += mutation_power * np.random.randn()
        
        else:
            for i0 in range(param.shape[0]):
                param[i0] += mutation_power * np.random.randn()

    return child



def get_champs(agents, results, cutoff):
    top = sorted(results)[-cutoff:]

    champs = []
    for t in top:
        champs.append(copy.deepcopy(agents[t[1]]))

    del agents
    
    return champs

#return a bunch of random agents
def init_agents(n, in_size, out_size, hidden_units):
    agents = []
    hidden_units = int(sys.argv[3])
    for _ in range(n):
        new_agent = GeneticSnake(in_size, out_size, hidden_units)
        init_weights(new_agent)
        agents.append(new_agent)
    return agents


def load_models(gen_cutoff, num_agents, in_size, num_actions, hidden_units, dir):
    
    if not os.path.isfile(dir+"0.pt"):
        print("Creating first generation!")
        return init_agents(num_agents, in_size, num_actions, hidden_units) #If no models are saved

    agents = []
    for x in range(gen_cutoff):
        if not os.path.isfile(dir+str(x)+".pt"):
            break
        new_agent = GeneticSnake(in_size, num_actions, hidden_units)
        new_agent.load_state_dict(torch.load(dir+str(x)+".pt"))
        new_agent.eval()
        agents.append(new_agent)

    
    return evolve(agents, num_agents, gen_cutoff)



def run():
    print("\n")
    print("                           ____\n"+
    "  ________________________/ O  \___/\n"+
    " <_/_\_/_\_/_\_/_\_/_\_/_______/   \  \n")
    print("Mutating champions")

    board_size = 10
    in_size = 9
    num_actions = 4

    num_agents = int(sys.argv[1])
    gen_cutoff = int(sys.argv[2])
    hidden_units = int(sys.argv[3])
    name = sys.argv[4]

    dir = "C:/Users/Kilby/Code/Waterloo/CS680/Project/cross-validation/"+name+"/"
    if not os.path.exists(dir):
        os.makedirs(dir)

    agents = load_models(gen_cutoff, num_agents, in_size, num_actions, hidden_units, dir)

    print("\n\nTesting agents...")
    results = test_agents(agents, 40)

    sum = 0
    for r in results:
        sum += r[0]
    avg = round(sum/len(results),2)

    print("\nAverage score:",avg)
    print("Grand champion:",sorted(results)[-1][0])
    print("\n")

    champions = get_champs(agents, results, gen_cutoff)

    #Save champions
    for c in range(len(champions)):
        torch.save(champions[c].state_dict(),dir+str(c)+".pt")

    #Save average
    with open(dir+"avg_per_generation.txt", "a") as f:
        f.write(str(avg)+"\n")
    os.chdir(dir)
    avgs = []
    with open(dir+"avg_per_generation.txt","r",encoding="utf8") as f:
        for line in f.readlines():
            avg = round(float(line),2)
            avgs.append(avg)
    np.savetxt(dir+"avg_per_generation.csv", np.asarray(avgs), delimiter=",")

run()