from tkinter import *
import numpy as np
import random
import math
import pandas as pd
import time
#import qlearning

class Snake:
    def __init__(self, size):
        self.action_dict = {
            0:np.array((1,0)), #down
            1:np.array((-1,0)),  #up
            2:np.array((0,1)),  #right
            3:np.array((0,-1))  #left
        }
        self.reset(size)

    def turn(self, dir):
        self.dir = dir
    
    def spawn_fruit(self):
        grid_size = self.state.shape[0]
        spawn = random.randint( 0, grid_size**2 - np.count_nonzero(self.state) - 1)

        row = 0
        while spawn >= grid_size - np.count_nonzero(self.state[row]):
            spawn = spawn - (grid_size - np.count_nonzero(self.state[row]))
            row += 1

        col = 0
        while spawn > 0 or self.state.item((row,col)) != 0:
            if self.state.item((row,col)) == 0:
                spawn -= 1
            col += 1
        
        self.fruit_location = np.array((row,col))
        self.state.itemset((row,col),3)
    
    def step(self):
        #self.score -= 0.1

        #check end game scenarios
        new_head = self.body[0]+self.dir
        if not((new_head>=0).all() and (new_head<self.state.shape[0]).all()): #If out of bounds
            #self.score -= 5
            self.game_over = True

        elif self.state.item((new_head.item(0),new_head.item(1))) == 1: #If collsion with self
            #self.score -= 5
            self.game_over = True

        

        #Update if no collision
        else:
            #If snake ate fruit
            if self.state.item((new_head[0],new_head[1])) == 3:
                self.score += 1
                self.grow += 3
                self.spawn_fruit()

            #+1 score if move gets closer to fruit
            #elif np.linalg.norm(new_head-self.fruit_location) <= np.linalg.norm(self.body[0]-self.fruit_location):
            #    self.score += 1
            
            #else:
            #    self.score -= 1.1
            
            self.state.itemset((self.body.item((0,0)),self.body.item((0,1))),1) #old head becomes body in state
            self.body = np.vstack([np.array([new_head]),self.body]) #Add new head
            self.state.itemset((self.body.item((0,0)),self.body.item((0,1))),2) #Add new head to state

            #Move tail if necessary
            if self.grow > 0:
                self.grow -= 1
            else:
                self.state.itemset((self.body.item((-1,0)),self.body.item((-1,1))),0)
                self.body = np.delete(self.body,-1, axis=0)
    

    def reset(self, size):

        self.state = np.zeros((size,size)) #Make empty board
        self.body = np.array([[math.floor(size/2),math.floor(size/2)]]) #Place snake in center
        self.state.itemset((self.body.item((0,0)),self.body.item((0,1))),2)
        self.dir = np.array((0,1)) #Set initial direction
        self.grow = 2
        self.spawn_fruit() #Make a fruit
        self.game_over = False
        self.score = 0
    
    def get_basic_state2(self):
        state = []
        head = self.body[0]

        #Get board state in 4 directions
        moves = [np.array((1,0)),np.array((-1,0)),np.array((0,1)),np.array((0,-1))]
        for move in moves:
            #Get possible next moves for input
            coord = head + move
            if not((coord>=0).all() and (coord<self.state.shape[0]).all()): #If out of bounds
                new = 1.0
            else:
                new = self.state[coord[0],coord[1]]
            state.append(new)
     
        #Append coord diff between head and fruit
        food = np.sign(head - self.fruit_location)
        state.append(food[0])
        state.append(food[1])
        
        tail = np.sign(head - self.body[-1])
        state.append(tail[0])
        state.append(tail[1])

        state.append(len(self.body))

        return np.asarray(state)


class Draw:

    def __init__(self, snake):

        #Make canvas
        self.canvas_size = 300
        self.grid_size = snake.state.shape[0]
        self.master_tk = Tk()
        self.canvas = Canvas(self.master_tk, bg = "black", height = self.canvas_size-1, width = self.canvas_size-1)

        #Fill canvas with squares
        self.grid = []
        square_size = self.canvas_size/self.grid_size
        for y in range(self.grid_size):
            row = []
            for x in range(self.grid_size):
                coords = (x*square_size,y*square_size)
                row.append(self.canvas.create_rectangle(coords[0], coords[1], coords[0]+square_size, coords[1]+square_size, fill="black", outline="black"))
            self.grid.append(row)

        #Initialize game
        self.tail = [snake.body.item((-1,0)),snake.body.item((-1,1))]
        self.update_grid([snake.body.item((0,0)),snake.body.item((0,1))],"green")
        self.update_grid(snake.fruit_location,"red")

        self.canvas.pack()

        #Draw everything
        self.update(snake)

    #Change the color of a particular square
    def update_grid(self,coord,fill):
        self.canvas.itemconfig(self.grid[coord[0]][coord[1]], fill=fill, outline=fill)

    def update(self, snake):

        #Make everything black
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                self.update_grid([x,y],"black")

        #Draw fruit
        fruit = [snake.fruit_location.item(0),snake.fruit_location.item(1)]
        self.update_grid(fruit,"red")

        #Draw snake
        snake_colour = (0, 204, 44)
        green_change_rate = snake_colour[1]/(self.grid_size**2)
        blue_change_rate = snake_colour[2]/(self.grid_size**2)
        for b in range(len(snake.body)):
            #snake_colour = (snake_colour[0],snake_colour[1]-green_change_rate,snake_colour[2]-blue_change_rate)
            s_colour = (
                math.floor(snake_colour[0]),
                math.floor(snake_colour[1]-(0.8*snake_colour[1])*(b/len(snake.body))),
                math.floor(snake_colour[2]-(0.8*snake_colour[2])*(b/len(snake.body)))
            )
            self.update_grid(snake.body[b],'#%02x%02x%02x' % s_colour)



        #If game over, reset
        if snake.game_over:
            snake.reset(self.grid_size)
            self.reset_grid(snake)

        self.master_tk.update() # update the display

            

    #Reset after game over
    def reset_grid(self, snake):
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                self.update_grid([x,y],"black")
        
        self.tail = [snake.body.item((-1,0)),snake.body.item((-1,1))]
        self.update_grid([snake.body.item((0,0)),snake.body.item((0,1))],"green")
        self.update_grid(snake.fruit_location,"red")
