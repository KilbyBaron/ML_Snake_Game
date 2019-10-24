
# Empirically Evaluating 3 Machine Learning Algorithms and Their Ability to Learn the Game Snake

## Introduction:

The rules of Snake are very simple; the player controls the head of a snake as it navigates a NxN grid and tries to collect “fruit”. At every step, the snake can move its head in one of four cardinal directions and its body will follow. If the snake runs into a wall or its own body the game is over. If the snake runs into a fruit its score increases and the length of its tail increases by 3. The objective is to collect as many fruits as possible. I programmed the game in Python and implemented the graphics using the tkinter library.  This allowed me to easily let my models control the game and collect game data for training.
I decided to create 3 AIs and I trained them to play snake using 3 machine learning techniques that are supposedly the best for solving this type of problem. The purpose of this project is to allow me to explore new facets of machine learning and develop my practical skills. I hope that being able to watch the final models play the game in real time will also offer some additional perspective about how each technique learns.

## Neural network using human priors 

### The Algorithm:
The first technique that I experimented with is simple supervised learning of a neural network. For this experiment, I collected a set of human priors by playing the game for about 30 minutes. This allowed me to collect a set of game states and corresponding actions which I could use as training data for a simple neural network. In total I collected 4300 game states and actions. 
The input was fed into a neural network implemented in python using the Keras library. The Keras model consists of two dense hidden layers, each with 20 hidden units and sigmoid activation functions. I equipped the model with an Adam optimizer and measured loss by mean squared error. I experimented with different numbers of hidden layers, hidden units, and activation functions, but the results were always very similar.
Every input state is vector of length 6 that represents the current game state. The first 4 values indicate what lies immediately in each direction; the walls and snake body are represented by the value -1, open space is 0, and fruit is a 1. The 5th and 6th values can be -1, 0, or 1 and indicate if the fruit is up or down and left or right. The output is a softmax vector of length 4, each representing a direction. The final direction is the argmax of the output vector.
The training results after 500 epochs are below. The model does not improve very much after about 150 epochs. The MSE levels out at about 0.07 which is due to inconsistency in my own actions. For example, if I make two different actions in the same state then the model will never be able to perfectly separate the data.

### Results:
To evaluate the competency of the model trained based on my own gameplay, I made the model play 100 games of Snake and recorded its scores, with every fruit being worth 1 point. After 100 trials, the model’s average score was 7.25 and its average length was 22.75. To put this score in perspective, I put myself through the same test; after 100 games my average score was only 5.42 and my high score was 13. Therefore, the AI trained off of my own gameplay is better at snake than I am. Most of my own crashes are due to timing mistakes, whereas the AI tends to crash because it corners itself. I believe that the AI performs better than me because my timing mistakes are infrequent enough that they did not get learned by the model.
When I play the game, I tend to weave back and forth and make big arcs to avoid cornering myself. Interestingly, the AI seems to have adopted these strategies as well, and I suspect this helps it survive for longer. I included some screenshots of the AI’s gameplay below.





## Q-Learning

### The Algorithm:
The second technique that I experimented with to train an AI to play snake is called Q-Learning. My implementation of a Q-Learning algorithm is based on the 1992 paper “Q-Learning”, by Watkins and Dayan [5]. As described in this paper, “Q-Learning provides agents with the capability of learning to act optimally in Markovian domains by experiencing the consequences of actions, without requiring them to build maps of the domains” [5].  Markov models are used for reasoning and computation of randomly changing systems that would otherwise be intractable by assuming that future states rely only on the current state [5]. Since the game snake exhibits the Markov property, I concluded that Q-Learning would be a very effective type of reinforcement learning for this project.
As mentioned above, Q-Learning allows an agent to navigate an environment using a finite collection of actions without building a map of the environment itself [5]. Instead, the agent amasses an enormous list of every state that it has ever been to [5]. Each state is paired with a set of “Q-Values” which represent the expected reward for executing each action at the current state [5]. All of the states and respective Q-Values are stored in a table called a Q-Table. At every time step, the agent searches for its current state in the Q-Table; if the state is present, it executes the action with the largest Q-Value, if the state is not in the Q-Table, it chooses a random action and updates the Q-Table with the results [5].  
More specifically, at each state, n, the agent:
    • Observes the current state xn
    • Selects and performs an action an
    • Observes the subsequent state yn
    • Receives an immediate payoff rn
    • And adjusts its Qn-1 values using a learning factor αn
The Q-Values are adjusted using the following function Q(x, a) [5]:

In other words, the old value for Qn-1(x, a) is adjusted by the immediate reward plus the discounted optimal future value of the new state times the learning rate. Thus, over millions of iterations, the algorithm learns to take actions that maximize both immediate and future rewards. To ensure that the model explores all options, there is also a small possibility at each step that the agent ignores its Q-Table and chooses a random action. When training started, I set this value to 5%; however, once it had explored all of the most common states, I lowered the possibility of random action to 1%.

### Implementation:
In my implementation, each step will return one of three immediate rewards. If the step causes the snake to run into a wall or its own body, the reward is -5. If the step causes the snake to eat a piece of fruit, the reward is +5. In all other scenarios the reward is -0.1; this is meant to discourage the snake from taking long paths and getting stuck in loops but is kept small to maintain the value of long-term rewards.  
My goal in defining each state was to give the model as much information as possible with the least amount of unique states. Increasing the number of possible states will cause the snake to learn more slowly and cause the Q-Table to occupy more space. At each step, the state is defined by 9 values:
    • Values 1-4: What lies in each cardinal direction (-1 = wall or body, 0 = nothing, 1 = fruit)
    • Values 5&6: Direction of the fruit from the head (1/-1 for up/down, 1/-1 for left/right)
    • Values 7&8: Direction of the tail from the head
    • Value 9: Length of the snake
### Results:
Over the span of 48 hours, the snake took 116,000,000 steps, and visited 43628 unique states. The graph below illustrates how many new states were learned every thousand steps. 

As you can see, even after over 100 million states, the snake was still encountering new states. Keep in mind, also, that in order to make the most intelligent decision, the snake must encounter each state several times. For example, it will take a minimum of four encounters just to get an approximate value of the reward for each possible action. However, since future rewards propagate backwards one step at a time, the snake has to perform the same action at the same state several times to learn to target a fruit that is a few steps away. Therefore, the snake in my final result could still be improved given several more days of training.
To test the final Q-Learning AI (which I fondly named “Q-Bot”), I made it play 100 games of snake, like I did for the previous model. After 100 test games, the Q-Bot’s average score was 7.75 and its high score was 13. Q-Bot’s movement remains somewhat erratic and it does not implement a noticeable strategy, however its play style in the end is still effective. There are times that Q-Bot turns straight into the wall out of the blue; this indicates that there are still many states that have been visited only a handful of times. Given several more days of training I suspect Q-Bot would score much higher. I included screenshots of the Q-Bot’s gameplay to show how it differs from the first experiment.





## Neural network using genetic random mutation

### The Algorithm:
The third algorithm that I chose to implement for this project is a simple genetic algorithm. Evolution strategies have been proven to be an effective alternative to back-propagation algorithms like Q-Learning [3].  My genetic algorithm is based on the paper “Deep Neuro-evolution: Genetic Algorithms are a Competitive Alternative for Training Deep Neural Networks for Reinforcement Learning” by Uber AI Labs [3]. While most evolution strategies perform some form of stochastic gradient descent, this paper investigates the effectiveness of a gradient-free genetic algorithm with considerable success [3]. I chose to replicate this algorithm for my third solution because it was proven to be effective for simple problems and it converges quickly.
Genetic algorithms work by gradually evolving a population of unique agents called “genotypes” over several “generations” [3]. In this case every genotype is a neural network initialized with random weights. At each generation, every genotype is evaluated and assigned a “fitness score” for how effective it is at completing a given task [3]. The top T genotypes – those with the highest fitness scores - are selected to become parents of the next generation, this is known as truncated selection [3]. To produce a genotype of the next generation, a parent is selected at random and is “mutated” by applying additive Gaussian noise to the parameters of the selected parent [3]. This operation is typically performed N-1 times where N is the size of the original population [3]. In the paper, the Nth individual is a copy of the best individual from the previous generation, a technique known as elitism [3]. However, since the genotypes’ performances were very inconsistent in my project, I copied all of the parents into the new population at each iteration. 
Historically, there are a few different techniques for performing the mutations [4]. Some mutation techniques involve modifying the structure of the neural network itself; for example, changing the activation functions or number of hidden layers [4]. Other genetic algorithms use a technique called crossover, whereby the parameters of multiple parents are combined to produce offspring [4]. However, the reference paper did not include these types of mutation for simplicity, therefore neither did I. 
### Implementation:
In order to implement the genetic algorithm, I had to select several parameters. This includes the number of hidden layers in the neural network, the number of hidden units in each layer, the population size, and the number of top genotypes at each evolution. Since the number of hidden layers increases the run-time substantially, I restricted the neural network to one hidden layer. To select the other 3 parameters, I performed a series of tests to determine how each parameter affects the results.
The tests helped me compare the affects that the generation size, top performer cutoff range, and number of hidden units have on the model. For each comparison I trained 3-4 different models for 150 generations, varying one parameter while keeping the others constant. 
My tests show that when the number of hidden units in the neural network is between 10 and 40, the results are essentially the same. However, when the number of hidden units is increased to 80 the model learns just as quickly and performs twice as well. I could not test the number of hidden units beyond 80 because storing the models for each generation required too much computer memory.
Next, I discovered that generation size mainly affects the learning rate. Having more genotypes at each generation allows the model to discover optimal weights more quickly and show less variance, but after 150 generations the models converged to about the same average score.

Finally, the tests comparing the number of genotypes used to create the next generation showed that selecting only 10 top performers allows the model to learn more quickly than selecting 20 or 30. However, like the generation size, after 150 generations, all models perform very similarly.
Therefore, for my final test I trained a genetic neural network with 80 hidden units, with a population size of 300 and a generation cutoff size of 10. To help the model converge on the fruit, I made some modifications to the scoring. If the snake moved towards the fruit, I incremented the score by 0.1, but if it moved away from the fruit, I decreased the score by 0.11. This also helped prevent the snake from spinning in circles.


### Results:

Despite my effort to train the genetic algorithm with the most effective parameters, the accuracy converged to about the same average score as the trial models even after 940 generations. I selected the best performing genotype of the final generation to evaluate the effectiveness of the algorithm. Like the previous 2 evaluations, I made the final genetic champion play snake 100 times, awarding 1 point per fruit eaten. After 100 games, the genetic champion’s average score was 8.25, and its high score was 22.
The genetic algorithm converged on a strategy that is much more discernable than its Q-Learning counterpart. It mainly navigates the perimeter of the environment in wide counterclockwise circles and turns toward the fruit if it reaches the same row or column. It will also occasionally double back on itself when it reaches the opposite side. Interestingly, this is somewhat similar to the strategy that I attempt to implement in my own gameplay. Navigating the perimeter seems to work quite well because it minimizes the snake’s interactions with its own body. When the genetic champion does come close to its body it is much more likely to get confused and crash. 
I believe this may indicate a limitation of this simple genetic algorithm. If the model converges on a local maximum, and the global maximum lies on the other side of, for example, a saddle point, it may become stuck. This would explain why the genetic algorithm reaches a ceiling after about 200 generations. I included screenshots of the genetic champions gameplay.




## Conclusion

### Table I: Final Results

Human (me)
Human Priors
Q-Learning
Genetic
Average Score
5.42
7.25
7.75
8.25
High Score
13
15
13
22

After evaluating each method, all machine learning techniques performed better than I did. The neural network trained on human priors performs almost as well as the Q-Learning model, but both models employ very different navigation strategies. Surprisingly, the most successful technique was the genetic algorithm despite the fact that it does not employ any form of gradient descent.
Training the neural network with human priors takes the least amount of time – about a minute. The genetic algorithm was second fastest, a population of 300 can be mutated and tested in about 12 seconds; therefore 200 generations can be trained in about 40 minutes. Q-Learning takes the most time by far, the Q-Bot was trained for 48 hours before being tested; however, its behavior indicates that it has much more room for improvement. Given more time I would have liked to try training the Q-Bot with smaller game states. A smaller number of possible unique states would let the Q-Bot to train faster and have more experience at each state.
I believe that the human-trained model cannot improve much further unless it were retrained using the data of a more skilled player. The Training graph of the genetic AI also seems to indicate that more training time would not have improved its results. Therefore, given a short period of time the genetic algorithm is superior, but I believe that Q-Learning displays the most potential.
Bibliography

    1. Lelis, Italo. “LearnSnake: Teaching an AI to Play Snake Using Reinforcement Learning - Italo Lelis.” LearnSnake: Teaching an AI to Play Snake Using Reinforcement Learning - Italo Lelis - Engineering Student, https://italolelis.com/snake.
    2. Paras Chopra. “Reinforcement Learning without Gradients: Evolving Agents Using Genetic Algorithms.” Medium, Towards Data Science, 7 Jan. 2019, https://towardsdatascience.com/reinforcement-learning-without-gradients-evolving-agents-using-genetic-algorithms-8685817d84f.
    3. Such, Felipe Petroski, et al. "Deep neuroevolution: Genetic algorithms are a competitive alternative for training deep neural networks for reinforcement learning." arXiv preprint arXiv:1712.06567 (2017).
    4. Srinivas, M., & Patnaik, L. M. (1994). Adaptive probabilities of crossover and mutation in genetic algorithms. IEEE Transactions on Systems, Man, and Cybernetics, 24(4), 656-667.
    5. Watkins, C. J., & Dayan, P. (1992). Q-learning. Machine learning, 8(3-4), 279-292.

