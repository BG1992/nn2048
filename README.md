## Deep Reinforcement Learning - 2048 game
I guess you have heard of the 2048 game :) if not, please have a look here: https://play2048.co/

The goal of this game is to achieve the highest score, by moving and merging tiles being powers of 2. 
Sometimes, achieving the highest maximum tile on the board is also considered as the goal of this game.

Lately, I built a deep neural network intended to play 2048 game. 
This neural network has been trained following **deep reinforcement learning** approach. 
This is a pure reinforcement learning - the network learns how to play best while self-playing, starting from knowledge of rules only. 
No hardcoding, no strategies applying were used while training.

## Details of the learning approach

The learning approach used in the network can be called **TD(0) Expectimax learning**.

Let <a href="https://www.codecogs.com/eqnedit.php?latex=\Omega" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\Omega" title="\Omega" /></a> be a set of states in the game, **after** player made a move and **before** new tile is generated.
Then, let
<a href="https://www.codecogs.com/eqnedit.php?latex=V:&space;\Omega&space;\rightarrow&space;\mathbb{R}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?V:&space;\Omega&space;\rightarrow&space;\mathbb{R}" title="V: \Omega \rightarrow \mathbb{R}" /></a> denote value function of a given state. The highest value is, the 'better' corresponding state is.

How to model the V function? We will follow an approach in the spirit of Hamilton-Jacobi-Bellman equation. Let <a href="https://www.codecogs.com/eqnedit.php?latex=s" target="_blank"><img src="https://latex.codecogs.com/gif.latex?s" title="s" /></a> be a state from the set <a href="https://www.codecogs.com/eqnedit.php?latex=\Omega" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\Omega" title="\Omega" /></a>. Then,

<a href="https://www.codecogs.com/eqnedit.php?latex=V(s)&space;=&space;V(s)&space;&plus;&space;\alpha(EV_s(s')&space;-&space;V(s)))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?V(s)&space;=&space;V(s)&space;&plus;&space;\alpha(EM(s)&space;-&space;V(s)))" title="V(s) = V(s) + \alpha(EM(s) - V(s)))" /></a>,
where <a href="https://www.codecogs.com/eqnedit.php?latex=\alpha" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\alpha" title="\alpha" /></a> is a learning rate and <a href="https://www.codecogs.com/eqnedit.php?latex=EM(s)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?EM(s)" title="EM(s)" /></a> is an expectimax of the state <a href="https://www.codecogs.com/eqnedit.php?latex=s" target="_blank"><img src="https://latex.codecogs.com/gif.latex?s" title="s" /></a>.

What the expectimax of a state <a href="https://www.codecogs.com/eqnedit.php?latex=s" target="_blank"><img src="https://latex.codecogs.com/gif.latex?s" title="s" /></a> is?

This is the expected gain we can obtain from the state <a href="https://www.codecogs.com/eqnedit.php?latex=s" target="_blank"><img src="https://latex.codecogs.com/gif.latex?s" title="s" /></a>, assuming that after a new pile is generated, the best move is chosen. Being more formal:

<a href="https://www.codecogs.com/eqnedit.php?latex=EM(s)&space;=&space;\sum_{i&space;\in&space;T}&space;p_{i}\Big&space;({\max_{j&space;\in&space;A}}&space;\Big&space;(&space;r_{ij}&space;&plus;&space;\gamma&space;V(s_{ij})&space;\Big&space;)&space;\Big&space;)," target="_blank"><img src="https://latex.codecogs.com/gif.latex?EM(s)&space;=&space;\sum_{i&space;\in&space;T}&space;p_{i}\Big&space;({\max_{j&space;\in&space;A}}&space;\Big&space;(&space;r_{ij}&space;&plus;&space;\gamma&space;V(s_{ij})&space;\Big&space;)&space;\Big&space;)," title="EM(s) = \sum_{i \in T} p_{i}\Big ({\max_{j \in A}} \Big ( r_{ij} + \gamma V(s_{ij}) \Big ) \Big )," /></a>

where:
- <a href="https://www.codecogs.com/eqnedit.php?latex=A" target="_blank"><img src="https://latex.codecogs.com/gif.latex?A" title="A" /></a> is a space of available actions we can made from the state <a href="https://www.codecogs.com/eqnedit.php?latex=s_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?s_i" title="s_i" /></a>.
- <a href="https://www.codecogs.com/eqnedit.php?latex=T" target="_blank"><img src="https://latex.codecogs.com/gif.latex?T" title="T" /></a> is a space of available boards that can be generated from the state <a href="https://www.codecogs.com/eqnedit.php?latex=s" target="_blank"><img src="https://latex.codecogs.com/gif.latex?s" title="s" /></a>.
- <a href="https://www.codecogs.com/eqnedit.php?latex=p_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p_i" title="p_i" /></a> is a probability of generating <a href="https://www.codecogs.com/eqnedit.php?latex=i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?i" title="i" /></a>-th board from the state <a href="https://www.codecogs.com/eqnedit.php?latex=s" target="_blank"><img src="https://latex.codecogs.com/gif.latex?s" title="s" /></a>. 
- <a href="https://www.codecogs.com/eqnedit.php?latex=r_{ij},&space;s_{ij}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?r_{ij},&space;s_{ij}" title="r_{ij}, s_{ij}" /></a> are respectively rewards and states obtained after making move <a href="https://www.codecogs.com/eqnedit.php?latex=j&space;\in&space;A" target="_blank"><img src="https://latex.codecogs.com/gif.latex?j&space;\in&space;A" title="j \in A" /></a> in the state <a href="https://www.codecogs.com/eqnedit.php?latex=s_{i}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?s_{i}" title="s_{i}" /></a>.
- <a href="https://www.codecogs.com/eqnedit.php?latex=\gamma" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\gamma" title="\gamma" /></a> is a discount factor.

For all terminal positions <a href="https://www.codecogs.com/eqnedit.php?latex=s_t" target="_blank"><img src="https://latex.codecogs.com/gif.latex?s_t" title="s_t" /></a> (positions from which we can not make any legal move), <a href="https://www.codecogs.com/eqnedit.php?latex=V(s_t)&space;=&space;0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?V(s_t)&space;=&space;0" title="V(s_t) = 0" /></a>.

Reward in this game, after each move, is a sum of numbers of tiles that have been created. To avoid dealing issues with enormous outliers, I decided to smooth the reward a bit, namely: 

<a href="https://www.codecogs.com/eqnedit.php?latex=r_{ij}&space;=&space;\frac{log_2(reward_{ij}&space;&plus;&space;1)}{2}," target="_blank"><img src="https://latex.codecogs.com/gif.latex?r_{ij}&space;=&space;\frac{log_2(reward_{ij}&space;&plus;&space;1)}{2}," title="r_{ij} = \frac{log_2(reward_{ij} + 1)}{2}," /></a> 

where <a href="https://www.codecogs.com/eqnedit.php?latex=reward_{ij}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?reward_{ij}" title="reward_{ij}" /></a> is an actual reward received after making move <a href="https://www.codecogs.com/eqnedit.php?latex=j" target="_blank"><img src="https://latex.codecogs.com/gif.latex?j" title="j" /></a> in the state <a href="https://www.codecogs.com/eqnedit.php?latex=s_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?s_i" title="s_i" /></a>.

To achieve the convergence, <a href="https://www.codecogs.com/eqnedit.php?latex=\alpha" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\alpha" title="\alpha" /></a> is being decreased over phases of the learning, <a href="https://www.codecogs.com/eqnedit.php?latex=\gamma" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\gamma" title="\gamma" /></a> is being increased. They are modelled as polynomial/exponential functions, details can be found in the *nn2048Train* module.

While learning, parameter <a href="https://www.codecogs.com/eqnedit.php?latex=\epsilon=0.3" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\epsilon=0.3" title="\epsilon=0.3" /></a> is introduced, which corresponds to the probability of making 'random' move in a given state (if a random number from the uniform distribution over the interval <a href="https://www.codecogs.com/eqnedit.php?latex=[0,1]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?[0,1]" title="[0,1]" /></a> is smaller than <a href="https://www.codecogs.com/eqnedit.php?latex=\epsilon" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\epsilon" title="\epsilon" /></a>). Its aim is to explore new possibilities - this is especially important in the initial phases of learning. It becomes less important over next phases of the learning. Therefore, the <a href="https://www.codecogs.com/eqnedit.php?latex=\epsilon" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\epsilon" title="\epsilon" /></a> is being decreased over phases of the learning.

## Learning parameters and more details

- The network has been trained over <a href="https://www.codecogs.com/eqnedit.php?latex=100" target="_blank"><img src="https://latex.codecogs.com/gif.latex?110" title="110" /></a> 'episodes'.

- Each episode consists of around <a href="https://www.codecogs.com/eqnedit.php?latex=5000" target="_blank"><img src="https://latex.codecogs.com/gif.latex?5000" title="5000" /></a> states and their corresponding scores.

- Till the episode <a href="https://www.codecogs.com/eqnedit.php?latex=50." target="_blank"><img src="https://latex.codecogs.com/gif.latex?50." title="50." /></a> network was trained from games starting from the initial game state. In the next <a href="https://www.codecogs.com/eqnedit.php?latex=60" target="_blank"><img src="https://latex.codecogs.com/gif.latex?60" title="60" /></a> episodes the network was trained from such games with the probability <a href="https://www.codecogs.com/eqnedit.php?latex=0.3" target="_blank"><img src="https://latex.codecogs.com/gif.latex?0.3" title="0.3" /></a> and from games starting from some random non-initial state with the probability <a href="https://www.codecogs.com/eqnedit.php?latex=0.7" target="_blank"><img src="https://latex.codecogs.com/gif.latex?0.7" title="0.7" /></a>.

- While training, MSE loss and Adam optimizer were used.

- Batch size was equal to <a href="https://www.codecogs.com/eqnedit.php?latex=128" target="_blank"><img src="https://latex.codecogs.com/gif.latex?128" title="128" /></a>.

- Before the first training, all values of the <a href="https://www.codecogs.com/eqnedit.php?latex=V" target="_blank"><img src="https://latex.codecogs.com/gif.latex?V" title="V" /></a> function were equal to <a href="https://www.codecogs.com/eqnedit.php?latex=0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?0" title="0" /></a>.

## Neural network design

Each board representing state <a href="https://www.codecogs.com/eqnedit.php?latex=s" target="_blank"><img src="https://latex.codecogs.com/gif.latex?s" title="s" /></a> is converted to the <a href="https://www.codecogs.com/eqnedit.php?latex=18&space;\times&space;4&space;\times&space;4" target="_blank"><img src="https://latex.codecogs.com/gif.latex?18&space;\times&space;4&space;\times&space;4" title="18 \times 4 \times 4" /></a> binary tensor. If <a href="https://www.codecogs.com/eqnedit.php?latex=2^k" target="_blank"><img src="https://latex.codecogs.com/gif.latex?2^k" title="2^k" /></a> is located in cell <a href="https://www.codecogs.com/eqnedit.php?latex=[r,c]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?[r,c]" title="[r,c]" /></a>, then <a href="https://www.codecogs.com/eqnedit.php?latex=[k,r,c]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?[k,r,c]" title="[k,r,c]" /></a> coordinate of the tensor is equal to <a href="https://www.codecogs.com/eqnedit.php?latex=1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?1" title="1" /></a>. Otherwise, it is equal to <a href="https://www.codecogs.com/eqnedit.php?latex=0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?0" title="0" /></a>.

<img src="/modelGraphVisualization.png"/>

## Search algorithm - how to use this network in the real game 

Once the network is trained, let us use it in the real game.

When we are to move in a given game, the simplest approach is to make a move that leads to the state <a href="https://www.codecogs.com/eqnedit.php?latex=s" target="_blank"><img src="https://latex.codecogs.com/gif.latex?s" title="s" /></a> with the highest <a href="https://www.codecogs.com/eqnedit.php?latex=V(s)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?V(s)" title="V(s)" /></a> value. This approach will be called further **1-Ply Expectimax Search** algorithm. However, we can extend this approach a bit.

Let me introduce the **Switching Ply Expectimax Search** algorithm.

Briefly speaking, the algorithm is intended to find the best move based on going through several plies. Number of plies to search through depends on the *depthsPolicy* table. Intuitively, more exact search should be performed if number of free cells in the board is low - since we are close to the end and we need to play carefully. On the other hand, if the number of free cells is high, we can decrease the number of plies to go through.

It is worth mentioning this approach fits well in the complexity of expectimax calculation. Basically, when the number of free cells is low, number of potential next boards is low as well - so expectimax can be computed quickly (what allows to perform more exact search). On the other hand, when the number of free cells is high, expectimax is computed over high number of next boards - hence we do not want to decrease a number of plies to search in this case.

Let me present the pseudocode of this algorithm. In the code I called it *expectiDepthMax*.

```
#Assume depthsPolicy table is given. depthsPolicy[i] means number of plies 
#we need to go through for a board with i free cells.
def expectiDepthMax(state, currDepth, finalDepth):
    if currDepth >= finalDepth:
        for move i in possible moves to do:
        choose move i with the highest r_i + gamma*V(s_i)
        #(r_i is a reward received after the move i is performed 
        #and s_i is a state after the move i is performed).
        
        return this move and corresponding r_i + gamma*V(s_i) value.
        
    else:
        for move i in possible moves to do:
        choose move i with the highest r_i + gamma*EV(s_i), 
        where EV(s_i) is an expected value of scores taken over 
        expectiDepthMax(s_i, Net, gamma, currDepth+1, depthsPolicy[freeCells_i]) values.
        #(r_i is a reward received after the move i is performed 
        #and s_i is a state after the move i is performed).
        #freeCells_i is a number of free cells in the s_i state.
        
        return this move and corresponding r_i + gamma*EV(s_i) value.
```

Have a look that the limit of recursions is dynamically updated.

## Tests

The network has been tested twice - using 1-Ply Expectimax Search algorithm and using Switching Ply Expectimax Search algorithm. Results can be found in the [Scores1Ply](https://github.com/BG1992/nn2048/blob/master/Scores1Ply.txt) and [ScoresSwitchingPly](https://github.com/BG1992/nn2048/blob/master/ScoresSwitchingPly.txt) files.

As we can notice, in the 1-Ply Expectimax Search approach, our network achieved average score <a href="https://www.codecogs.com/eqnedit.php?latex=15090.64" target="_blank"><img src="https://latex.codecogs.com/gif.latex?15090.64" title="15090.64" /></a>. Max tile appearing most often is <a href="https://www.codecogs.com/eqnedit.php?latex=1024" target="_blank"><img src="https://latex.codecogs.com/gif.latex?1024" title="1024" /></a> (68/100 times).

In the Switching Ply Expectimax Search algorithm, results are better. As a *depthsPolicy* I used the following table:
```
[3,3,2,2] + [1]*12 #3 plies for 0-1 free cells, 2 plies for 2-3 free cells and 1 ply for remaining cases.
```

Average score achieved by the network is equal to <a href="https://www.codecogs.com/eqnedit.php?latex=30107.56" target="_blank"><img src="https://latex.codecogs.com/gif.latex?30107.56" title="30107.56" /></a>. Max tile appearing the most often is <a href="https://www.codecogs.com/eqnedit.php?latex=2048" target="_blank"><img src="https://latex.codecogs.com/gif.latex?2048" title="2048" /></a> (71/100 times).

For both approaches, I used samples of 100 games. That's not a huge number (the second approach requires some time to run.. and also games are longer..), however I believe it is sufficient to illustrate capabilities of both approaches (and the network we trained).

## Prerequisites

I was using:
- PyTorch 1.4.0
- NumPy 1.18.1 
