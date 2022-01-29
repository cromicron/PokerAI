<h1>Creating and Training an AI to Play No Limit Holdem Heads-Up Poker through Deep Reinforcement Self-Play Learning with Tensorflow/Keras </h1>
This project came about as a challenge to a (semi-)professional poker player and dear friend of mine, that I would program an AI, that would beat him in poker. I am a Data-Scientist in the making and one purpose of this project is to push my limits in python and machine-learning. <br>
I am building the poker game from scratch as an environment an agent can later use to train in. I am also building a user-interface for my friend (and myself) to play against the AI. The actual model will be a neural network generated with tensorflow-keras. It is going to learn through deep-reinforcement learning by playing against itself. I start with a deterministic approach (deep-q learning). Taking into account that the environment in poker is probabilistic, I will later on try to implement a stochastic approach. State-of-the-art poker theory suggests that probabilistic strategies are superior to deterministic strategies. I plan to also host a video of my friend playing against the AI. <h2> PokerAI Short </h2>
In this folder you can find all the python files necessary to run all the machine learning code. There are a few Jupyter Notebook that are hopefully self-explanatory. The actual deep-learning process can be found in the Jupyter Notebook called "PokerAI10BB.ipynb". The trained models folder contains the current trained neural network.
<h2> Poker Game </h2>
The files necessary to run the poker game are  
 <ul>
  <li>Agent2: Class that interacts with the poker environment</li>
  <li>PokerGame: Poker Environment to train on</li>
  <li>Poker Hand Strengths: Entails the compare function which evaluates two poker hands on the river</li>
  <li>StrengthEvaluator2: Evaluator Class which can evaluate the hand strengths on all boards and return the players winning and losing probabilities against random hands as well as winning probabilities and standard deviations of avarage hands</li>
  <li>StrengthVillainFlop - StrangthVillainRiver: Jupyter Notebooks on which the strength Evaluator is trained </li>
</ul> 

