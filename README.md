# README
This repo contains my solution for the third project of the udacity Deep Reinforcement Learning Nanodegree.
The code and implementation is based on my solution and only microscopically adapted to a multi-agent task.


# Short Description of the environment

The environment is a slightly modified Unity ML Tennis environment [(link)](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher). It consists of two collaborating tennis playing agents, whose goal it is to keep the ball in the game as long as possible. Each agent scores  

## Observation Space

The observation space is local to each agent and consists of a 8 * 3 vector of continuous observations. 

## Action Space

The action space of each agent is a continuous 2 dimensional continuos vector that describes the two degrees of freedom of the racket.

## Solution criterion

The environment is considered to be solved, if the agent scores on average +30 on 100 consecutive episodes. 


# Files in the Repository

The files of interest in the repo are: 

- `Tennis.ipynb`: Notebook used to control and train the agent. The entry point to the code. 
- `DDPG_agent.py`: Create an Agent class that interacts with and learns from the environment 
- `models.py`: Contains the two networks for "actor" and "critic". values 
- `checkpoint_actor.pth`: Saved weights for actor network
- `checkpoint_critic.pth`: Saved weights for the critic network
- `report.pdf`: Project report including a short introduction of the DDGP algorithm used, the hyperparameters and a short discussion of the results. 


# Getting Started, Installation and Dependencies

To run this code, you have to install the dependencies and the environment.

## Dependencies  

The code requires Python 3. The  necessary dependencies can be found in `./python/requirements.txt` 
Batch installation is done like so: 
```
pip install ./python/requirements.txt
``` 
## Environment

The necessasry Unity environment can be downloaded from the following locations:

- Linux: [(link)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
- Mac OSX: [(link)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
- Windows (64-bit): [(link)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)



## Execution

Once you have cloned this repository and istalled the dependencies and the environment, the main entry point for execution is the Jupyter-notebook `Tennis.ipynb`.


# Reference and Credits

The implementation is based on my code from the the second project for the udacity DRL nanodegree. That code again was based heavily on the DDPG example given in the udacity DRL repo and only minimally adopted to the new environment. The complete list of references can be found in the project report.
