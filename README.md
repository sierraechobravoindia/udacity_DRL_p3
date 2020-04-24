# README
This repo contains my solution for the third project of the udacity Deep Reinforcement Learning Nanodegree.
The code and implementation is based on my solution for the second project and only microscopically adapted to a multi-agent task. 

The solution presented here is not a true multi-agent algorithm. The environment contains two agents with identical observation space and reward function and no need to communicate for collaboration, so the environment can be solved with only one agent without the need for a common Q-function. The experience tuples for both agents are collected in the replay buffer and used for training. 


# Short Description of the environment

The environment is a slightly modified Unity ML Tennis environment [(link)](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis). It consists of two collaborating tennis playing agents, whose goal it is to keep the ball in the game as long as possible. If the agent gets the ball over the net, it receives a reward of +0.1. If the agent drops the ball or hits it out, it receives a reward of -0.01.

## Observation Space

The observation space is local to each agent and consists of a 8 * 3 vector (3 frames of 8 obervables) of continuous observations. 

## Action Space

The action space of each agent is a continuous 2 dimensional vector that describes the two degrees of freedom of the racket.

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

To run this code, you have to do three things: Download or clone this reop, install the dependencies and download and copy the environment to the right folder. In general it is advisable to create and run the code in a virtual Python environment.

## Cloning this repo

You can either clone this repo from the command line or download and uncompress the content of this repo by using the green "Clone or Download and "Download ZIP" button. 

## Dependencies  

The code requires Python 3.6. The  necessary dependencies can be found in `./python/requirements.txt` 
Batch installation is done like so: 
```
pip install ./python/requirements.txt
``` 

This is also done in the first cell of the notebook `Tennis.ipynb` and has to be done only once.


## Environment

The necessary Unity environment can be downloaded from the following locations:

  - Linux: [link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
  - Mac OSX: [link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
  - Windows (64-bit): [link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

Download the appropriate version for your operating system, then unzip and copy to a location of your choice and change the path in the second cell of the Jupyter-notebook `Tennis.ipynb` to your location of the files:

```
env = UnityEnvironment(file_name="/data/Tennis_Linux_NoVis/Tennis")
``` 


## Execution

Once you have cloned this repository and istalled the dependencies and the environment, the main entry point for execution is the Jupyter-notebook `Tennis.ipynb`.


# Reference and Credits

The implementation is based on my code from the the second project for the udacity DRL nanodegree. That code again was based heavily on the DDPG example given in the udacity DRL repo and only minimally adopted to the new environment. The complete list of references can be found in the project report.
