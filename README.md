# Evolutionary Reinforcement Learning with Weight-Freezing and Markov Blanket-Based Dimensionality Reduction.

This repository contains the complete, executable source code that can be used to reproduce all experiments reported in the manuscript.

**PlatEMO (MATLAB) Component**

All evolutionary optimization algorithms used in this work (both proposed methods and baseline methods) are implemented inside the ```PlatEMO/ ``` directory, following standard PlatEMO conventions. 


** To run xSTMBA and the evolutionary algorithm baseline**

You need to perform the following  to install the required dependencies. 

Requires Python 3.11  and the Causal Learner: A Toolbox for Causal Structure and Markov Blanket Learning for MB Learning (https://github.com/z-dragonl/Causal-Learner).<br />
Clone this repository and install the packages specified in requirements.txt <br />
```
git clone https://github.com/oladayosolomon/xSTMBGA/
cd xSTMBGA
pip install -r requirements.txt
```
For the reacher environment, you'll need to install pybullet-gym from https://github.com/benelot/pybullet-gym<br />
```
git clone https://github.com/benelot/pybullet-gym.git
cd pybullet-gym
pip install -e.
```
You should download the causal learner toolbox and add its path to MATLAB <br />

Path related information<br />

```
pyenv("Version",'C:\Users\ecis\anaconda3\envs\RL_Bench\python.exe') 

```
Note: This code is based on PlatEMO 4.2 framework. Documentation available at (https://github.com/BIMK/PlatEMO).


All output files from the experiments conducted can be downloaded at https://drive.google.com/drive/folders/1OfrcP-KRnyWfjlM-HTHrF3d40mjRmF6X?usp=drive_link


**Python (DRL Baselines and Hybrid ERL References)**

The python/ directory contains baseline implementations used exclusively for comparison with traditional deep reinforcement learning (DRL) and hybrid Evolutionary–DRL methods.

** A2C, PPO, PDERL **
Requires Python 3.11 and the packages specified in requirements.txt.<br />

To run A2C  
```
python a2c_experiment.py
```

To run PPO 
```
python ppo_experiment.py
```

The official implementation of PDERL, available from https://github.com/crisbodnar/pderl.git, is used for PDERL.
 
