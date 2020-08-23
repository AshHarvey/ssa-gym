# ssa-gym

This is a repository of an OpenAI Gym environment for tasking Space Situational Awareness Sensors and some associated agents.

Original repository author : Maj. Ashton Harvey, Devayani Pawar

The repository includes:
* [Data Source and Transformation functions](envs/transformations.py)
* [Source code of SSA tasker and Plot results](envs/ssa_tasker_simple_2.py)
* [Implementation of dynamic functions](envs/dynamics.py)
* [Implementation of reward function](envs/reward.py)
* [Implementation of heuristic agents](envs/agents.py)
* [filter.py](envs/filter.py) is an extract of Roger Labbe's FilterPy (https://filterpy.readthedocs.io/) 
* [Test Cases](tests.py)
* [Implementation of Result & plot functions](envs/results.py)

## Requirements
Python 3.6, and other common packages listed in `requirements.txt`.

## Installation
1. Clone this repository
2. Install dependencies
   ```bash
   pip3 install -r requirements.txt
   ```
3. [optional] Run setup from the repository root directory
    ```bash
    python3 setup.py install
    ``` 
## Getting Started:


[Register's environment with Gym](envs/__init__.py )


#### Key Concepts used:
- [Deep Reinforcement Learning](https://spinningup.openai.com/en/latest/)
- Orbital Dynamics
- [Kalman Filter for uncertain information](https://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/)
  - [Unscented Kalman Filter](https://towardsdatascience.com/the-unscented-kalman-filter-anything-ekf-can-do-i-can-do-it-better-ce7c773cf88d)

#### Libraries used
- Reinforcement Learning in python: https://gym.openai.com/docs/
- FilterPy - Python library that implements a number of Bayesian filters, most notably Kalman filters: https://filterpy.readthedocs.io/en/latest/
- Numba - An open source JIT compiler that translates a subset of Python and NumPy code into fast machine code: http://numba.pydata.org/
- Astropy - A Community Python Library for Astronomy: https://www.astropy.org/
- ERFA (Essential Routines for Fundamental Astronomy) is a C library containing key algorithms for astronomy, and is based on the SOFA library published by the International Astronomical Union (IAU) https://github.com/liberfa/erfa
- Poliastro - An open source collection of Python subroutines for solving problems in Astrodynamics and Orbital Mechanics: 
https://docs.poliastro.space/en/stable/about.html 
- RLlib - RLlib is an open-source library for reinforcement learning that offers both high scalability and a unified API for a variety of applications : https://docs.ray.io/en/master/rllib.html
     - *Bonus* - RLlib is a library built on top of Ray core: https://docs.ray.io/en/master/ray-overview/index.html
    

## Viz




## Citation
```
@misc{ssa-gym_2020,
  title={An OpenAI Gym environment for tasking Space Situational Awareness Sensors and some associated agents.
},
  author={Maj. Ashton Harvey, Devayani Pawar},
  year={2020},
  publisher={Github},
  journal={GitHub repository},
  howpublished={\url{https://github.com/AshHarvey/ssa-gym}},
}
```



