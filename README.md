# ssa-gym
This is a repository of an OpenAI Gym environment for tasking Space Situational Awareness Sensors and some associated agents.

Original repository author : Maj. Ashton Harvey

The repository includes:
* [Data Source and Transformation functions](envs/transformations.py)
* [Source code of SSA tasker](envs/ssa_tasker_simple_2.py)
* [filter.py](envs/filter.py) is an extract of Roger Labbe's FilterPy (https://filterpy.readthedocs.io/) 
* [Test Cases](tests.py)
* [Results and Plots](envs/tests.py)

## Requirements
Python 3.6, and other common packages listed in `requirements.txt`.

## Installation
1. Clone this repository
2. Install dependencies
   ```bash
   pip3 install -r requirements.txt
   ```
3. Run setup from the repository root directory
    ```bash
    python3 setup.py install
    ``` 
## Getting Started:


## Viz

## References:

## Citation
```
@misc{ssa-gym_2020,
  title={An OpenAI Gym environment for tasking Space Situational Awareness Sensors and some associated agents.
},
  author={Maj. Ashton Harvey},
  year={2020},
  publisher={Github},
  journal={GitHub repository},
  howpublished={\url{https://github.com/AshHarvey/ssa-gym}},
}
```



File Descriptions:
filterjit.py is an extract of Roger Labbe's FilterPy (https://filterpy.readthedocs.io/) which we are currently refactoring for greater speed using Numba. 

SSA_Agents_v2.py runs several different agents against the environment and compares their performance. 

training_a2c.py trains the Stable Baselines' A2C policy against the SSA environment, https://stable-baselines.readthedocs.io/.

envs/__init__.py register's environment with Gym

envs/ssa_tasker_simple_1.py contains the Class and supporting functions for MAGE v1 conforming to OpenAI Gym's standards
