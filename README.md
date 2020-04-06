# ssa-gym
This is a repository of an OpenAI Gym environment for tasking Space Situational Awareness Sensors and some associated agents.

File Descriptions:
filterjit.py is an extract of Roger Labbe's FilterPy (https://filterpy.readthedocs.io/) which we are currently refactoring for greater speed using Numba. 

SSA_Agents_v2.py runs several different agents against the environment and compares their performance. 

training_a2c.py trains the Stable Baselines' A2C policy against the SSA environment, https://stable-baselines.readthedocs.io/.

envs/__init__.py register's environment with Gym

envs/ssa_tasker_simple_1.py contains the Class and supporting functions for MAGE v1 conforming to OpenAI Gym's standards
