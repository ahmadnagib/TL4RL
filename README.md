# Enhancing Policy Transfer to Accelrate Reinforcement Learning-based RAN Slicing (TL4RL)

This project is part of my PhD thesis. I use the problem of resource allocation in Radio Access Network (RAN) slicing to evaluate the potential of using transfer learning to accelerate DRL-based RRM. We mainly focus on acceleration when a mobile network operator (MNO) changes the priorities of fulfilling the service level agreements (SLAs) of the available slices. This can be applied by changing the weights of the reward function. 

I propose a distance metric and two supervised learning approaches which choose the best model to load whenever an MNO changes the priorities of fulfilling the SLAs of the available slices. The related publications are listed in a separate section below and will be continuosly updated.

In one part of this work, I resued and modified the code of q-learning and sarsa algorithms from the [reinforcement learning GitHub repository of Denny Britz](https://github.com/dennybritz/reinforcement-learning).


# Documentation

The information included in this documentation is as follows:

- [System Setup](#system-setup)
- [Quick Start](#quick-start)
- [What is included](#what-is-included)
- [Related Publications](#related-publications)
- [References](#references)
- [License](#license)


## System Setup

Make sure that you have Jupyter Notebook and that the following Python packages are installed:
- matplotlib
- numpy
- pandas
- gym
- scipy
- math
- os
- itertools
- sys

## Quick start

Go to the ```rl_agents_training``` folder to run the training of expert base models, and non-accelerated and accelerated learner agents.
Go to the ```visualization``` folder to check some of the graphs generated from the training data.

## What is included

Within the download you will find the following files:

```
TL4RL-master/
├── lib/
    ├── agents/
        ├── qlearning.py
        ├── sarsa.py
    ├── envs/
        ├── slicing_env.py
    ├── plotting.py
    ├── utils.py
├── rl_agents_training/
    ├── train_base_models.ipynb
    ├── train_learner_agents_accelerated.ipynb
    ├── train_learner_agents_non_accelerated.ipynb
    ├── .ipynb
    ├── .ipynb
├── saved_models/
├── visualization/
    ├── visualization.ipynb
    ├── error_data.pkl
├── LICENSE
├── README.md
```


## Related Publications

+ [Toward Safe and Accelerated Deep Reinforcement Learning for Next-Generation Wireless Networks](https://ieeexplore.ieee.org/abstract/document/9903386)
+ [Transfer Learning-Based Accelerated Deep Reinforcement Learning for 5G RAN Slicing](https://ieeexplore.ieee.org/abstract/document/9524965)


## References

+ [Reinforcement learning GitHub repository of Denny Britz](https://github.com/dennybritz/reinforcement-learning)


## License

TL4RL is Copyright © 2022 Ahmad Nagib. It is free software, and may be redistributed under the terms specified in the [LICENSE](/LICENSE) file.
A human-readable summary of (and not a substitute for) the license is available at https://creativecommons.org/licenses/by-nc-sa/4.0/
