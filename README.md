# Improving long-horizon decision making with hierarchical goal-conditioned planning 

## Final Project for UT Reinforcement Learning course (Fall 2019)  
Kai-Chi Huang, Wei-Jen Ko


### Youtube video: <https://youtu.be/c_G16ep3f-I>
### Report PDF: <https://github.com/kevin00036/ut-rl2019/raw/master/RL_Final_Project.pdf>

--------------

Dependencies:

- Python 3.7+
- gym (0.15.4)
- numpy (1.17.4)
- torch (1.3.1 with CUDA)
- A machine with a CUDA-compatible GPU

--------------

To run codes:

Simply run 
> python3 test.py

To change environments, change (or uncomment) the environment `env` on Line 17 in test.py. Note that we currently support discrete-action environments currently.

To switch the TD3 optimization (mitigating maximization bias), toggle the `use_td3` variable on Line 24.

To switch between Goal-conditioned RL and standard RL, uncomment the corresponding agent on Line 27-28.

To change the maximum environment steps, change the `max_steps` variable on Line 40.


The execution log will be save at `<project base>/logs/<algorithm_name>/xxxxxxxxxx_yyyyy.json`

