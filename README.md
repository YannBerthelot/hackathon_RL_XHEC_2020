This repository contains the materials necessary for the ending Hackathon for the **2020 RL@XHEC course**.

The goal is for the students to define the rewards and train&tweak the agent to achieve the best score possible.

**How-to use it:**
1. Clone this repository : `git clone --recursive https://gitlab.com/the_insighters/x-hec/hackathon-rlxhec.git`
2. Enter the repo folder `cd hackathon-rlxhec`
3. (Optional if already done) : install virtualenv `pip install virtualenv`
4. Create a virtualenv : `virtualenv "hackathon_rl_env"`
5. Activate the virtualenv : `source hackathon_rl_env/bin/activate` for MAC.
6. Install the libraries : `pip install -r requirements.txt`
7. Move to the custom environment folder : `cd gym_rocketLander`
8. Install the custom environment : `pip install -e .`
9. Go back to main folder : `cd ..`
10. You can now run main.py : `python main.py`
11. Modify main.py to tweak the agent and reward.
12. Good luck !

**To-do list**
- [x] Transform the Rocker lander from https://github.com/EmbersArc/gym_rocketLander/blob/master/gym/envs/box2d/rocket_lander.py into an exploitable environment.
- [x] Customize the environment for the reward function to be definable from main.py
- [x] Create a nice training&testing interface
- [X] Add a save Agent functionnality
- [X] Solve the task with PPO)
- [x] Add video recording (instead of just being able to watch the episode at the time it is done)
- [X] Add a cue when the task is "solved"
- [X] Add landed and landed full to available variables
- [x] Add difficulty levels (done but not very useful, maybe a score system will be beter)
- [ ] Add variables description
- [ ] Solve the task with simpler agents
- [ ] Tweak the difficulty for a nice experience
- [ ] Test the exercise with different people to measure difficulty and time to beat

**Architecture:**
- main.py : This is where the student will define reward function and agent and will be able to train and see the results
- utils.py : Scripts to help the students. non directly related to RL.
- gym_rocketLander (submodule) : Custom gym environment for SpaceX Rocket Lander (original work from EmbersArc modified to tweak difficulty)
- requirements.txt : Specifies the needed libraries
- .gitignore : Specifies which files to ignore when commiting.
- data (folder):  stores the data needed for the plots and agent. It is of no use for the user
- figs (folder): stores the graphs (rewards over the episodes)
- vids (folder): stores the vids of the agent's execution for each batch




**Disclaimer:**
The environment is based on https://github.com/EmbersArc/gym_rocketLander



