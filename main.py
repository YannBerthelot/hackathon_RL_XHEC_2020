# General
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# RL
import tensorforce
from tensorforce import Agent, Environment

# Gym
import gym

# utils
from utils import clip_states, info_extractor, display_info, save_progress, save_graph

# from PyPowerGadget import PowerMeter

# ower_meter = PowerMeter(project_name="RL hackathon XHEC")


class SpaceXRL:
    def __init__(self):
        self.landed_ticks = 0
        self.number_of_landings = 0
        self.fraction_good_landings = 0
        self.cracked = False
        self.env = None
        self.level_number = 0

    def run(
        self,
        n_episodes,
        level=0,
        load=False,
        save_frequency=10,
        threshold=70,
        test=True,
        verbose=False,
        visualize=True,
        save_video=True,
        visualize_directory=None,
    ):
        """
        ### NO WORK NEEDED ###
        You can look at the structure but you do not need to modify it.
        You can print whatever you feel necessary.
        ######################

        Train agent for n_episodes

        if test == True the agent will not explore and will only exploit.
        if verbose == True the function will print more information during the training (this will messup the progress bar)
        if visualize == True the function will display an animation of the rocket landing for every episode.
        This is at the expense of speed though. If false it will only show it every n episodes.
        """
        self.level_number = level
        self.env = tensorforce.environments.OpenAIGym(
            "RocketLander-v0", level_number=level
        )

        if n_episodes < save_frequency:
            str_error = f"n_episodes<save frequency, the model won't be able to save, set n_episodes to a value >={save_frenquency}"
            raise (ValueError(str_error))

        agent = self.create_agent(self.env, n_episodes, save_frequency, load)

        # Loop over episodes
        reward_list_episodes = []
        reward_list = []
        landing_fraction_list = []
        tqdm_bar = tqdm(range(1, n_episodes + 1))
        self.number_of_landings = 0

        for i in tqdm_bar:
            self.fraction_good_landings = self.number_of_landings * 100 / i
            if i > 1:
                tqdm_bar.set_description(
                    "Episode %d/%d reward: %d (max:%d, min:%d, mean:%d, std:%d), successful landings:%d(%d%%)"
                    % (
                        i,
                        n_episodes,
                        np.round(np.sum(reward_list), 3),
                        np.max(reward_list_episodes),
                        np.min(reward_list_episodes),
                        np.round(np.mean(reward_list_episodes), 3),
                        np.round(np.std(reward_list_episodes), 3),
                        self.number_of_landings,
                        np.round(self.fraction_good_landings, 3),
                    )
                )

            if i % save_frequency == 0:
                if save_video:
                    self.env = tensorforce.environments.OpenAIGym(
                        "RocketLander-v0",
                        visualize=True,
                        visualize_directory=visualize_directory,
                        level_number=level,
                    )
                else:
                    self.env = tensorforce.environments.OpenAIGym(
                        "RocketLander-v0", visualize=True, level_number=level
                    )

                reward_list = self.episode(self.env, i, agent, test=True, verbose=False)
            else:
                self.env = tensorforce.environments.OpenAIGym(
                    "RocketLander-v0", level_number=level
                )
                reward_list = self.episode(
                    self.env, i, agent, test=False, verbose=False
                )
            reward_list_episodes.append(np.sum(reward_list))
            landing_fraction_list.append(self.fraction_good_landings)
            if self.env.environment.landed_ticks > 59:
                self.number_of_landings += 1
            if (self.fraction_good_landings > threshold) and (i > 50):
                self.cracked = True
                break

        # Show Sum of reward over 1 episode vs number of episodes graph
        reward_list_episodes = save_progress(
            load, reward_list_episodes, "reward_list_episodes.txt", level=level
        )
        landing_fraction_list = save_progress(
            load, landing_fraction_list, "landing_fraction.txt", level=level
        )
        save_graph(
            reward_list_episodes,
            "Sum of reward over 1 episode vs number of episodes",
            "rewards_vs_episodes.png",
            rolling=True,
            level=level,
        )
        save_graph(
            landing_fraction_list,
            "Landing fraction vs number of episodes",
            "landing_fraction_vs_episodes.png",
            level=level,
        )

    def create_agent(
        self,
        env,
        n_episodes,
        save_frenquency,
        load=False,
    ):
        ########### WORK NEEDED ###########
        ### You need to tweak the Agent ###
        ###################################
        """
        Agent definition. Tweak the Agent's parameters to your convenience

        Use any agent from tensorforce and refer to the documentation for the available hyperparameters :
        -Vanilla Policy Gradient : https://tensorforce.readthedocs.io/en/latest/agents/vpg.html
        -Proximal Policy Optimization : https://tensorforce.readthedocs.io/en/latest/agents/ppo.html
        -Trust-Region Policy Optimization : https://tensorforce.readthedocs.io/en/latest/agents/trpo.html
        -Deterministic Policy Gradient : https://tensorforce.readthedocs.io/en/latest/agents/dpg.html
        -Deep Q-Network : https://tensorforce.readthedocs.io/en/latest/agents/dqn.html
        -Double DQN : https://tensorforce.readthedocs.io/en/latest/agents/double_dqn.html
        -Dueling DQN : https://tensorforce.readthedocs.io/en/latest/agents/dueling_dqn.html
        -Actor-Critic : https://tensorforce.readthedocs.io/en/latest/agents/ac.html
        -Advantage Actor-Critic : https://tensorforce.readthedocs.io/en/latest/agents/a2c.html

        For the network parameters :
        https://tensorforce.readthedocs.io/en/latest/modules/networks.html


        """
        ##### Agent definition ########
        if not (load):
            agent = Agent.create(
                "ppo",
                environment=env,
                # Automatically configured network
                network={"type": "auto", "rnn": False},
                # Optimization,
                batch_size=64,
                use_beta_distribution=True,
                memory="minimum",
                update_frequency=8,
                learning_rate=0.0001,
                multi_step=5,
                subsampling_fraction=0.90,
                likelihood_ratio_clipping=0.10,
                discount=0.99,
                predict_terminal_values=False,
                baseline={"type": "auto", "rnn": False},
                baseline_optimizer={
                    "optimizer": "adam",
                    "learning_rate": 0.003670157218888348,
                    "multi_step": 20,
                },
                state_preprocessing="linear_normalization",
                reward_preprocessing=None,
                exploration=dict(
                    type="exponential",
                    unit="episodes",
                    num_steps=1000,
                    initial_value=0.05,
                    decay_rate=0.01,
                ),
                variable_noise=0.0,
                l2_regularization=0.0,
                entropy_regularization=0.0001,
                parallel_interactions=1,
                config=None,
                saver=dict(
                    directory="data/checkpoints",
                    frequency=10,  # save checkpoint every 10 updates
                ),
                summarizer=None,
                recorder=None,
                optimization_steps=None,
                estimate_terminal=None,
                critic_network=None,
                baseline_network=None,
                critic_optimizer=None,
            )

        else:
            agent = Agent.load(directory="data/checkpoints")
        return agent

    def episode(self, env, episode_number, agent, test=False, verbose=False):
        """
        ### NO WORK NEEDED ###
        You can look at the structure but you do not need to modify it.
        You can print whatever you feel necessary.
        ######################

        This function computes an episode in the given environment with the given agent
        episode_number is just for display purpose

        if test == True the agent will not explore and will only exploit.
        if verbose == True the function will print more information during the training (this will messup the progress bar)


        """
        # Initialize episode
        episode_length = 0
        states = env.reset()
        internals = agent.initial_internals()
        terminal = False
        timestep = 0
        reward_list = []
        while not terminal:
            timestep += 1
            # Run episode
            episode_length += 1
            # clip states to be between -1 and 1
            states = clip_states(states)
            if test:
                actions, internals = agent.act(
                    states=states, internals=internals, independent=True
                )
                # actions = agent.act(states=states, independent=False)
            else:
                actions = agent.act(states=states, independent=False)
            if (timestep % 10 == 0) and verbose:
                print("actions", actions)

            states, terminal, reward = env.execute(actions=actions)
            reward = self.reward_function(states, timestep)
            reward_list.append(reward)
            if not (test):
                agent.observe(terminal=terminal, reward=reward)
            # if test:
            #     agent.observe(terminal=terminal, reward=reward)
        return reward_list

    def reward_function(
        self, states_list, timestep=0, print_states=True, print_additionnal_info=True
    ):
        ########## WORK NEEDED #############
        ### You need to shape the reward ###
        ####################################
        """
        Available information:
        x : horizontal position
        y : vertical position
        angle : angle relative to the vertical (negative = right, positive = left)
        first_leg_contact : Left leg touches ground
        second_leg_contact : Right leg touches ground
        throttle : Throttle intensity
        gimbal : Gimbal angle relative to the rocket axis
        velocity_x : horizontal velocity (negative : going Left, positive : going Right)
        velocity_y : vertical velocity (negative : going Down, positive : going Up)
        angular_velocity : angular velocity (negative : turning anti-clockwise, positive : turning clockwise)
        distance : distance from the center of the ship
        velocity : norm of the velocity vector (velocity_x,velocity_y)
        landed : both legs touching the ground
        landed_full : both legs touching ground for a second (60frames)
        states : dictionnary containing all variables in the state vector. For display purpose
        additionnal_information : dictionnary containing additionnal information. For display purpose

        **Hints**
        Be careful with the sign of the different variables

        Go on and shape the reward !
        """
        # states information extraction
        (
            x,
            y,
            angle,
            first_leg_contact,
            second_leg_contact,
            throttle,
            gimbal,
            velocity_x,
            velocity_y,
            angular_velocity,
            distance,
            velocity,
            landed,
            landed_full,
            states,
            additionnal_information,
        ) = info_extractor(states_list, self.env)

        ######## REWARD SHAPING ###########
        # reward definition (per timestep) : You have to fill it !
        reward = min(0, max(-(angle * angular_velocity), -abs(x) - abs(distance)))
        reward = reward / 10
        if self.env.environment.landed_ticks > 59:
            reward += 150

        display_info(states, additionnal_information, reward, timestep, verbose=False)

        return reward


if __name__ == "__main__":
    """
    Run the training over n_episodes
    n_episode_per_batch determines the frequency at which the results are saved and added to the graph
    and at which frequency the agent video is displayed
    if load = False, train from scratch else train from where you left
    """

    environment = SpaceXRL()
    level = 1

    n_episodes = 1000
    n_episode_per_batch = 100
    # Switch it to True if you want to restart from your previous agent
    load = True

    n_batch = (n_episodes // n_episode_per_batch) + 1
    tqdm_bar = tqdm(range(1, n_batch + 1))
    for i in tqdm_bar:
        if not (os.path.exists("vids")):
            os.mkdir("vids")
        visualize_directory = "vids/" + str(i)
        if not (os.path.exists(visualize_directory)):
            os.mkdir(visualize_directory)
        tqdm_bar.set_description(f"Batch {i}/{n_batch} level {level}")
        environment.run(
            n_episode_per_batch,
            level=level,
            save_frequency=n_episode_per_batch,
            load=load,
            visualize=False,
            visualize_directory=visualize_directory,
            save_video=True,
            threshold=100,
        )
        load = True
        if environment.cracked:
            print("")
            print(
                "level %d cleared : %d%% good landings"
                % (level, environment.fraction_good_landings)
            )
            level += 0  # change to one if you want to automatically change level when the current level is beaten
            environment.cracked = False
            if level > 3:
                break
    if environment.cracked:
        print(
            "level %d cleared : %d%% good landings"
            % (level, environment.fraction_good_landings)
        )
