import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import requests


def clip_states(states):
    """
    Clip states between -1 and 1 except for the last 3 infinite values (velocity x y and angular velocity)
    """
    for j in range(len(states) - 2):
        states[j] = np.clip(states[j], -1, 1)
    return states


def info_extractor(states_list, env):
    """
    Extracts information from the state vector into separate variables for usage and 2 dictionnaries for display
    """
    x = states_list[0]
    y = states_list[1] + 1
    angle = states_list[2]
    first_leg_contact = states_list[3] > 0
    second_leg_contact = states_list[4] > 0
    throttle = states_list[5]
    gimbal = states_list[6]
    velocity_x = states_list[7]
    velocity_y = states_list[8]
    angular_velocity = states_list[9]
    distance = np.linalg.norm((x, y))
    velocity = np.linalg.norm((velocity_x, velocity_y))
    landed = (first_leg_contact > 0) and (second_leg_contact > 0) and velocity < 10
    landed_full = env.environment.landed_ticks > 59

    states = {
        "x": x,
        "y": y,
        "angle": angle,
        "first leg ground contact": first_leg_contact,
        "second leg ground contact": second_leg_contact,
        "throttle": throttle,
        "gimbal": gimbal,
        "x velocity": velocity_x,
        "y velocity": velocity_y,
        "angular velocity": angular_velocity,
    }
    # additionnal useful variables
    additionnal_information = {
        "distance": distance,
        "velocity": velocity,
        "landed": landed,
        "landed_full": landed_full,
    }
    return (
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
    )


def display_info(
    states, additionnal_information, reward, timestep, n_timesteps=50, verbose=True
):
    """
    Displays information every n_timesteps, during training
    States information
    Additionnal information
    Reward
    """
    if timestep % n_timesteps == 0:
        if verbose:
            states_display = {key: np.round(states[key], 2) for key in states.keys()}
            print("states", states_display)
            additionnal_information_display = {
                key: np.round(additionnal_information[key], 2)
                if type(additionnal_information[key]) == "float"
                else additionnal_information[key]
                for key in additionnal_information.keys()
            }
            print("additionnal info", additionnal_information_display)
            print("reward value", np.round(reward, 5))


def save_graph(
    reward_list_episodes, title, filename, rolling=False, window_size=10, level=0
):
    if not (os.path.exists("figs")):
        os.mkdir("figs")
    fig, ax = plt.subplots()
    pd.Series(reward_list_episodes).plot(title=title, ax=ax)
    if rolling & (len(reward_list_episodes) > window_size):
        pd.Series(reward_list_episodes).rolling(window_size).mean().plot(ax=ax)
    plt.savefig("figs/" + str(filename))


def save_progress(load, data, filename, level=0):
    if not (os.path.exists("data")):
        os.mkdir("data")
    path = "data/figs/"  # + str(level)
    if not (os.path.exists("data/checkpoints")):
        os.mkdir("data/checkpoints")
    if not (os.path.exists(path)):
        os.mkdir(path)
    file_name = path + str(filename)
    if not (load):
        pickle.dump(data, open(file_name, "wb"))
    else:
        data_prev = pickle.load(open(file_name, "rb"))
        data_prev.extend(data)
        data = data_prev
        pickle.dump(data, open(file_name, "wb"))
    return data


def prep_data_to_send(inputs, GROUP_NAME, DATE):
    dict_data = {
        "id": GROUP_NAME + "__" + str(DATE),
        "group_name": GROUP_NAME,
        "datetime": DATE,
        "info": json.dumps(inputs),
    }
    return dict_data


def send_result(data):

    url = "https://pakmcaujg0.execute-api.eu-west-3.amazonaws.com/post-result"

    payload = json.dumps(data)
    headers = {"Content-Type": "application/json"}
    response = requests.request("POST", url, headers=headers, data=json.dumps(data))

    return print(response.text)