import pandas as pd
import os

# path variables
game_url = "chrome://dino"
chrome_driver_path = "../chromedriver"
loss_file_path = "./objects/loss_df.csv"
actions_file_path = "./objects/actions_df.csv"
q_value_file_path = "./objects/q_values.csv"
scores_file_path = "./objects/scores_df.csv"

# scripts
# create id for canvas for faster selection from DOM
init_script = "document.getElementsByClassName('runner-canvas')[0].id = 'runner-canvas'"

# get image from canvas
getbase64Script = "canvasRunner = document.getElementById('runner-canvas'); \
return canvasRunner.toDataURL().substring(22)"


# game parameters
ACTIONS = 2  # possible actions: jump, do nothing
GAMMA = 0.99  # decay rate of past observations original 0.99
OBSERVATION = 100.0  # timesteps to observe before training
EXPLORE = 100000  # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001  # final value of epsilon
INITIAL_EPSILON = 0.1  # starting value of epsilon
REPLAY_MEMORY = 50000  # number of previous transitions to remember
BATCH = 16  # size of minibatch
FRAME_PER_ACTION = 1
LEARNING_RATE = 1e-4
img_rows, img_cols = 80, 80
img_channels = 4  # We stack 4 frames


# Intialize log structures from file if exists else create new
loss_df = (
    pd.read_csv(loss_file_path)
    if os.path.isfile(loss_file_path)
    else pd.DataFrame(columns=["loss"])
)
scores_df = (
    pd.read_csv(scores_file_path)
    if os.path.isfile(loss_file_path)
    else pd.DataFrame(columns=["scores"])
)
actions_df = (
    pd.read_csv(actions_file_path)
    if os.path.isfile(actions_file_path)
    else pd.DataFrame(columns=["actions"])
)
q_values_df = (
    pd.read_csv(actions_file_path)
    if os.path.isfile(q_value_file_path)
    else pd.DataFrame(columns=["qvalues"])
)

