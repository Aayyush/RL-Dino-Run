from module_imports import *

from Game import Game

# from Models import *

# from Models import *

# Build Model.
def build_model():
    if DEBUG:
        print("Building the model.")
    # Build a sequential model.
    model = Sequential()

    # First convolutional layer with
    #   Filter Size: 8 * 8
    #   Strides    : 4 * 4
    #   No. filter : 32
    #   Pool size  : 2 * 2.
    model.add(
        Conv2D(
            32,
            (8, 8),
            padding="same",
            strides=(4, 4),
            input_shape=(img_cols, img_rows, img_channels),
        )
    )  # 80*80*4
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation("relu"))

    # Second convolutional layer with
    #   Filter Size: 4 * 4
    #   Strides    : 2 * 2
    #   No. filter : 64
    #   Pool size  : 2 * 2.
    model.add(Conv2D(64, (4, 4), strides=(2, 2), padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation("relu"))

    # Third convolutional layer with
    #   Filter Size: 3 * 3
    #   Strides    : 1 * 1
    #   No. filter : 64
    #   Pool size  : 2 * 2.
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation("relu"))

    # Flatten the pooling layer.
    model.add(Flatten())

    # Dense layer to generalize the input.
    model.add(Dense(512))
    model.add(Activation("relu"))

    # Output layer.
    model.add(Dense(ACTIONS))
    adam = Adam(lr=LEARNING_RATE)
    model.compile(loss="mse", optimizer=adam)

    # Create the model file if not present.
    if not os.path.isfile(loss_file_path):
        model.save_weights("model.h5")

    if DEBUG:
        print("The model has been built.")

    return model


# Train Model.
def train_network(model, game_state, observe=False):
    # Record the current time.
    last_time = time.time()

    # Store the previous observations in replay memory.
    D = load_obj("D")  # Load from file system.

    # Get the first state by doing nothing.
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1  # 0 => do nothing 1=> jump.

    # Get next step after performing the action.
    x_t, r_0, terminal = game_state.get_state(do_nothing)

    # Stack 4 images to create placeholder input.
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
    s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])  # 1 * 20 * 40 * 4

    initial_state = s_t

    if observe:
        # We keep observing the frames not train.
        OBSERVE = 999999999
        epsilon = FINAL_EPSILON

        if DEBUG:
            print("Now we load weight")

        model.load_weights("model.h5")
        adam = Adam(lr=LEARNING_RATE)
        model.compile(loss="mse", optimizer=adam)
        print("Weight load successfully")
    # We go to training mode.
    else:
        OBSERVE = OBSERVATION
        epsilon = load_obj("epsilon")
        model.load_weights("model.h5")
        adam = Adam(lr=LEARNING_RATE)
        model.compile(loss="mse", optimizer=adam)

    t = load_obj("time")  # Resume from the previous time step stored in file system
    while True:  # Endless running

        loss = 0
        Q_sa = 0
        action_index = 0
        r_t = 0  # reward at 4
        a_t = np.zeros([ACTIONS])  # action at t

        # choose an action epsilon greedy
        if t % FRAME_PER_ACTION == 0:  # parameter to skip frames for actions
            if random.random() <= epsilon:  # randomly explore an action
                print("----------Random Action----------")
                action_index = random.randrange(ACTIONS)
                a_t[action_index] = 1
            else:  # predict the output
                q = model.predict(s_t)  # input a stack of 4 images, get the prediction
                max_Q = np.argmax(q)  # chosing index with maximum q value
                action_index = max_Q
                a_t[action_index] = 1  # o=> do nothing, 1=> jump

        # We reduced the epsilon (exploration parameter) gradually
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # run the selected action and observed next state and reward
        x_t1, r_t, terminal = game_state.get_state(a_t)
        print(
            "fps: {0}".format(1 / (time.time() - last_time))
        )  # helpful for measuring frame rate
        last_time = time.time()
        x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1)  # 1x20x40x1
        s_t1 = np.append(
            x_t1, s_t[:, :, :, :3], axis=3
        )  # append the new image to input stack and remove the first one

        # store the transition in D
        D.append((s_t, action_index, r_t, s_t1, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        # only train if done observing
        if t > OBSERVE:

            # sample a minibatch to train on
            minibatch = random.sample(D, BATCH)
            inputs = np.zeros(
                (BATCH, s_t.shape[1], s_t.shape[2], s_t.shape[3])
            )  # 32, 20, 40, 4
            targets = np.zeros((inputs.shape[0], ACTIONS))  # 32, 2

            # Now we do the experience replay
            for i in range(0, len(minibatch)):
                state_t = minibatch[i][0]  # 4D stack of images
                action_t = minibatch[i][1]  # This is action index
                reward_t = minibatch[i][2]  # reward at state_t due to action_t
                state_t1 = minibatch[i][3]  # next state
                terminal = minibatch[i][
                    4
                ]  # wheather the agent died or survided due the action

                inputs[i : i + 1] = state_t

                targets[i] = model.predict(state_t)  # predicted q values
                Q_sa = model.predict(state_t1)  # predict q values for next step

                if terminal:
                    targets[i, action_t] = reward_t  # if terminated, only equals reward
                else:
                    targets[i, action_t] = reward_t + GAMMA * np.max(Q_sa)

            loss += model.train_on_batch(inputs, targets)
            loss_df.loc[len(loss_df)] = loss
            q_values_df.loc[len(q_values_df)] = np.max(Q_sa)
        s_t = (
            initial_state if terminal else s_t1
        )  # reset game to initial frame if terminate
        t = t + 1

        # save progress every 1000 iterations
        if t % 1000 == 0:
            print("Now we save model")
            game_state._game.pause()  # pause game while saving to filesystem
            model.save_weights("model.h5", overwrite=True)
            save_obj(D, "D")  # saving episodes
            save_obj(t, "time")  # caching time steps
            save_obj(
                epsilon, "epsilon"
            )  # cache epsilon to avoid repeated randomness in actions
            loss_df.to_csv("./objects/loss_df.csv", index=False)
            scores_df.to_csv("./objects/scores_df.csv", index=False)
            actions_df.to_csv("./objects/actions_df.csv", index=False)
            q_values_df.to_csv(q_value_file_path, index=False)
            with open("model.json", "w") as outfile:
                json.dump(model.to_json(), outfile)
            clear_output()
            game_state._game.resume()
        # print info
        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        print(
            "TIMESTEP",
            t,
            "/ STATE",
            state,
            "/ EPSILON",
            epsilon,
            "/ ACTION",
            action_index,
            "/ REWARD",
            r_t,
            "/ Q_MAX ",
            np.max(Q_sa),
            "/ Loss ",
            loss,
        )

    print("Episode finished!")
    print("************************")


def playGame():
    game = Game()
    # dino = DinoAgent.DinoAgent(game)
    # game_state = GameState.GameState(dino, game)
    # # model = buildmodel()
    # try:
    #     trainNetwork(model,game_state,observe=observe)
    # except StopIteration:
    #     game.end()


if __name__ == "__main__":

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

    playGame()
