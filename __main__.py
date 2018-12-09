from module_imports import *

# training variables saved as checkpoints to filesystem to resume training from the same step
def init_cache():
    """initial variable caching, done only once"""
    save_object(INITIAL_EPSILON,"epsilon")
    t = 0
    save_object(t,"time")
    D = deque()
    save_object(D,"D")

def save_object(obj, name):
    with open("/Users/aayushgupta/Desktop/RL-Dino-Run/objects/" + name + ".pkl", "wb") as f:  # dump files into objects folder
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_object(name):
    with open("/Users/aayushgupta/Desktop/RL-Dino-Run/objects/" + name + ".pkl", "rb") as f:
        return pickle.load(f)

# Build Model.
def buildModelForDinoRun():

    """
    Convolution Layer 1: Filter size: 8 * 8, Strides: 4 * 4, Number of filters: 32, Pool Size: 2 * 2
    Convolution Layer 2: Filter size: 4 * 4, Strides: 2 * 2, Number of filters: 64, Pool Size: 2 * 2
    Convolution Layer 3: Filter size: 3 * 3, Strides: 1 * 1, Number of filters: 64, Pool Size: 2 * 2
    """
    dinoRunModel = Sequential()

    dinoRunModel.add(Conv2D(32,(8, 8),padding="same",strides=(4, 4),input_shape=(img_cols, img_rows, img_channels),))
    dinoRunModel.add(MaxPooling2D(pool_size=(2, 2)))
    dinoRunModel.add(Activation("relu"))

    dinoRunModel.add(Conv2D(64, (5, 5), strides=(2, 2), padding="same"))
    dinoRunModel.add(MaxPooling2D(pool_size=(2, 2)))
    dinoRunModel.add(Activation("relu"))

    dinoRunModel.add(Conv2D(64, (3, 3), strides=(1, 1), padding="same"))
    dinoRunModel.add(MaxPooling2D(pool_size=(2, 2)))
    dinoRunModel.add(Activation("relu"))

    # Flatten the pooling layer.
    dinoRunModel.add(Flatten())

    # Dense layer to generalize the input.
    dinoRunModel.add(Dense(512))
    dinoRunModel.add(Dense(256))
    dinoRunModel.add(Activation("relu"))

    # Output layer.
    dinoRunModel.add(Dense(ACTIONS))
    adam = Adam(lr=LEARNING_RATE)
    dinoRunModel.compile(loss="mse", optimizer=adam)

    # Save loss file if not available, 
    try:
        loss_file_handler = open(loss_file_path, 'r')
    except FileNotFoundError:
        dinoRunModel.save_weights("model.h5")
       
    return dinoRunModel


# Train Model.
def trainNeuralNetForDinoRun(model, gameState, isObservation=False):
    # Record the current time.
    lastTimeStamp = time.time()

    # Store the previous observations in replay memory.
    D = load_object("D")  # Load from file system.

    # Get the first state by doing nothing.
    emptyAction = np.zeros(ACTIONS)
    emptyAction[0] = 1  # 0 => do nothing 1=> jump.

    # Get next step after performing the action.
    imageAtStartOfGame, rewardAtStartOfGame, terminal = gameState.getState(emptyAction)

    # Stack 4 images to create placeholder input.
    consecutiveImagesAsInput = np.stack((imageAtStartOfGame, imageAtStartOfGame, imageAtStartOfGame, imageAtStartOfGame), axis=2)
    consecutiveImagesAsInput = consecutiveImagesAsInput.reshape(1, consecutiveImagesAsInput.shape[0], consecutiveImagesAsInput.shape[1], consecutiveImagesAsInput.shape[2])  # 1 * 20 * 40 * 4

    initialGameState = consecutiveImagesAsInput

    if isObservation:
        # We keep observing the frames not train.
        OBSERVE = 999999999
        epsilon = FINAL_EPSILON

        model.load_weights("model.h5")
        adam = Adam(lr=LEARNING_RATE)
        model.compile(loss="mse", optimizer=adam)
        print("Weight load successfully")
    # We go to training mode.
    else:
        OBSERVE = OBSERVATION
        epsilon = load_object("epsilon")
        model.load_weights("model.h5")
        adam = Adam(lr=LEARNING_RATE)
        model.compile(loss="mse", optimizer=adam)

    t = load_object("time")  # Resume from the previous time step stored in file system
    while True:  # Endless running

        loss = 0
        QScoreAfterAction = 0
        actionIndexForNextAction = 0
        rewardAtStartOfGame = 0  # reward at 4
        action = np.zeros([ACTIONS])  # action at t

        # choose an action epsilon greedy
        if t % FRAME_PER_ACTION == 0:  # parameter to skip frames for actions
            if random.random() <= epsilon:  # randomly explore an action
                print("----------Random Action----------")
                actionIndexForNextAction = random.randrange(ACTIONS)
                action[actionIndexForNextAction] = 1
            else:  # predict the output
                qValueForPrediction = model.predict(consecutiveImagesAsInput)  # input a stack of 4 images, get the prediction
                maxQValueForPrediction = np.argmax(qValueForPrediction)  # chosing index with maximum q value
                actionIndexForNextAction = maxQValueForPrediction
                action[actionIndexForNextAction] = 1  # o=> do nothing, 1=> jump

        # We reduced the epsilon (exploration parameter) gradually
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # run the selected action and observed next state and reward
        imageAfterAction, rewardAfterAction, terminal = gameState.getState(action)
        print(
            "fps: {0}".format(1 / (time.time() - lastTimeStamp))
        )  # helpful for measuring frame rate
        lastTimeStamp = time.time()
        imageAfterAction = imageAfterAction.reshape(1, imageAfterAction.shape[0], imageAfterAction.shape[1], 1)  # 1x20x40x1
        gameStateAfterAction = np.append(
            imageAfterAction, consecutiveImagesAsInput[:, :, :, :3], axis=3
        )  # append the new image to input stack and remove the first one

        # store the transition in D
        D.append((consecutiveImagesAsInput, actionIndexForNextAction, rewardAtStartOfGame, gameStateAfterAction, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        # only train if done observing
        if t > OBSERVE:

            # sample a minibatch to train on
            minibatch = random.sample(D, BATCH)
            inputs = np.zeros(
                (BATCH, consecutiveImagesAsInput.shape[1], consecutiveImagesAsInput.shape[2], consecutiveImagesAsInput.shape[3])
            )  # 32, 20, 40, 4
            targets = np.zeros((inputs.shape[0], ACTIONS))  # 32, 2

            # Now we do the experience replay
            for i in range(0, len(minibatch)):
                gameStateAtTime = minibatch[i][0]  # 4D stack of images
                actionAtTime = minibatch[i][1]  # This is action index
                rewardAtTime = minibatch[i][2]  # reward at state_t due to action_t
                gameStateAtTimeAfterAction = minibatch[i][3]  # next state
                terminal = minibatch[i][
                    4
                ]  # wheather the agent died or survided due the action

                inputs[i : i + 1] = gameStateAtTime

                targets[i] = model.predict(gameStateAtTime)  # predicted q values
                QScoreAfterAction = model.predict(gameStateAtTime)  # predict q values for next step

                if terminal:
                    targets[i, actionAtTime] = rewardAtTime  # if terminated, only equals reward
                else:
                    targets[i, actionAtTime] = rewardAtTime + GAMMA * np.max(QScoreAfterAction)

            loss += model.train_on_batch(inputs, targets)
            loss_df.loc[len(loss_df)] = loss
            q_values_df.loc[len(q_values_df)] = np.max(QScoreAfterAction)

        consecutiveImagesAsInput = (
            initialGameState if terminal else gameStateAfterAction
        )  # reset game to initial frame if terminate
        t = t + 1

        # save progress every 1000 iterations
        if t % 1000 == 0:
            print("Now we save model")
            gameState._game.pause()  # pause game while saving to filesystem
            model.save_weights("model.h5", overwrite=True)
            save_object(D, "D")  # saving episodes
            save_object(t, "time")  # caching time steps
            save_object(
                epsilon, "epsilon"
            )  # cache epsilon to avoid repeated randomness in actions
            loss_df.to_csv("./objects/loss_df.csv", index=False)
            scores_df.to_csv("./objects/scores_df.csv", index=False)
            actions_df.to_csv("./objects/actions_df.csv", index=False)
            q_values_df.to_csv(q_value_file_path, index=False)
            with open("model.json", "w") as outfile:
                json.dump(model.to_json(), outfile)
            clear_output()
            gameState._game.resume()
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
            actionIndexForNextAction,
            "/ REWARD",
            rewardAtStartOfGame,
            "/ Q_MAX ",
            np.max(QScoreAfterAction),
            "/ Loss ",
            loss,
        )

    print("Episode finished!")
    print("************************")


def playGame(observe=False):
    game = Game()
    dino = DinoAgent(game)
    game_state = GameState(dino, game)
    model = buildModelForDinoRun()
    try:
        trainNeuralNetForDinoRun(model,game_state,isObservation=observe)
    except StopIteration:
        game.game_end()


if __name__ == "__main__":

    init_cache()

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

    # Initialize the server to communicate with the game.=
    playGame()
