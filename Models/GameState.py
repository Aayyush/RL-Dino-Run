from constants import *
import pickle
import numpy as np
import cv2
from PIL import Image


class GameState:
    def __init__(self, agent, game):
        self._agent = agent
        self._game = game
        self._display = show_image()
        self._display.__next__()

    def getState(self, actions):
        actions_df.loc[len(actions_df)] = actions[1]
        score = self._game.get_score()
        reward = 0.1
        isGameOver = False

        if actions[1] == 1:  # 1 => index for jump and 0 for nothing
            self._game.jump()
        image = grabScreen(self._game._driver)
        self._display.send(image)

        # Check if the agent has crashed and add reward.
        if self._agent.is_crashed():
            scores_df.loc[len(scores_df)] = score
            self._game.restart()
            reward = -1
            isGameOver = True

        return image, reward, isGameOver


def save_object(obj, name):
    with open("objects/" + name + ".pkl", "wb") as f:  # dump files into objects folder
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_object(name):
    with open("objects/" + name + ".pkl", "rb") as f:
        return pickle.load(f)


def grab_screen(_driver):
    image_base_64 = _driver.execute_script(getbase64Script)
    screen = np.array(Image.open(BytesIO(base64.b64decode(image_base_64))))
    image = process_image(screen)
    return image


def process_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # RGB to Grey Scale
    image = image[:300, :500]  # Crop Region of Interest(ROI)
    image = cv2.resize(image, (80, 80))
    return image


def show_image(graphs=False):
    while True:
        screen = (yield)
        window_title = "logs" if graphs else "game_play"
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        imS = cv2.resize(screen, (800, 400))
        cv2.imshow(window_title, screen)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break
