# from constants import *
# import pickle
# import numpy as np
# import cv2
# from PIL import Image


from module_imports import *

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
            self._game.press_up()
        image = grab_screen(self._game._driver)
        self._display.send(image)

        # Check if the agent has crashed and add reward.
        if self._agent.is_crashed():
            scores_df.loc[len(scores_df)] = score
            self._game.restart_game()
            reward = -1
            isGameOver = True

        return image, reward, isGameOver


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
