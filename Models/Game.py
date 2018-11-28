from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys

from constants import *


class Game:
    """
    Class to interact with the game. 
    """

    def __init__(self):
        chrome_options = Options()
        chrome_options.add_argument("diasble-infobars")
        chrome_options.add_argument("--mute-audio")

        self._driver = webdriver.Chrome(
            executable_path=chrome_driver_path, chrome_options=chrome_options
        )
        self._driver.set_window_position(x=-10, y=0)
        self._driver.get("chrome://dino")
        self._driver.execute_script("Runner.config.ACCELERATION=0")
        self._driver.execute_script(init_script)

    def get_crashed(self):
        return self._driver.execute_script("return Runner.instance_.crashed")

    def get_playing(self):
        return self._driver.execute_script("return Runner.instance_.playing")

    def restart_game(self):
        return self._driver.execute_script("return Runner.instance_.restart()")

    def press_up(self):
        return self._driver.find_element_by_tag_name("body").send_keys(Keys.ARROW_UP)

    def get_score(self):
        totalScore = self._driver.execute_script(
            "return Runner.instance_.distanceMeter.digits"
        )
        totalScore = "".join(totalScore)
        return totalScore

    def game_pause(self):
        return self._driver.execute_script("return Runner.instance_.stop()")

    def game_resume(self):
        return self._driver.execute_script("return Runner.instance_.play()")

    def game_end(self):
        self._driver.close()
