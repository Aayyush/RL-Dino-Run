class DinoAgent:

    def __init__(self, game):
        self._game = game
        self.jump()  # To start the game we need to jump once.
    
    def is_running(self):
        return self._game.get_playing()
    
    def is_crashed(self):
        return self._game.get_crashed()
    
    def jump(self):
        return self._game.press_up()
    
    def duck(self):
        return self._game.press_down()
    
