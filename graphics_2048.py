from appJar import gui
import numpy as np


class Graphics2048:

    def __init__(self, game, manual_input=False):
        # define gui window
        self.app = gui("2048 Game Window", "400x400")

        self.app.setBg("Gray")  # set background colour to grey
        self.app.setFont(20)  # set font size to 20
        self.app.setSticky("news")  # set widgets to stick to their sides and not reshape to content
        self.app.setExpand("both")  # set widgets to expand in both directions
        self.app.setPadding([5, 5])

        if manual_input:
            self.app.bindKey("Up", self.keyPress)
            self.app.bindKey("Down", self.keyPress)
            self.app.bindKey("Right", self.keyPress)
            self.app.bindKey("Left", self.keyPress)
            # app.bindKey("r", self.keyPress)

        self.game = game

        # define the colours for different tiles based on value
        self.colors = ["DarkGray", "LightGrey", "Beige", "Orange", "DarkOrange", "OrangeRed", "Red",
                  "LightYellow", "LemonChiffon", "LemonChiffon", "Yellow", "Gold"]

        self.update_graphic(initialise=True)
        self.app.registerEvent(self.update_graphic)

    def start(self):
        self.app.go()

    def keyPress(self, key):
        direction = np.zeros(4)
        if key == "Up":
            # key pressed is up
            direction[0] = 1
            self.game.move(direction)
            self.update_graphic()
        elif key == "Down":
            # key pressed is down
            direction[1] = 1
            self.game.move(direction)
            self.update_graphic()
        elif key == "Right":
            # key pressed is right
            direction[2] = 1
            self.game.move(direction)
            self.update_graphic()
        else:
            # key pressed is left
            direction[3] = 1
            self.game.move(direction)
            self.update_graphic()

    def update_graphic(self, initialise=False, outside=False):
        # create 16 labels as the game tiles
        count = 0
        row_count = 0
        for row in self.game.board:
            element_count = 0
            for element in row:
                if initialise:
                    self.app.addLabel("l" + str(count), str(element) if element is not 0 else "", row_count, element_count)
                elif outside:
                    self.app.queueFunction(self.app.setLabel("l" + str(count), str(element) if element is not 0 else ""))
                else:
                    self.app.setLabel("l" + str(count), str(element) if element is not 0 else "")
                if element == 0:
                    color_index = 0
                else:
                    color_index = int(np.log2(element))

                if outside:
                    self.app.queueFunction(self.app.setLabelBg("l" + str(count), self.colors[color_index]))
                else:
                    self.app.setLabelBg("l" + str(count), self.colors[color_index])
                element_count += 1
                count += 1
            row_count += 1
