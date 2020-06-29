import numpy as np


class Game2048:

    def __init__(self):

        self.board_size = 4
        self.board = np.zeros([self.board_size, self.board_size], dtype=int)
        self.restart()
        self.valid_moves = np.ones(4)
        self.get_valid_moves()
        self.game_over = False

    def move(self, direction):
        """
        method for all associated logic for a move. Checks which moves are possible
        checks the game isn't ended etc. Only operates on the board associated with this
        instance of the class.

        inputs:
            direction - desired move in one hot vector form
        output:
            None
        """
        if self.game_over:
            return

        if not self.test_direction(direction):
            return

        if self.valid_moves[np.where(direction == 1)] == 0:
            # move is not valid
            print("invalid move attempted")
            return

        board_copy = self.board.copy()
        self.manipulate(direction)

        # check if board moved
        if self.compare_boards(self.board, board_copy):
            return

        # spawn new tile
        self.spawn()

        # update valid moves
        self.get_valid_moves()

        # check if game is over
        if sum(self.valid_moves) == 0:
            # no valid moves, therefore game over
            self.game_over = True

        return self.board

    def manipulate(self, direction, input_board=None):
        """
        method for creating board dynamics

        inputs:
            direction - direction for the board to move specified by a 1-hot
                vector of length 4 where the positions specify [up, down, right, left]
            input_board - default is board tied to current game and this
                but allows other boards to be input for checking things
        output:
            input_board - the manipulated board
        """

        if input_board is None:
            input_board = self.board

        if not self.test_direction(direction):
            return input_board

        if not self.test_shape(input_board):
            return input_board

        if direction[0] == 1:
            # board motion is up
            # cycle over vertical slices
            for i in range(len(input_board)):
                slice = input_board[:, i]  # extract vertical slice
                slice = slice[slice != 0]  # remove all zeros from the slice
                combined = False
                for j in range(len(slice) - 1):  # -1 because if we get to the last element
                    # there is nothing it can combine with
                    if combined:
                        # combination on this slice has already happened
                        continue
                    elif slice[j] == slice[j + 1]:
                        slice[j] = slice[j] * 2
                        slice = np.delete(slice, j + 1)
                        combined = True
                # new slice which contains no zeros and the combinations have happened
                # need to re insert the relevant zeros
                while len(slice) < len(input_board):
                    slice = np.append(slice, 0)
                input_board[:, i] = slice  # reassign new values to the vertical slice in the original array

        elif direction[1] == 1:
            # board motion is down
            # cycle over vertical slices
            for i in range(len(input_board)):
                slice = input_board[:, i]  # extract vertical slice
                slice = slice[slice != 0]  # remove all zeros from the slice
                combined = False
                for j in range(len(slice) - 1, 0, -1):  # iterate in reverse order and don't
                    # iterate to 0 because if the last element
                    # is reached there is nothing for it to combine with anyway.
                    if combined:
                        continue
                    elif slice[j] == slice[j - 1]:
                        slice[j] = slice[j] * 2
                        slice = np.delete(slice, j - 1)
                        combined = True
                # reinsert zeros
                while len(slice) < len(input_board):
                    slice = np.insert(slice, 0, 0)
                input_board[:, i] = slice

        elif direction[2] == 1:
            # board motion is right
            # cycle over horizontal slices
            for i in range(len(input_board)):
                slice = input_board[i, :]  # extract horizontal slice
                slice = slice[slice != 0]  # remove all zeros from the slice
                combined = False
                for j in range(len(slice) - 1, 0, -1):  # iterate in reverse order and don't
                    # iterate to 0 because if the last element
                    # is reached there is nothing for it to combine with anyway.
                    if combined:
                        continue
                    elif slice[j] == slice[j - 1]:
                        slice[j] = slice[j] * 2
                        slice = np.delete(slice, j - 1)
                        combined = True
                # reinsert zeros
                while len(slice) < len(input_board):
                    slice = np.insert(slice, 0, 0)
                input_board[i, :] = slice

        else:
            # board motion is left
            # cycle over horizontal slices
            for i in range(len(input_board)):
                slice = input_board[i, :]
                slice = slice[slice != 0]
                combined = False
                for j in range(len(slice) - 1):  # -1 because if we get to the last element
                    # there is nothing it can combine with
                    if combined:
                        continue
                    elif slice[j] == slice[j + 1]:
                        slice[j] = slice[j] * 2
                        slice = np.delete(slice, j + 1)
                        combined = True
                # new slice which contains no zeros and the combinations have happened
                # need to re insert the relevant zeros
                while len(slice) < len(input_board):
                    slice = np.append(slice, 0)
                input_board[i, :] = slice  # reassign new values to the vertical slice in the original array

        return input_board

    def spawn(self, input_board=None):
        """
        method for spawning new numbers on the board after a movement

        inputs:
            input_board - default will be the board tied to this class, can input others
        output:
            input_board - the original input board plus a new tile in a position where there was a zero
        """

        if input_board is None:
            input_board = self.board

        if not self.test_shape(input_board):
            return input_board

        # pick which value tile to add
        if np.random.rand() > 0.9:
            spawn_number = 4
        else:
            spawn_number = 2
        zero_indices = np.where(input_board == 0)  # find where there are zeros on the board
        if len(zero_indices[0]) != 0:  # if there is at least one zero on the board
            randint = np.random.randint(0, len(zero_indices[0]))  # use random integer to pick where to spawn tile
            spawn_indx = (zero_indices[0][randint], zero_indices[1][randint])  # extract index of selected tile
            input_board[spawn_indx] = spawn_number  # input tile at selected location

        return input_board

    @staticmethod
    def test_shape(input_board):
        """
        method to test if the board shape is correct. Used by other functions

        inputs:
            input_board - a board to check
        output:
            either true or false depending if the board is a 4x4 matrix or not
        """

        if input_board.shape != (4, 4):
            print("input board is of the wrong shape should be [4, 4]")
            return False
        else:
            return True

    @staticmethod
    def test_direction(direction):
        """
        method to check a direction vector is valid

        inputs:
            direction - direction vector in np array form
        output:
            either true or false depending on if vector is one hot with length 4
        """
        # check the format of the direction vector is correct
        if not np.sum(direction) == 1:
            print("direction vector is not a one hot vector")
            return False
        if len(direction) != 4:
            print("length of direction vector is wrong, should be 4")
            return False
        return True

    def compare_boards(self, board1, board2=None):
        """
        method for checking if two boards are the same input two 4x4 np array
        boards and it will return true if they're the same and false if not
        """
        if board2 is None:
            board2 = self.board

        if not self.test_shape(board1):
            print("first board has the wrong shape")
            return False
        if not self.test_shape(board2):
            print("second board has the wrong shape")
            return False

        if np.array_equal(board1, board2):
            # boards are the same
            return True
        else:
            # boards are different
            return False

    def board_to_vector(self, input_board=None):
        """
        method for representing the board as a column vector of length 16

        inputs:
            input_board: 4x4 input board from the game with raw game values ie 0 to 2048

        output:
            board_vector: A**2 length vector representing the game board as floats in matrix format
        """
        if input_board is None:
            input_board = self.board

        if not self.test_shape(input_board):
            return None

        board_vector = input_board.flatten()  # flatten array to a vector
        board_vector = np.reshape(board_vector, (len(board_vector), 1))
        board_vector = np.log2(board_vector)  # take log base 2 of all game tiles
        np.place(board_vector, board_vector < 0, 0)  # replace -inf with 0
        board_vector = board_vector / 11  # normalise tile values by dividing by 11 (max possible value)
        return np.matrix(board_vector)

    def get_valid_moves(self, input_board=None):
        """
        method for finding the moves that are valid from the current board

        inputs:
            input_board - 4x4 numpy array
        output:
            valid_moves - vector of length 4 with ones corresponding to valid
                for the board [up, down, right, left]
        """

        if input_board is None:
            input_board = self.board

        for i in range(4):
            board_copy = input_board.copy()
            test_direction = np.zeros(4)
            test_direction[i] = 1
            self.manipulate(test_direction, input_board=board_copy)
            if self.compare_boards(board_copy, input_board):
                # boards are the same and move is not valid
                self.valid_moves[i] = 0
            else:
                # boards weren't the same and the move is valid
                self.valid_moves[i] = 1

    def log2_board(self, input_board=None):
        """
        method to compute the log2 version of the board

        inputs:
            input_board - 4x4 numpy array
        output:
            input_board - 4x4 numpy array of integers representing
                the log2 of the input boards values
        """
        if input_board is None:
            input_board = self.board

        # input_board = np.log2(input_board, where=input_board > 0)  # take log base 2 of all game tiles
        input_board = np.log2(input_board)
        # np.place(input_board, input_board < 0, 0)  # replace -inf with 0
        input_board[np.where(input_board < 0)] = 0  # replace -inf with 0

        return input_board

    def restart(self):
        """
        Method to initialise the board
        """
        self.board = np.zeros([4, 4], dtype=int)
        self.board = np.array(self.board)
        for i in range(2):
            self.board[np.random.randint(0, 4), np.random.randint(0, 4)] = 2

        self.game_over = False
        self.valid_moves = np.ones(4)
        self.get_valid_moves()

    def board_to_binary(self, input_board=None):
        """
        Method for taking the game board and generating a binary representation
        for value function approximation. Each tile is represented by a 1-hot vector
        which are combined into one long vector which represents the entire board in
        a binary fashion. Call with no argument to use the current game board

        input:
        input_board - a board that isn't the current game board in np array 4x4 format
        output:
        board_vector - 176 bit binary vector representation of board
        """
        if input_board is None:
            input_board = self.board  # if no board given use the current game board

        input_board = self.log2_board(input_board)  # log 2 of board to show what index to put the one at
        input_board = np.array(input_board, dtype="int8").flatten()  # convert board to a 16x1 vector of integer values

        board_vector = []  # initialise array to store binary tiles
        # loop over each tile in the board and create it's one-hot vector
        for tile in input_board:
            binary_tile = np.zeros(12, dtype="int8")  # initialise a vector of zeros length 16 (represents one tile)
            binary_tile[tile] = 1  # put a 1 at the location that corresponds to tile value
            board_vector.append(binary_tile)  # add the one hot tile vector to the total vector

        # create an array from the one hot vector tile
        # representations then flatten to create binary board vector
        board_vector = np.array(board_vector)
        board_vector = board_vector.flatten()
        board_vector = board_vector.reshape([len(board_vector), 1])

        return board_vector  # return the vectorised binary board representation

    def get_tile_distribution(self, input_board=None):
        """
        Method for getting the distribution of each tile type on the board

        Inputs:
        input_board - the board to be evaluated. If left the current game board is used

        Outputs:
        distribution - 12 length vector representing the number of each tile on the board
        """
        if input_board is None:
            input_board = self.board

        input_board = self.log2_board(input_board)

        distribution = np.zeros(12, dtype="int8")
        for i in range(12):
            distribution[i] = len(np.where(input_board == i)[0])

        return distribution

    def get_board_sum(self, input_board=None):
        """
        Method for getting the sum of all tiles on a board

        inputs:
        input_board - the board to be evaluated. If left the current game board is used
        outputs:
        board_sum - sum of all tiles on the input board
        """
        if input_board is None:
            input_board = self.board

        board_sum = np.sum(input_board)

        return board_sum

    def random_board(self):
        """
        method to create a random board, for debugging
        """

        board = np.random.randint(0, 11, (4, 4))
        for i in range(len(board)):
            for j in range(len(board[i])):
                if board[i, j] != 0:
                    board[i, j] = 2**board[i, j]

        return board

    def get_tile_boolean_distribution(self):
        """
        Method that returns a 11 length vector with either a 1 or 0 in each position.
        A 1 means the tile of that value is present in the board and a 0 it isn't present.
        """