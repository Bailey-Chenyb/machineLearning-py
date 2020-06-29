import numpy as np
from Project.dynamics_2048 import Game2048
import time


def create_trace(weights, value_function, representation, epsilon=0.1):
    """
    Method to play the game using a given feature representation and value function.
    It will return a trace of states, actions and rewards for each move.

    Inputs:
    value_function - the value function to be used for evaluating which move to make.
                    Currently either "v" or "q" depending on whether the weights multiplied
                    by state representation will give a board value ("v") or move value ("q")
                    respectively.
    representation - the feature representation used for the board. Currently either "vector"
                    or "binary".
    epsilon - the epsilon value used in epsilon greedy move selection.

    Outputs:
    trace - a list of numpy arrays which contain 3 parts: the board state in the given feature
            feature representation format, a 4x1 one hot vector representing the direction chosen
            [up, down, right, left] and the reward for taking that move from the given state.
    """

    # initialise empty array to store the board states, moves and rewards during while loop
    trace = []

    # initialise some counters for the while loop
    move_count = 0
    bug_count = 0
    while not game.game_over:
        move_count += 1  # increment move counter by 1 each time loop is executed

        # calculate the direction using an epsilon greedy policy
        direction = epsilon_greedy(weights, representation, value_function, epsilon)

        # following if block used for debugging as the game would sometimes get stuck constantly
        # attempting the same invalid move so if the move count goes above 1000 the code will
        # print out some information to help debug the problem then exit after 5 more moves
        if move_count > 1000:
            print("valid moves: " + str(game.valid_moves))
            print("direction: " + str(direction))
            print("game board: " + str(game.board))
            bug_count += 1
            if bug_count > 5:
                exit()

        # make a copy of the board before executing the move then execute the move and
        # calculate the reward using the old and new board configurations
        old_board = game.board.copy()  # create copy of board before move
        game.move(direction)  # execute move
        reward = calculate_reward(old_board, game.board)  # calculate reward for the move

        # add to the trace for this move
        if representation == "vector":
            trace.append([np.array(game.board_to_vector(old_board)), direction, reward])
        elif representation == "binary":
            trace.append([np.array(game.board_to_binary(old_board)), direction, reward])
        else:
            print("invalid representation type passed to create trace function: " + str(representation))
            return

    # print out some information on the trace
    print("game over")
    print("length of trace: %d" % len(trace))
    return trace


def epsilon_greedy(weights, representation, value_function, epsilon):
    """
    Function for calculating the next move direction based on an epsilon greedy
    policy using any representation or value function

    Inputs:
    weights - current weights for the given representation and algorithm
    representation - feature representation being used. Either "vector" or "binary"
    value_function - the value function being used. Currently either "v" or "q"
    epsilon - probability of a move other than the greedy move choice being chosen

    Outputs:
    direction - a 4x1 one hot vector representing the direction for the agent to move
    """

    if representation == "vector":
        feature_vector = np.matrix(game.board_to_vector())
    elif representation == "binary":
        feature_vector = np.matrix(game.board_to_binary())
    else:
        print("incorrect feature representation passed to the epsilon greedy function: " + str(representation))
        return

    if value_function == "v":
        # calculate state value for each move
        # start by initialising an empty 4x1 vector to store calculated values
        # for each board that can be moved in to
        values = np.zeros(4)

        # loop over the 4 possible move options
        for i in range(4):
            # check the move is possible in the current board state
            if game.valid_moves[i] == 1:
                # move is valid so we will calculate the new state value
                direction = np.zeros(4)  # initialise empty 4x1 vector to represent move direction
                direction[i] = 1  # make it one hot by assigning 1 to the move being analysed for this loop

                # create copy of board and move it in the direction specified for this loop then calculate the
                # value of the board state moved into using that direction
                board_copy = game.board.copy()
                board_copy = game.manipulate(direction, input_board=board_copy)
                if representation == "vector":
                    board_vector = game.board_to_vector(input_board=board_copy)
                else:
                    board_vector = game.board_to_binary(input_board=board_copy)
                values[i] = float(np.dot(weights, board_vector))

    elif value_function == "q":
        # calculate values of each move by dot producting the weight matrix and feature vector
        values = np.dot(weights, feature_vector)
        values = np.multiply(game.valid_moves.reshape([4, 1]), values)
    else:
        print("incorrect value function passed to the epsilon greedy function: " + str(value_function))
        return

    # get the number of available moves then choose one in the epsilon greedy fashion
    nz_moves = len(values[values != 0])
    values[values == 0] = -1 * np.inf  # replace not available moves with -inf value so they aren't chosen in the argmax
    move_idx = np.argmax(values)
    # create a random number between 0 and 1 and if it's below epsilon a direction not chosen greedily is chosen
    if np.random.rand() < epsilon*(nz_moves-1) and nz_moves > 1:
        while move_idx == np.argmax(values) or values[move_idx] == -1 * np.inf:
            move_idx = np.random.randint(0, 4)

    # create 4x1 one hot vector representing the chosen move direction
    direction = np.zeros(4)
    direction[move_idx] = 1
    return direction


def calculate_weight_changes(weights, trace, alpha, cumulative="normal"):
    """
    Simplified method to calculate the weight changes for any representation or value function

    Inputs:
    weights - current weights used to create the trace over which weight changes are being calculated
    trace - a trace containing the states in their respective feature representation format, actions
            taken and rewards for taking those actions for en entire episode
    alpha - learning rate to be used in calculating the weight changes
    cumulative - either "normal", "reverse" or "none" to specify the type of reward summing used for
                weight change evaluation.

    Outputs:
    delta_ws - weight changes to apply in the same shape as the input weights
    """
    # make sure the trace is in the form of an np array
    trace = np.array(trace)

    # initiate empty array to store weight changes for each state in the trace
    delta_ws = []
    for i in range(len(trace)):
        # extract state and make sure it's an np matrix for multiplication later
        state = np.matrix(trace[i, 0])
        # extract direction and make sure it's a 4x1 one hot vector
        direction = np.reshape(trace[i, 1], [4, 1])

        # calculate reward for this move depending on whether cumulative rewards
        # are used or not
        if cumulative == "normal":
            reward = np.sum(trace[i:len(trace), 2])
        elif cumulative == "reverse":
            reward = np.sum(trace[0:i, 2])
        elif cumulative == "none":
            reward = trace[i, 2]
        else:
            print("incorrect cumulative reward type passed to the calculate "
                  "weight changes function: " + str(cumulative))
            return

        # calculate the state or state action value depending on the value function
        # type
        if weights.shape[0] == 4:
            # using q value function
            value = np.multiply(np.dot(weights, state), direction)
            value = value[value != 0]
            value = value[0, 0]

            # calculate weight change for the state action reward sequence and reshape
            # to be the same size as the main weight array
            delta_w = (alpha*(reward - value)*state).reshape(len(state))
            empty_w = np.zeros(weights.shape)
            empty_w[np.argmax(direction)] = delta_w
            delta_w = empty_w
            delta_ws.append(delta_w)
        else:
            # using a value approximation value
            value = float(np.dot(weights, state))

            # calculate changes to the weights from this state action reward sequence
            # and append it to the list of weight changes for the trace
            delta_ws.append((alpha*(reward - value)*state).reshape(len(state)))

    if weights.shape[0] == 4:
        # use NaN mean to create a mean across the number of times the move was actually
        # taken not over the entire trace length. Then remove the NaNs left from
        # moves that weren't taken
        delta_ws = np.array(delta_ws)
        delta_ws[delta_ws == 0] = np.nan
        delta_w = np.nanmean(delta_ws, axis=0)
        delta_w = np.nan_to_num(delta_w)
    else:
        # average weight changes for each move in the trace
        delta_ws = np.array(delta_ws)
        delta_w = np.average(delta_ws, axis=0)

    return delta_w


def calculate_reward(board1, board2):
    """
    method to calculate the reward for a given move by taking the board before
    and after the move

    inputs:
        board1 - board before move
        board2 - board after move
    output:
        reward - float reward for the move
    """
    board1_sum = int(np.sum(game.log2_board(board1)))
    board2_sum = int(np.sum(game.log2_board(board2)))
    # if no tiles combined
    if len(board2[board2 != 0].flatten()) > len(board1[board1 != 0].flatten()):
        reward = -1

    # if tiles combined score is:
    else:
        reward = board1_sum - board2_sum + 1
        # +1 because will always add at least 1 for a spawned tile
        # sum of log2 values should reduce if large tiles combine therefore a
        # positive score is given if the log2 sum decreases
    return reward


def evaluate_policies(all_weights, no_weights, no_episodes, algorithm, representation):
    """
    Method to use after learning to evaluate the learned weights.
    Code is essentially the same as for learning but epsilon is set to zero so the
    learned weights max value is always chosen (epsilon greedy)

    Inputs:
    no_weights - the number of weights to evaluate from the most recent backward
    no_episodes - the number of episodes to use on each set of weights to evaluate them

    Outputs:
    max_tiles_evaluation.txt - csv file containing the average largest tiles for each of the weight
                                sets evaluated
    performance_info_evaluation.txt - csv file containing performance information about the episodes
                                    played using each weight set. Final tile distribution average, max
                                    tiles average, board sum average and max tile for each batch are
                                    stored.
    """
    # extract the value function type from the algorithm string
    val_fun_type = algorithm.split("_")
    if len(val_fun_type) == 2:
        if val_fun_type[1] == "v" or val_fun_type[1] == "q":
            val_fun_type = val_fun_type[1]
        else:
            print("invalid algorithm type passed to learn function of: " + str(algorithm))
    else:
        print("invalid algorithm type passed to learn function of: " + str(algorithm))

    # initialise list to store performance info
    all_info = []

    # loop the number of different weight sets to be evaluated.
    # index is reversed so the last set of weights is chosen first and so forth backwards
    # the specified number of times.
    for i in range(all_weights.shape[0] - 1, all_weights.shape[0] - no_weights - 1, -1):
        # extract the weight matrix from the list of all weight matrices for the learning process
        if val_fun_type == "v":
            weights = np.matrix(all_weights[i, :])
        else:
            weights = np.matrix(all_weights[i, :].reshape([4, int(len(all_weights[i, :])/4)]))

        # initialise list to store the performance info of each episode
        info = []
        # loop the number of episodes to be performed on each weight set
        for j in range(no_episodes):
            # play game and create trace then calculate the largest tile, final tile distribution
            # and board sum then append these to the relevant lists and restart the game board for
            # the next loop
            trace = create_trace(weights, val_fun_type, representation, epsilon=0)
            info.append(extract_info(raw=True, trace=trace))
            print("game %d evaluation weights %d max tile: %d" % (j, i, game.board.max()))
            game.restart()

        # append information to the relevant lists. Calculate distributions, maximum tile and board sum averages
        # then append to the info array for this weight set which is a 3x1 np array. This array is then appended
        # to the larger all_info list.
        info = np.array(info)
        all_info.append(extract_info(average=True, info=info))

    # format the information array then save it and and the largest tiles array into their respective
    # text files.
    all_info = np.array(all_info)
    info_header = extract_info(header=True)
    np.savetxt("%sperformance_info_evaluation.txt" % file_prepend, all_info, delimiter=",", header=info_header)

    # print some information about the evaluation process
    print("all biggest tiles for evaluation: " + str(all_info[:, 12]))


def learn(representation, algorithm, batches=10, episodes=10, alpha=0.05,
          epsilon=0.05, cumulative="normal", taper_epsilon=False):
    """
    Single method to implement learning weights using any of the available algorithms and
    feature representations

    Inputs:
    representation - type of feature representation to use for game states. Currently either
                        binary or vector
    algorithm - learning algorithm to use to attain learned weights. Current options are:
                mc_v (monte carlo board value function)
                mc_q (monte carlo q value function)
    batches - the number of batches to use before applying weight changes when using monte carlo
                learning
    episodes - number of episodes to use per batch or overall if not using monte carlo
    alpha - learning rate to be used for the given algorithm
    epsilon - probability of choosing an action other than the one prescribed by the value function

    Outputs:
    weights.txt - csv file with all the different sets used in the learning process
    performance_info.txt - csv file detailing some performance metrics of each set of learned weights
    """
    # calculate how much to taper epsilon by each batch
    if taper_epsilon:
        taper_amount = epsilon/batches
    else:
        taper_amount = 0

    # extract the value function type from the algorithm string
    val_fun_type = algorithm.split("_")
    if len(val_fun_type) == 2:
        if val_fun_type[1] == "v" or val_fun_type[1] == "q":
            val_fun_type = val_fun_type[1]
        else:
            print("invalid algorithm type passed to learn function of: " + str(algorithm))
    else:
        print("invalid algorithm type passed to learn function of: " + str(algorithm))

    # first the weights to be used must be created which is a factor of the
    # representation and algorithm to be used so the following if else block calculates the correct
    # starting set of weights
    if representation == "vector":
        if val_fun_type == "v":
            # create random 16x1 vector of weights
            weights = np.matrix(np.random.rand(1, game.board_size**2))
        elif val_fun_type == "q":
            # create random 4x16 matrix of weights. Each row represents a different move
            weights = np.matrix(np.random.rand(4, game.board_size**2))
        else:
            print("invalid algorithm type given to learn function after split of: " + str(algorithm))
            return
    elif representation == "binary":
        # check binary representation length
        rep_len = len(game.board_to_binary())
        if val_fun_type == "v":
            # create vector of random weights of size Lx1 where L is
            # the length of binary representation vector
            weights = np.matrix(np.random.rand(1, rep_len))
        elif val_fun_type == "q":
            # create array of weights of size 4xL where L is the length of the
            # binary representation vector. Each row represents a different move
            weights = np.matrix(np.random.rand(4, rep_len))
        else:
            print("invalid algorithm type given to learn function after split of: " + str(algorithm))
            return
    else:
        print("invalid representation type given to learn function of: " + str(representation))
        return

    # initialise empty arrays to store the weights and performance information from all batches
    all_weights = []
    all_info = []
    for i in range(batches):
        all_weights.append(weights)

        # taper epsilon down if specified to do so
        epsilon = epsilon - taper_amount

        # initialise empty arrays to store the weight changes and performance information of
        # each episode within the batch
        delta_ws = []
        info = []
        for j in range(episodes):
            # create trace using the given value function and representation
            trace = create_trace(weights, val_fun_type, representation, epsilon=epsilon)

            # calculate weight changes for the trace and append them to the delta_ws list
            delta_ws.append(calculate_weight_changes(weights, trace, alpha, cumulative=cumulative))

            # store trace performance info in a list within the info list
            info.append(extract_info(raw=True, trace=trace))

            # print some episode info and restart the game dynamics class
            print("game %d batch %d max tile: %d" % (j, i, game.board.max()))
            game.restart()

        # take average of weights calculated from each
        delta_w = np.array(delta_ws)
        delta_w = np.average(delta_w, axis=0)
        weights = weights + delta_w

        # format performance information and append to the list for performance information over all batches
        info = np.array(info)
        all_info.append(extract_info(average=True, info=info))

        print("\n\n batch %d average max tiles: %d \n\n" % (i, np.average(info[:, 12])))

    all_info = np.array(all_info)
    all_weights = np.array(all_weights)
    all_weights = np.reshape(all_weights, [all_weights.shape[0], all_weights.shape[1]*all_weights.shape[2]])

    np.savetxt("%sweights.txt" % file_prepend, all_weights, delimiter=",")
    info_header = extract_info(header=True)
    np.savetxt("%sperformance_info.txt" % file_prepend, all_info, delimiter=",", header=info_header)

    return all_weights


def extract_info(raw=False, trace=None, average=False, info=None, header=False):
    """
    Method for extracting the relevant performance information for a trace. Edit
    this method to change what information about each run is collected and how it's
    averaged across a batch.

    Inputs:
    raw - boolean specifying whether the raw information is required or averaging of
            the collected data is required
    average - boolean specifying that averaged information is required
    info - an np array containing the raw collected info for all episodes in a batch
            ready to be averaged
    header - boolean to state the info header is required for the performance info file

    Outputs:
    info_line - an array containing the relevant performance information for one episode
    all_info_line - averaged performance information for one batch
    info_header - string containing the categories of information collected
    """
    if raw:

        if trace is None:
            print("Trace is required to store the raw episode information")
            return

        info_line = game.get_tile_distribution()
        info_line = np.append(info_line, game.board.max())
        info_line = np.append(info_line, game.get_board_sum())
        info_line = np.append(info_line, len(trace))
        info_line = np.append(info_line, np.sum(np.array(trace)[:, 2]))

        boolean_distribution = np.zeros(12)
        boolean_distribution[0:int(np.log2(game.board.max()))+1] = 1
        info_line = np.concatenate([info_line, boolean_distribution])
        return info_line

    elif average:

        if info is None:
            print("Info list not supplied. It is needed for averaging")
            return

        all_info_line = np.average(info[:, 0:12], axis=0)  # extract and average board distribution
        all_info_line = np.append(all_info_line, np.average(info[:, 12]))  # extract and average max tiles
        all_info_line = np.append(all_info_line, np.average(info[:, 13]))  # extract and average board sum
        all_info_line = np.append(all_info_line, np.max(info[:, 12]))  # extract and store batch max tile
        all_info_line = np.append(all_info_line, np.average(info[:, 14]))  # extract and average trace length
        all_info_line = np.append(all_info_line, np.average(info[:, 15]))  # extract and average reward
        all_info_line = np.concatenate([all_info_line, np.average(info[:, 16:28], axis=0)])
        # extract and average boolean board distribution

        return all_info_line

    elif header:

        info_header = "0,2,4,8,16,32,64,128,256,512,1024,2048,max tiles average," \
                  "board sum average,max tile for batch,average trace length,average reward," \
                      "0,2,4,8,16,32,64,128,256,512,1024,2048"

        return info_header


def format_files():
    """
    Method to reformat the output files so they can be directly copied into excel
    """

    weight_file = open(file_prepend + "weights.txt")
    lines = weight_file.readlines()
    weight_file.close()

    # insert the batch number at the beginning of the line
    for i in range(len(lines)):
        line = lines[i]
        line = line.split(",")
        line.insert(0, str(i + 1))
        line = ",".join(line)
        lines[i] = line

    # insert top line detailing what the weights represent
    if feature_representation == "vector":
        if algorithm == "mc_v":
            lines.insert(0, "batches,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16\n")
        else:
            line = "batches"
            for i in range(4):
                line = line + ",1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16"
            line = line + "\n"
            lines.insert(0, line)
    else:
        top_line = "batches"
        if algorithm == "mc_q":
            num = 64
        else:
            num = 16
        for i in range(num):
            top_line = top_line + ",0,2,4,8,16,32,64,128,256,512,1024,2048"
        top_line = top_line + "\n"
        lines.insert(0, top_line)

    weight_file = open(file_prepend + "weights_formatted.txt", "w")
    weight_file.writelines(lines)
    weight_file.close()

    # reformat performance information
    # first read the 3 relevant files
    setup_file = open(file_prepend + "setup.txt", "r")
    setup_lines = setup_file.readlines()
    setup_lines.insert(0, "Setup,\n")
    setup_file.close()

    pi_file = open(file_prepend + "performance_info.txt", "r")  # performance information file
    pi_lines = pi_file.readlines()
    pi_file.close()

    pie_file = open(file_prepend + "performance_info_evaluation.txt", "r")  # performance information evaluation file
    pie_lines = pie_file.readlines()
    pie_file.close()

    pif_lines = []  # performance information formatted lines
    for i in range(len(pi_lines) + len(pie_lines) + 1):
        if i < len(setup_lines):
            line = setup_lines[i].strip("\n")
        else:
            line = ","

        if i < len(pi_lines):
            if i == 0:
                line = line + ",,Batch," + pi_lines[i]
            else:
                line = line + ",," + str(i) + "," + pi_lines[i]
        elif i == len(pi_lines):
            line = "\n"
        elif i == len(pi_lines) + 1:
            line = line + ",,Weights," + pie_lines[i - len(pi_lines) - 1]
        else:
            line = line + ",," + str(i - len(pi_lines) - 1) + "," + pie_lines[i - len(pi_lines) - 1]
        pif_lines.append(line)

    pif_file = open(file_prepend + "performance_info_formatted.txt", "w")
    pif_file.writelines(pif_lines)
    pif_file.close()


def evaluate_manual_weights():
    # following is a set of weights where the corners are highest valued but it's roughly symmetrical
    # it performs pretty badly. Uncomment to try.
    # weights = [[0.91, 0.751, 0.752, 0.92], [0.753, 0.51, 0.52, 0.75],
    #            [0.749, 0.49, 0.5, 0.748], [0.89, 0.748, 0.747, 0.9]]

    # following are weights that encourage tiles on the bottom row especially on the right hand side.
    # works better than the one above average max tile of 148 after 100 runs.
    # weights = [[0.1, 0.12, 0.13, 0.2], [0.2, 0.3, 0.4, 0.5],
    #            [0.3, 0.4, 0.5, 0.6], [0.6, 0.7, 0.8, 0.9]]

    # snake pattern. better than first worse than second. average max tile 99 after 100 runs. 
    weights = [[0.4, 0.3, 0.2, 0.1], [0.5, 0.6, 0.7, 0.8],
               [1.2, 1.1, 1.0, 0.9], [1.3, 1.4, 1.5, 1.6]]

    weights = np.array(weights)
    weights = weights.flatten()
    weights = weights.reshape([1, 16])
    evaluate_policies(np.array([weights]), 1, 100, algorithm="mc_v", representation="vector")
    exit()

# run the actual code from here
# specify the parameters you want to run the test

# create game instance for use in learning process
game = Game2048()

# use a text file to store the experiment number that you're on so the data from each
# execution of the code is identifiable. Load the number and store it then increment it
# by one and save file again. The boolean record specifies whether to use this as you
# may want to run this but not store the data afterwards so don't want the number to
# increment, in this case set record as false (useful for debugging)
record = False

if record:
    with open("experiment_number.txt") as exp_num_file:
        experiment_number = exp_num_file.read()
        exp_num_file.close()

    with open("experiment_number.txt", "w") as exp_num_file:
        exp_num_int = int(experiment_number) + 1
        exp_num_file.write(str(exp_num_int))
        exp_num_file.close()

    file_prepend = str(experiment_number) + "-"
else:
    experiment_number = 0
    file_prepend = ""

# create file to store the setup information for this run of the code
setup_file = open("%ssetup.txt" % file_prepend, "w")

# add experiment number to the setup file if it's being used
if record:
    setup_file.write("experiment number,%s\n" % experiment_number)

# pick a representation, either "vector" or "binary":
feature_representation = "vector"
setup_file.write("representation,%s\n" % feature_representation)

# pick a learning algorithm and value function type, either "mc_v" (monte carlo value function learning)
# or "mc_q" (monte carlo q learning):
algorithm = "mc_q"
setup_file.write("algorithm,%s\n" % algorithm)

# specify the number of episodes and batches to be performed for monte carlo:
batches = 30
episodes = 100
setup_file.write("batches,%d\nepisodes,%d\n" % (batches, episodes))

# specify epsilon and the learning rate:
epsilon = 0.01
alpha = 0.05
taper_epsilon = False
setup_file.write("epsilon,%s\nalpha,%s\ntaper epsilon,%s\n" % (str(epsilon), str(alpha), str(taper_epsilon)))

# specify what type of cumulative rewards to use out of: "normal", "reverse" or "none":
cumulative = "normal"
setup_file.write("cumulative rewards,%s\n" % str(cumulative))

# specify the number of evaluation weight sets to look at and the number of
# episodes to evaluate them with:
eval_no_weights = 5
eval_episodes = 100
setup_file.write("number of evaluation weights,%d\nevaluation episodes,%d\n" % (eval_no_weights, eval_episodes))


# start timing block to see how long learning takes
start_time = time.time()

# execute learning function using the parameters set above:
# all_weights = learn(feature_representation, algorithm, batches=batches, episodes=episodes,
#                     alpha=alpha, epsilon=epsilon, cumulative=cumulative, taper_epsilon=taper_epsilon)
# evaluate_policies(all_weights, eval_no_weights, eval_episodes,
#                   algorithm=algorithm, representation=feature_representation)
evaluate_manual_weights()
print("elapsed time: " + str(time.time() - start_time))
setup_file.write("time taken,%s\n" % str(time.time() - start_time))
setup_file.close()
format_files()

