from tensorflow.keras.callbacks import ModelCheckpoint
import os
import numpy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
import re
import numpy as np
import sys

SEQUENCE_LENGTH = 40
EPOCHS = 30
BATCH_SIZE = 64
EPS = 1e-6


# Example input = "testText.txt"
def clear_file(input):
    def isCorrectChar(c):
        return c.isspace() or c == '”' or c == '\'' or (c.isalpha() and (c not in allowed))

    allowed = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
               'v', 'w', 'x', 'y', 'z', 'ê', 'ê', 'ê']
    inputFile = os.path.join("results", f'{input}.txt')
    outputFile = os.path.join("results", f'___clear___{input}.txt')

    if os.path.exists(outputFile):
        print(f'WARNING: {outputFile} is exist!!!', flush=True)
        return outputFile

    print(f'opening on clearing file {inputFile}', flush=True)
    text = open(inputFile).read()
    text = text.lower()
    text = "".join(list(filter(isCorrectChar, text)))
    text = re.sub('\n+', '\n', re.sub('\n ', '\n', re.sub(' +', ' ', text)))

    print(f'writing cleared file {outputFile}', flush=True)
    open(outputFile, "w").write(text)

    return outputFile


def init_data(file):
    raw_text = open(file).read()

    chars = sorted(list(set(raw_text)))
    chars_int_map = dict((c, i) for i, c in enumerate(chars))
    int_chars_map = dict((i, c) for i, c in enumerate(chars))
    amount_chars, amount_different_chars = len(raw_text), len(chars)

    x_arr_dataset_custom_tmp, y_arr_dataset_custom_tmp = [], []
    for i in range(amount_chars - SEQUENCE_LENGTH):
        sequence_from, char_out = raw_text[i:i + SEQUENCE_LENGTH], raw_text[i + SEQUENCE_LENGTH]
        x_arr_tmp = list(map(lambda char: chars_int_map[char], sequence_from))
        x_arr_dataset_custom_tmp.append(x_arr_tmp)
        y_arr_dataset_custom_tmp.append(chars_int_map[char_out])

    x_arr_dataset = numpy.reshape(x_arr_dataset_custom_tmp, (len(x_arr_dataset_custom_tmp), SEQUENCE_LENGTH, 1))
    x_arr_dataset = x_arr_dataset / float(amount_different_chars)
    y_arr_dataset = to_categorical(y_arr_dataset_custom_tmp)

    return x_arr_dataset, y_arr_dataset, x_arr_dataset_custom_tmp, int_chars_map


def init_model(x_arr_dataset, y_arr_dataset):
    sgd = SGD(learning_rate=0.1, decay=1e-6, momentum=0.9, nesterov=True)

    model = Sequential()
    
    model.add(LSTM(256, input_shape=(x_arr_dataset.shape[1], x_arr_dataset.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(y_arr_dataset.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    return model


def teach_model(file, teached_model_folder):
    print("Start teaching model", flush=True)

    x_arr_dataset, y_arr_dataset, _, _ = init_data(file)

    model = init_model(x_arr_dataset, y_arr_dataset)

    filepath = f'results/{teached_model_folder}/' + "epoch_{epoch:02d}__loss_{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    model.fit(x_arr_dataset, y_arr_dataset, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks_list)
    print("End teaching model", flush=True)


def generate_text_RNN(file, teached_model_file_path, amount_sequence=100):
    print(f'Start generating text under teached model on {teached_model_file_path}', flush=True)

    x_arr_dataset, y_arr_dataset, x_arr_dataset_custom_tmp, int_chars_map = init_data(file)
    amount_different_chars = len(int_chars_map)

    model = init_model(x_arr_dataset, y_arr_dataset)
    #model.load_weights(teached_model_file_path)

    start_sequence_id = numpy.random.randint(0, len(x_arr_dataset_custom_tmp) - 1)
    start_sequence = x_arr_dataset_custom_tmp[start_sequence_id]
    sequence_from = "".join([int_chars_map[value] for value in start_sequence])
    print(f'Start phrase:\n{sequence_from}', flush=True)
    print("Generating:", flush=True)
    for i in range(amount_sequence):
        x = numpy.reshape(start_sequence, (1, len(start_sequence), 1))
        x = x / float(amount_different_chars)
        prediction = model.predict(x, verbose=0)
        index = numpy.argmax(prediction)
        result = int_chars_map[index]
        sys.stdout.write(result)
        start_sequence.append(index)
        start_sequence = start_sequence[1:len(start_sequence)]
    print(f'\nEnd generating text', flush=True)


def do_mark(file, n, k, m):
    print(f'Start generating text for {file}', flush=True)
    text = open(file).read()
    windows = get_windows(text, n)
    graph, w_to_int, int_to_w = get_graph(windows, text, n)
    start_line = get_random_start(text)
    generating_by_graph(n, k, m, start_line, graph, w_to_int, int_to_w)


def get_windows(text, n):
    windows = set()
    for i in range(len(text) - n + 1):
        windows.add(text[i:i + n])
    print(f'Windows: {len(windows)}', flush=True)
    return windows


def get_graph(windows, text, n):
    window_to_int = dict((c, i) for i, c in enumerate(windows))
    int_to_wwindow = dict((i, c) for i, c in enumerate(windows))
    matrix = [[0 for _ in range(len(windows))] for _ in range(len(windows))]
    for i in range(len(text) - n):
        cur_w = text[i:i + n]
        next_w = text[(i + 1):(i + n + 1)]
        matrix[window_to_int[cur_w]][window_to_int[next_w]] += 1
    matrix = np.array(norm_matrix(matrix))
    return matrix, window_to_int, int_to_wwindow


def get_random_start(text):
    lines = text.split('\n')
    start = np.random.randint(0, len(lines) - 1)
    return lines[start]


# K have to more M
def generating_by_graph(n, k, m, start_line, matrix, w_to_int, int_to_w):
    prefix = start_line[0:k]
    print(f'Start phrase:\n{prefix}', flush=True)
    print("Generating:", flush=True)
    start_window = prefix[len(prefix) - n:]
    for i in range(m):
        sug_next_pos = get_all_by_max(matrix[w_to_int[start_window]])
        if len(sug_next_pos) == 0:
            print(f'Can\'t continue', flush=True)
            break
        elif len(sug_next_pos) == 1:
            ind = 0
        else:
            ind = np.random.randint(0, len(sug_next_pos) - 1)
        next_pos = sug_next_pos[ind]
        start_window = int_to_w[next_pos]
        sys.stdout.write(start_window[len(start_window) - 1])
    print('\nEnd generating text', flush=True)


def get_all_by_max(data):
    max_v = max(data)
    res = []
    for i in range(len(data)):
        if abs(data[i] - max_v) < EPS:
            res.append(i)
    return res


def norm_matrix(matrix):
    new_matrix = []
    for row in matrix:
        sum_v = sum(row)
        if sum_v != 0:
            new_row = list(map(lambda x: x / sum_v, row))
            new_matrix.append(new_row)
        else:
            new_matrix.append(row)
    return new_matrix


if __name__ == '__main__':
    # raw_file = "testText"
    # teached_model_folder = "testFolder"
    # teached_model = "epoch_30__loss_3.1055.hdf5"

    # raw_file = "evgeny_onegin"
    # teached_model_folder = "evgeny"
    # teached_model = "epoch_30__loss_2.4439.hdf5"

    raw_file = "prestuplenie_i_nakazanie"
    teached_model_folder = "radion"
    teached_model = "epoch_30__loss_1.8504.hdf5"

    teached_model_file_path = f'results/{teached_model_folder}/{teached_model}'

    cleared_file = clear_file(raw_file)
    teach_model(cleared_file, teached_model_folder)

    generate_text_RNN(cleared_file, teached_model_file_path, amount_sequence=240)

    do_mark(cleared_file, 4, 10, 200)
