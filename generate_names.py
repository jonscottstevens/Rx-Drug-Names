"""
A neural network for generating prescription drug names
by Jon Stevens
github.com/jonscottstevens
"""

from __future__ import division
import os
import sys
import re
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM

"""
Load the data file.  The data file consists of drug names separated by special character '#'
"""

os.chdir(os.path.dirname(os.path.abspath(__file__)))
filename = "drug_names.txt"
raw_text = open(filename).read()
raw_text = raw_text.lower()


"""
We need to featurize our data.  We'll use a time series approach, carving our data up into sequences of 10-character strings that predict the 11th
"""

alphabet = 'abcdefghijklmnopqrstuvwxyz-.#'
char_to_int = dict((c, i) for i, c in enumerate(alphabet))
int_to_char = dict((i, c) for i, c in enumerate(alphabet))

n_chars = len(raw_text)

seq_length = 10
dataX = []
dataY = []
for i in range(0, n_chars - seq_length):
	seq_in = raw_text[i:i + seq_length]
	seq_out = raw_text[i + seq_length]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(char_to_int[seq_out])

# X is the input data (time series of 10-character strings)
X = np.reshape(dataX, (len(dataX), seq_length, 1))

# y is the output data (the 11th character to be predicted from the preceding 10)
y = np_utils.to_categorical(dataY)

"""
Now we define an LSTM model and fit it to the data.
"""

model = Sequential()
model.add(LSTM(200, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

model.fit(X, y, epochs=50, batch_size=100)

"""
Now we define some filters on the output data to increase the realism of our drug names
"""

# this regexp matches double letters... you get more realistic results if you filter out double letters
regexp = re.compile(r"(.)\1")

# define what consonants and vowels are
vowl = 'aeiou'
cons = 'bcdfghjklmnpqrstvwxz'

# function below filters results first by double letters then by consonant-to-vowel ratio
def realistic(word):
    if re.search(regexp, generated_name):
        return False
    else:
        cv_ratio = len([char for char in word if char in cons]) / len([char for char in word if char in vowl]) + 0.001
        if cv_ratio >= 2:
            return False
        if cv_ratio <= 0.5:
            return False
        else:
            return True

"""
Below is the code for generating 100 names from the model.
"""

generated_names = []

while len(generated_names) < 100:
    sequence = list(dataX[np.random.randint(0, len(dataX)-1)])
    result = ''
    # the for loop starts with a random data point from X, then predicts 25 characters to follow it
    for i in range(25):
        x = np.reshape(sequence, (1, len(sequence), 1))
        prediction = model.predict(x, verbose=0)
        index = np.argmax(prediction)
        character = int_to_char[index]
        result = result + character
        sequence.append(index)
        sequence = sequence[1:len(sequence)]
    # extract the first complete word from the 25-character sequence
    generated_name = result.split('#')[1]
    """
    filter words by:
    (i) whether they are already in our data,
    (ii) whether they have already been generated, and
    (iii) whether they are realistic
    """
    if generated_name not in raw_text:
        if generated_name not in generated_names:
            if realistic(generated_name):
                generated_names.append(generated_name)

generated_names = [name[0].upper() + name[1:len(name)] for name in generated_names]

for name in generated_names:
    print name + '\n'
