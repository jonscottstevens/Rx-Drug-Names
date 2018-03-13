# Rx-Drug-Names

This is a fun exercise I made for myself to generate hypothetical brand names for prescription drugs starting with a list of existing drug names and an LSTM (long short-term memory) neural net.  This was implemented in Python 2.7 using the Keras package.  The implementation was inspired by Jason Brownlee's neural net for generating text from Alice in Wonderland, found [here](https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/).

## Data

I started with the data in drug_info.txt, which is simply a copy-paste from [this website](http://www.rxassist.org/pap-info/brand-drug-list-print) listing drug names and information.  The script extract_names.py extracts only the names of the drugs on the list, which it stores in the file drug_names.txt.  The format of the data file is a list of names each separated by the character '#', which denotes the beginning/end of a word.

## Model

The script generate_names.py is where the action is.  An LSTM model processes the data as a time series of 10-character strings, and tries to predict the 11th character.  Thus, the training looks like like this:

| Input        | Output           |
| ------------- |:-------------:|
| #iclusig#a      | c |
| iclusig#ac      | z      |
| clusig#acz | o      |
| lusig#aczo | n      |
| usig#aczon | e      |
| sig#aczone | #      |

From the trained model we can generate predictions of arbitrary length to follow any given string from the input data, from which complete names can be extracted.  If the script is run as-is, it will do this until there are 100 acceptable candidates, and then print those 100 candidates.  What is acceptable is determined by a couple of post-hoc filters.

## Output filters

The neural net has a thing for double letters, for some reason, and thus I filter the output to exclude names with double letters.  I also filter the output to exclude existing drug names from the input data as well as duplicate generated names.  Finally, I implement a filter that excludes candidates whose consonant-to-vowel ratio is not strictly between 0.5 and 2.  This filters out many of the phonologically impossible candidates.

## Sample output

The file 100_rx_drug_names.txt gives the output of my first complete run of the final version of the script.