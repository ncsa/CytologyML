# CytologyML
A classification engine for cytology data

# Objective:
We aim to create a machine learning pipeline using Python’s ML modules to perform cell classification of flow cytometry data. We produce a system that can replace manual gating of flow cytometry studies (the process by which cells analysed in a flow cytometer are assigned increasingly specific cellular identities).

# Steps:
-       Clone the repository and import the necessary Python Libraries\
        - Keras, Tensorflow, Sklearn, Pandas, Numpy
        - Requires Python version 3.7.0
-       Edit paths in the json to point to the appropriate files/folders. You may have to create paths to the models and metrics folders, and the evaluation_matrix.txt file in the metrics folder\
-       Run the script with the –O flag for debugging, -m for the method being used (this flag is only used in deployment.py), and –j for the json file\
-       Use the test files while running training.py and use the studies while running deployment.py\

# Organization of the Code:
The code is broken up into 2 main files:

(1) training.py - we check the integrity of the data and set up our classification modules. We save our trained models for further use.\
(2) deployment.py – we classify cells into smaller populations using the different classification methods (only the method input in command line with the –m flag will ultimately be used)


# How to Run the Code:
Without debugging:\
> python3 deployment.py -m <method> -j <path_to_json>\
ex. python3 deployment.py -m y_and -j /Users/Home/file.json\

With debugging:\
> python3 -O training.py -j <path_to_json>

Can also be run with the help command, -h

Running the code with debugging mode requires a separate JSON with paths to all the files in addition to metrics and models folders (this JSON will resemble the JSON used for training.py). The JSON used for running without Debug mode has a select few files: a path to all the files (ex. `All.csv`), path to Metrics folder, and the path to the Models folder.
