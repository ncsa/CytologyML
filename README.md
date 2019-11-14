# CytologyML
A classification engine for cytology data

# Objective:
We aim to create a machine learning pipeline using Python’s ML modules to perform cell classification of flow cytometry data. We produce a system that can replace manual gating of flow cytometry studies (the process by which cells analysed in a flow cytometer are assigned increasingly specific cellular identities).

# Steps:
-       Clone the repository and import the necessary modules\ 
-       Edit paths in the json to point to the appropriate files/folders. You may have to create paths to the models and metrics folders, and the evaluation_matrix.txt file in the metrics folder\
-       Run the script with the –O flag for debugging, -m for the method being used (this flag is only used in deployment.py), and –j for the json file\
-       Use the test files while running training.py and use the studies while running deployment.py\
 
# Organization of the Code:
The code is broken up into 2 main files: 
(1) training.py - we check the integrity of the data and set up our classification modules. We save our trained models for further use.\
(2) deployment.py – we classify cells into smaller populations using the different classification methods (only the method input in command line with the –m flag will ultimately be used)



