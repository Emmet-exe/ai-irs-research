import sys
import os

# Just a file to delete all of the logs for a network - name of model is first command line argument
networkName = sys.argv[1]
if os.path.exists("eval_logs/" + networkName + ".txt"):
    os.remove("eval_logs/" + networkName + ".txt")
if os.path.exists("models/" + networkName + ".h5"):
    os.remove("models/" + networkName + ".h5")
if os.path.exists("training_logs/" + networkName + ".txt"):
    os.remove("training_logs/" + networkName + ".txt")