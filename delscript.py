import sys
import os

networkName = sys.argv[1]
if os.path.exists("eval_logs/" + networkName + ".txt"):
    os.remove("eval_logs/" + networkName + ".txt")
if os.path.exists("models/" + networkName + ".h5"):
    os.remove("models/" + networkName + ".h5")
if os.path.exists("training_logs/" + networkName + ".txt"):
    os.remove("training_logs/" + networkName + ".txt")