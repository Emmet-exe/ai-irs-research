import json
import random
import utils
import neuralnet
import time
import numpy
import sys
import dataset
from config import *

# load the dataset from file, or generate it if it doesn't exist
def load_training_data(startingYear):
    try:
        f = open(TRAINING_DATA_FILE + str(startingYear) + ".json", "r")
        trainingData = json.loads(f.read())
        f.close()
        inputs = numpy.array(trainingData["inputs"])
        labels = numpy.array(trainingData["labels"])
        print("[\u2713] Loaded training data (Start year: " + str(startingYear) +  ", Batch size: " + str(len(trainingData["labels"])) + ")")
        return inputs, labels
    except:
        dataset.generate_dataset(startingYear)
        return load_training_data(startingYear)

# Generate a unique neural net id for later
def gen_nnid():
    return time.strftime("%d%m%Y-%H%M%S")

# run an experiment
def train(seed = 0):
    # create a unique neural net id
    nnid = gen_nnid()

    # load up the training dataset
    inputs, labels = load_training_data(2010)
    
    # generate the network
    neuralNet = neuralnet.createNN(seed, len(inputs[0]))
    # train the network
    history = neuralnet.train(neuralNet, nnid, inputs, labels)
    neuralnet.plotTraining(history)

# evaluate a model on a different dataset
def test(name):
    try:
        nn = neuralnet.load_nn(name)
    except:
        print(name + " is not saved as a model.")
        return
    startYear = 2014
    
    inputs, labels = load_training_data(startYear)
    ft = open("training_logs/" + name + ".txt", "r")
    training_log = ft.read()
    ft.close()
    loss = neuralnet.evaluate(nn, inputs, labels)
    f = open("eval_logs/" + name + ".txt", "w")
    f.write("----- Evaluation of Neural Network #" + name + " -----\n")
    f.write("Absolute Evaluation Loss: " + str(loss) + "\n")
    
    f.write("\nExample Predictions:\n")
    eins = dataset.get_eins(startYear, 4)
    i, j = 0, 10
    while i < j:
        try:
            ein = random.choice(tuple(eins))
            currentinput, label = dataset.ein_to_training_data(ein, startYear)
            inputArr = []
            inputArr.append(currentinput)
            prediction = neuralnet.predict(nn, numpy.array(inputArr))
            # prevYear = float(currentinput[-2] + currentinput[-3])
            # f.write("Prediction for EIN " + str(ein) + ": " + str(float(prediction)) + " (" + utils.strround((prediction / prevYear - 1) * 100) +  "%) | Label: " + str(label) + " (" + utils.strround((label / prevYear - 1) * 100) + "%)\n")
            f.write("Prediction for EIN " + str(ein) + ": " + utils.strround(float(prediction)) + " | Label: " + utils.strround(label) + "\n")
        except Exception as e:
            j += 1
        i += 1
    
    f.write("\nTraining log for backreference...\n")
    f.write(training_log)
    f.close()
    print("[\u2713] Evaluated network " + name)

if len(sys.argv) > 1:
    networkName = sys.argv[1]
    test(networkName)
else:
    train()