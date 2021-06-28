import csv
import json
import random
import utils
import neuralnet
import time
import datetime
import numpy
from config import *
from collections import OrderedDict

# DATASET DICTIONARY
dataset = OrderedDict()
for year in range(YEAR_MIN, YEAR_MAX + 1):
    dataset[year] = {}

# Load the csv files for [span] years starting with yearMin
def load_dataset(yMin, span = 1):
    for year in range(yMin, yMin + span):
        f = open("irs990_main/irs990_main_" + str(year) + ".csv")
        csv_reader = csv.reader(f, delimiter=",")
        line_count = 0
        for row in csv_reader:
            if (line_count) != 0:
                dataset[year][row[0]] = row[1:]
            line_count += 1
        f.close()
    print("[\u2713] Dataset loaded - Start:", yMin, "- Span:", span)

# Create a set of viable eins for a given set of years
def get_eins(yMin, span):
    for year in range(yMin, yMin + span):
        if (len(dataset[year]) == 0):
            load_dataset(yMin, span)

    eins = dataset[yMin].keys()
    trainingEINs = set()
    for company in eins:
        valid = True
        for year in range(yMin, yMin + span):
            # Check the company is a 501c(3) corporation and that there are records for all years in the range
            if not company in dataset[year] or (dataset[year][company][42] != "true" and dataset[year][company][41] != "3") or dataset[year][company][46] != "true": 
                valid = False
        if valid:
            trainingEINs.add(company)

    return trainingEINs

    # f = open(TRAINING_EIN_FILE + "2", "w")
    # f.write(json.dumps(trainingEINs, default = utils.serialize_set))
    # f.close()
    # print("[\u2713] Wrote", len(trainingEINs), "EINs to " + TRAINING_EIN_FILE)

# load the training eins from the file
# def load_training_eins():
#     try:
#         f = open(TRAINING_EIN_FILE, "r")
#         einSet = set(json.loads(f.read()))
#         f.close()
#         if len(einSet) == 0:
#             return 0
#         print("[\u2713] Read", len(einSet), "EINs from " + TRAINING_EIN_FILE)
#         return einSet
#     except:
#         return 0

# generate and organize the training dataset 
# Dataset span is 4 years
def generate_dataset(startingYear):
    eins = get_eins(startingYear, DATA_SPAN)
    
    trainingData = {
        "inputs": [],
        "labels": []
    }

    for ein in eins:
        try:
            orgInput, label = ein_to_training_data(ein, startingYear)
            trainingData["inputs"].append(orgInput)
            trainingData["labels"].append(label)
        except:
            continue
        
    f = open(TRAINING_DATA_FILE + str(startingYear) + ".json", "w")
    f.write(json.dumps(trainingData))
    f.close()
    print("[\u2713] Serialized training data (Start year: " + str(startingYear) +  ", Batch size: " + str(len(trainingData["labels"])) + ")")

# WILL THROW EXCEPTION if dataset is not properly loaded
def ein_to_training_data(ein, startingYear):
    finalYear = startingYear + DATA_SPAN - 1
    org = dataset[finalYear][ein]
    orgInput = []
    # grants and similar amounts paid + growth in net assets, current year
    label = int(org[91]) - int(org[90]) + utils.zeroint(org[71])
    # label = int(org[91]) + utils.zeroint(org[71])
    for year in range(startingYear, finalYear + 1):
        current = dataset[year][ein]
        # contributions for past 4 years (line 8)
        orgInput.append(utils.zeroint(current[62]))
        # program revenue + other revenue for past 4 years (line 9 + 11)
        orgInput.append(utils.zeroint(current[64]) + utils.zeroint(current[68]))
        # investment income past 4 years (line 10)
        orgInput.append(utils.zeroint(current[66]))
        # total expenses minus contributions for last 4 years (line 18 - 13)
        orgInput.append(utils.zeroint(current[83]) - utils.zeroint(current[72]))
    
    # net assets vs liabilities, as of prior year (line 22)
    orgInput.append(int(org[90]))
    # contributions, prior year (line 13)
    orgInput.append(utils.zeroint(org[72]))
    # years in existence
    orgInput.append(datetime.datetime.now().year - int(org[50]))
    
    return orgInput, label

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
        generate_dataset(startingYear)
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
    eins = get_eins(startYear, 4)
    i, j = 0, 10
    while i < j:
        try:
            ein = random.choice(tuple(eins))
            currentinput, label = ein_to_training_data(ein, startYear)
            inputArr = []
            inputArr.append(currentinput)
            prediction = neuralnet.predict(nn, numpy.array(inputArr))[0].item()
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

# train()
test("27062021-124600")