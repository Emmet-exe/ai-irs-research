import csv
import json
import utils
import datetime
import numpy
from config import *
from neuralnet import scalar_activation
from collections import OrderedDict

# DATASET DICTIONARY
dataset = OrderedDict()
for year in range(YEAR_MIN, YEAR_MAX + 1):
    dataset[year] = {}

# Load the csv files for [span) years starting with yearMin
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

# list of EINs in a certain year
def all_eins(year):
    return dataset[year].keys()

# Create a set of viable eins for a given set of years
def get_eins(yMin, span):
    for year in range(yMin, yMin + span):
        if (len(dataset[year]) == 0):
            load_dataset(yMin, span)

    eins = all_eins(yMin)
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
            if (label >= 10000):
                trainingData["inputs"].append(orgInput)
                # trainingData["labels"].append(scalar_activation(label))
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
    org = get_org(finalYear, ein)
    orgInput = []
    
    label = get_label(org)
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
        # net assets vs liabilities for last 4 years (line 22)
        orgInput.append(int(current[90]))
    
    # contributions, prior year (line 13)
    orgInput.append(utils.zeroint(org[72]))
    # years in existence
    yearsSince = years_in_existence(org)
    if yearsSince < 0:
        raise Exception("Cannot have existed for negative years") 
    orgInput.append(yearsSince)
    
    return orgInput, label


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
        generate_dataset(startingYear)
        return load_training_data(startingYear)


def all_training_data():
    inputs, labels = firstHalf = load_training_data(YEAR_MIN)
    inputs2, labels2 = secondHalf = load_training_data(YEAR_MIN + DATA_SPAN)
    inputs = numpy.concatenate((inputs, inputs2))
    labels = numpy.concatenate((labels, labels2))
    return inputs, labels

def get_label(org):
    # grants and similar amounts paid + growth in net assets, current year
    return utils.zeroint(org[91]) - utils.zeroint(org[90]) + utils.zeroint(org[71])

def years_in_existence(org):
    try:
        years = datetime.datetime.now().year - int(org[50])
        if years > 0:
            return years
        return -1
    except:
        return -1

def get_org(year, ein):
    return dataset[year][ein]

def dataset_obj():
    return dataset