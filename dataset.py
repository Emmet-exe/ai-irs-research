import csv
import json
import utils
import datetime
from config import *
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
    
    # net assets vs liabilities, as of prior year (line 22)
    orgInput.append(int(org[90]))
    # contributions, prior year (line 13)
    orgInput.append(utils.zeroint(org[72]))
    # years in existence
    yearsSince = years_in_existence(org)
    if yearsSince < 0:
        raise Exception("Cannot have existed for negative years") 
    orgInput.append(yearsSince)
    
    return orgInput, label

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