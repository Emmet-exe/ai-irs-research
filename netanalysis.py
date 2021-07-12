import dataset
import plotting
from config import *
from matplotlib import pyplot
import random
import math
import sys
import neuralnet
import utils
import numpy as np

x, y = [], []
dataset.load_dataset(YEAR_MIN, YEAR_MAX - YEAR_MIN + 1)
nn = None

def year_analysis():
    # setup the pyplot
    pyplot.title("Years in Existence vs. Growth")
    pyplot.xlabel('Years since founding')
    pyplot.ylabel('Predicted Growth')

    i = 0
    while i < 1:
        ein = year = existed = -1
        org = []
        while existed < 0:
            year = random.choice(range(YEAR_MIN, YEAR_MAX + 1))
            ein = random.choice(list(dataset.all_eins(year)))
            org = dataset.get_org(year, ein)
            existed = dataset.years_in_existence(org)

        orgInputs = label = None
        try:
            orgInputs, label = dataset.ein_to_training_data(ein, year)
            step = math.ceil(existed / 10.0)
            if step == 0:
                step += 1
            
            x, y = [], []
            # Make a prediction
            for j in range(0, 50, 2):
                x.append(j)
                inputs = []
                orgInputs[-1] = j
                inputs.append(orgInputs)
                y.append(neuralnet.activation_inverse(neuralnet.predict(nn, np.array(inputs))))

            pyplot.scatter(x, y)
            pyplot.plot(x, plotting.line_fit(x, y))
            i += 1
        except Exception as e:
            continue
    pyplot.show()

def decline_analysis():
    inputs, labels = dataset.all_training_data()
    declineCount, predictionCount, successCount = 0, 0, 0
    for i in range(0, len(inputs), 50):
        prediction = neuralnet.predict(nn, np.array([inputs[i]]))
        if prediction < 0:
            predictionCount += 1
        if labels[i] < 0:
            declineCount += 1
            if prediction < 0:
                successCount += 1

    # setup the pyplot
    xLabels = ["Total # of Declining Orgs (" + utils.strround(declineCount / (len(inputs) / 50.0) * 100) + "% of total)", "Total Predicted Declining Orgs", "Successful Predictions (" + utils.strround(successCount / float(declineCount) * 100) + "% Success with " + utils.strround(successCount / float(predictionCount) * 100) + "% Accuracy)"]
    xVals = [declineCount, predictionCount, successCount]
    xPos = [i for i, _ in enumerate(xLabels)]

    pyplot.title("Identifying Declining Organizations")
    pyplot.bar(xPos, xVals, color=MAIN_COLOR)
    pyplot.ylabel("# of Orgs")
    pyplot.xticks(xPos, xLabels)
    pyplot.show()

# Network id is passed as cmd line argument
if len(sys.argv) > 1:
    nnid = sys.argv[1]
    # Load the neural net model
    try:
        nn = neuralnet.load_nn(nnid)
        for arg in sys.argv[2:]:
            if arg == "--year":
                year_analysis()
            elif arg == "--decline":
                decline_analysis()
    except:
        print(nnid + " is not saved as a model.")
    
    

else:
    print("Please pass a network name for analysis")
