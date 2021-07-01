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

def year_analysis(nnid):
    # setup the pyplot
    pyplot.title("Years in Existence vs. Growth")
    pyplot.xlabel('Years since founding')
    pyplot.ylabel('Predicted Growth')
    
    # Load the neural net model
    nn = None
    try:
        nn = neuralnet.load_nn(nnid)
    except:
        print(nnid + " is not saved as a model.")
        return

    i = 0
    while i < 1:
        print(i)
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
            for j in range(0, 400, 5):
                x.append(j)
                inputs = []
                orgInputs[-1] = j
                inputs.append(orgInputs)
                y.append(neuralnet.predict(nn, np.array(inputs)))

            pyplot.scatter(x, y)
            pyplot.plot(x, plotting.line_fit(x, y))
            i += 1
        except Exception as e:
            print(e)
            continue
    pyplot.show()

# Network id is passed as cmd line argument
if len(sys.argv) > 1:
    networkName = sys.argv[1]
    year_analysis(networkName)
else:
    print("Please pass a network name for analysis")