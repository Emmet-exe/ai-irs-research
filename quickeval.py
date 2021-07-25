import neuralnet
import dataset
import sys
from config import *
import numpy

# evaluate a model on a different dataset
def eval(name):
    try:
        nn = neuralnet.load_nn(name)
    except Exception as e:
        print(name + " is not saved as a model.")
        return

    startYear = 2014
    dataset.load_dataset(startYear, YEAR_MAX - startYear + 1)    
    eins = list(dataset.get_eins(startYear, DATA_SPAN))

    correct = 0
    percentErr = 0
    step = 50
    for i in range(0, len(eins), step):
        ein = eins[i]
        try:
            currentinput, label = dataset.ein_to_training_data(ein, startYear)
            prediction = neuralnet.predict(nn, numpy.array([currentinput]))
            if label * prediction > 0:
                correct += 1
            percentErr += abs((prediction + label) / label - 1)
        except:
            continue 
    
    total = len(eins) / step
    percentErr /= total
    print("Growth/Decline Success Rate:", correct / total)
    print("Average % Error:", percentErr)
    print("[\u2713] Evaluated network " + name)

if len(sys.argv) > 1:
    networkName = sys.argv[1]
    eval(networkName)
else:
    print("Please provide a network name for evaluation")