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
    percentChange = 0
    step = 1
    total = 0
    for i in range(0, len(eins), step):
        ein = eins[i]
        try:
            currentinput, label = dataset.ein_to_training_data(ein, startYear)
            if abs(label) >= INSIG_CONST:
                prediction = neuralnet.predict(nn, numpy.array([currentinput]))                
                prev = currentinput[-3] + currentinput[-7]
                pctErr = abs((prev + prediction) / (prev + label) - 1)
                if pctErr < 1000:
                    total += 1
                    percentErr += pctErr
                    percentChange += abs((prev + label) / prev - 1)
                    if numpy.sign(label) * numpy.sign(prediction) > 0:
                        correct += 1
            if i % 1000 == 0:
                print(i, len(eins))
        except:
            continue 
    
    percentErr /= total
    percentChange /= total
    print("Growth/Decline Success Rate:", correct / total)
    print("Average % Error:", percentErr * 100)
    print("Average % Change:", percentChange * 100)
    print("[\u2713] Evaluated network " + name)

if len(sys.argv) > 1:
    networkName = sys.argv[1]
    eval(networkName)
else:
    print("Please provide a network name for evaluation")