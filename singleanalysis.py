import dataset
from config import *
import sys
import neuralnet
import numpy as np
import utils
import json
import os
import matplotlib.pyplot as plt

# nn = neuralnet.load_nn("15072021-205218")
# nn = neuralnet.load_nn("25072021-144709")
# nn = neuralnet.load_nn("28072021-152316")
nn = neuralnet.load_nn("01082021-193621")
dataset.load_dataset(YEAR_MIN, YEAR_MAX - YEAR_MIN + 1)
NEGATIVE_EIN_FILENAME = "negative_eins.json"

def confident_negative_eins():
    year = 2014
    print("EXTENSIVE PREDICTIONS TO BE MADE TO IDENTIFY EINS - EXPECT LONG WAIT")
    losing_eins = set()
    gaining_count = 0
    eins = dataset.get_eins(year, DATA_SPAN)
    
    # things we analyzing
    # 0) CONTRIBUTIONS
    # 1) OTHER INCOME
    # 2) INVESTMENT INCOME
    # 3) EXPENSES
    # 4) ASSETS OVER TIME
    negative_data = [0] * 5
    positive_data = [0] * 5

    for ein in eins:
        try:
            inputs, label = dataset.ein_to_training_data(ein, year)
            prediction = neuralnet.predict(nn, np.array([inputs]))
            assets = []
            if prediction < 0 and label < 0:
                for i in range(0, 20):
                    if i == 4 or i == 19:
                        assets.append(inputs[i])
                    else:
                        negative_data[i % 5] += inputs[i]
                losing_eins.add(ein)
                negative_data[4] += ((assets[-1] - assets[0]) / (float(len(assets) - 1)))
            elif prediction > 0 and label > 0:
                for i in range(0, 20):
                    if i == 4 or i == 19:
                        assets.append(inputs[i])
                    else:
                        positive_data[i % 5] += inputs[i]
                gaining_count += 1
                positive_data[4] += ((assets[-1] - assets[0]) / (float(len(assets) - 1)))
        except Exception as e:
            continue
    
    negative_data = [x / float(len(losing_eins)) for x in negative_data]
    positive_data = [x / float(gaining_count) for x in positive_data]

    print(negative_data)
    print(positive_data)

    if not os.path.isfile(NEGATIVE_EIN_FILENAME):
        f = open(NEGATIVE_EIN_FILENAME, "w")
        f.write(json.dumps(utils.serialize_set(losing_eins)))
        f.close()

def negative_indicators():
    try:
        f = open(NEGATIVE_EIN_FILENAME, "r")
        f.close()
    except:
        confident_negative_eins()
        negative_indicators()

def inc_map():
    year = 2014
    eins = list(dataset.get_eins(year, DATA_SPAN))

    # setup the pyplot
    Z_MAX = 100
    xCoords, yCoords, z = [], [], []
    mesh = [[[0] for j in range(100)] for i in range(100)]

    # going 10x speed, still a lot of data
    for i in range(0, len(eins), 1):
        try:
            ein = eins[i]
            inputs, label = dataset.ein_to_training_data(ein, year)
            prediction = neuralnet.pctpredict(nn, np.array([inputs]), inputs[-3])
            if abs(prediction) <= Z_MAX:
                currentData = [0]*5
                for i in range(0, 20):
                    if i % 5 != 4 or i == 4:
                        currentData[i % 5] += inputs[i]
                    if i == 19:
                        currentData[4] = inputs[i] - currentData[4]
                # division by zero exception should get caught and the EIN will be thrown out so we should be fine
                totalIncome = float(sum(currentData[0:3]))
                invPct = neuralnet.scalar_activation(currentData[2])
                incExp = totalIncome / float(currentData[3])
                assetSlope = neuralnet.scalar_activation(currentData[4])
                # remove outliers
                if -2 < incExp < 8 and -2 <= assetSlope <= 2:
                    xCoords.append(invPct)
                    yCoords.append(incExp)
                    z.append(prediction)
                    row = int(incExp * 10) + 20
                    col = int(assetSlope * 25) + 50
                    if mesh[row][col][0] == 0:
                        mesh[row][col][0] = prediction
                    else:
                        mesh[row][col].append(prediction)
                # new graph
                # if -2 < assetSlope < 2 and -.1 <= invPct <= .4:
                #     xCoords.append(invPct)
                #     yCoords.append(incExp)
                #     z.append(prediction)
                #     row = int(assetSlope * 25) + 50
                #     col = int(invPct * 200) + 20
                #     if mesh[row][col][0] == 0:
                #         mesh[row][col][0] = prediction
                #     else:
                #         mesh[row][col].append(prediction)
        except Exception as e:
            continue
    for row in mesh:
        for i in range(0, len(row)):
            row[i] = np.mean(row[i])
    
    cmap = "seismic"
    finalMesh = np.array(mesh)
    finalMesh = finalMesh[:-1, :-1]
    fig, ax = plt.subplots()
    x, y = np.meshgrid(np.linspace(-2, 2, 100), np.linspace(-2, 8, 100))
    zMin, zMax = -Z_MAX, Z_MAX
    ax.set_title("Income Source Prediction Landscape")
    ax.set_xlabel('% Investment Income')
    ax.set_ylabel('Income / Expenses')
    ax.axis([x.min(), x.max(), y.min(), y.max()])
    c = ax.pcolormesh(x, y, finalMesh, cmap=cmap, vmin=zMin, vmax=zMax)
    fig.colorbar(c, ax=ax)
    # points = ax.scatter(xCoords, yCoords, c=z, cmap=cmap, vmin=-Z_MAX, vmax=Z_MAX)
    # fig.colorbar(points)
    plt.show()

    # cmap = "Blues"
    # f, ax = plt.subplots()
    # ax.set_title("Income Source Prediction Landscape")
    # ax.set_xlabel('% Investment Income')
    # ax.set_ylabel('Income / Expenses')
    # points = ax.scatter(xCoords, yCoords, c=z, cmap=cmap, vmin=-Z_MAX, vmax=Z_MAX)
    # f.colorbar(points)
    # plt.show()

def inc_map2():
    # f = open(NEGATIVE_EIN_FILENAME, "r")
    # eins = json.loads(f.read())
    # f.close()
    eins = list(dataset.get_eins(2014, DATA_SPAN))
    resolution = 32
    mesh = [[0 for j in range(resolution)] for i in range(resolution)]

    print("Loaded EINs")

    for i in range(0, len(eins), 2000):
        try:
            populate_inc_map2(eins[i], mesh, resolution)
        except:
            continue
    
    cmap = "RdBu"
    finalMesh = np.array(mesh)
    fig, ax = plt.subplots()
    x, y = np.meshgrid(np.linspace(-.5, 2, resolution), np.linspace(-.5, 2, resolution))
    # maxIndex = np.unravel_index(np.argmax(abs(finalMesh)), finalMesh.shape)
    # zMax = abs(finalMesh[maxIndex[0]][maxIndex[1]])
    maxIndex = np.unravel_index(np.argmax(finalMesh), finalMesh.shape)
    minIndex = np.unravel_index(np.argmin(finalMesh), finalMesh.shape)
    zMax = finalMesh[maxIndex[0]][maxIndex[1]]
    zMin = finalMesh[minIndex[0]][minIndex[1]]
    ax.set_title("Income Source Prediction Landscape")
    ax.set_xlabel('% Original Sustainable Income')
    ax.set_ylabel('% Original Expenses')
    ax.axis([x.min(), x.max(), y.min(), y.max()])
    c = ax.pcolormesh(x, y, finalMesh, cmap=cmap, vmin=zMin, vmax=zMax)
    fig.colorbar(c, ax=ax)
    plt.show()

def populate_inc_map2(ein, mesh, resolution):
    inputs, label = dataset.ein_to_training_data(ein, 2014)
        
    print("EIN", ein, "has been selected")

    # PREPARE THE INPUTS FOR MANIPULATION
    stepConstant = 1 / resolution * 2.5
    xsteps = [0] * 4
    x2steps = [0] * 4
    ysteps = [0] * 4
    for i in range(0, 20):
        mod  = i % 5
        if mod == 1:
            xsteps[i // 5] = stepConstant * inputs[i]
            inputs[i] = -abs(inputs[i]) / 2
        elif mod == 2:
            x2steps[i // 5] = stepConstant * inputs[i]
            inputs[i] = -abs(inputs[i]) / 2
        elif mod == 3:
            ysteps[i // 5] = stepConstant * inputs[i]
            inputs[i] = -abs(inputs[i]) / 2

    # print("Inputs have been prepared")    

    # backwards thru rows
    for i in range(resolution - 1, -1, -1):
        for k in range(3, 20, 5):
            inputs[k] = ysteps[k // 5] * (resolution - 1 - i)
        # forwards thru columns
        for j in range(0, resolution):
            for l in range(1, 20, 5):
                inputs[l] = xsteps[l // 5] * j
            for m in range(2, 20, 5):
                inputs[m] = x2steps[m // 5] * j

            mesh[i][j] += neuralnet.pctpredict(nn, np.array([inputs]), inputs[-3])

        # print(str((resolution - i) / resolution * 100) + "% Complete")

# Network id is passed as cmd line argument
if len(sys.argv) > 1:
    for arg in sys.argv[1:]:
        if arg == "--nindicators":
            negative_indicators()
        elif arg == "--incmap":
            inc_map()
        elif arg == "--incmap2":
            inc_map2()
else:
    print("Please specify analysis type")
