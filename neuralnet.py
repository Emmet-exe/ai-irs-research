from numpy.random import seed
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
from matplotlib import pyplot
import keras
import tensorflow

EPOCH_CAP = 300
NETWORK_SHAPE = [20, 16, 10, 6]
SLOW_IMPROVEMENT_THRESHOLD = 1.006
ACTIVATION = None

# create the neural network
def createNN(seedNum, inputNum):
    if (seedNum != 0):
        seed(seedNum)
        tensorflow.random.set_seed(seedNum)
    
    model = Sequential()
    model.add(Dense(NETWORK_SHAPE[0], activation=ACTIVATION, kernel_initializer="he_normal", input_shape=(inputNum,)))
    for i in range (1, len(NETWORK_SHAPE)):
        model.add(Dense(NETWORK_SHAPE[i], activation=ACTIVATION, kernel_initializer="he_normal"))
    model.add(Dense(1))
    model.compile(loss="mean_absolute_error", optimizer="adam")
    return model

# Callback class to monitor progress of neural network training
class MonitorNN(keras.callbacks.Callback):
    # log function
    def log(self, msg):
        self.log_file.write(msg + "\n")

    # stop function
    def end(self):
        self.model.stop_training = True
        self.log("----------- [!] Training Complete - Final Loss: " + str(self.losses[-1]) + " -----------")
        self.log_file.close()

    # variable initialization, called when neural net is created
    def __init__(self, nn, nnid, inputs, labels):
        self.nnid = nnid
        self.log_file = open("training_logs/" + str(nnid) + ".txt", "w")
        string_shape = [str(int) for int in NETWORK_SHAPE]
        self.log("----- Training Neural Net #" + str(nnid) + " with shape (" + ", ".join(string_shape) + ") -----")
        self.log("Pre-training Loss: " + str(evaluate(nn, inputs, labels)))

    # initialization function to set up loss lists
    def on_train_begin(self, logs={}):
        #Basic logs
        self.losses = []
        self.acc = []
        #Counter for epochs with slow improvement
        self.slow_improvement_count = 0
        

    # callback function that runs at the end of every epoch
    def on_epoch_end(self, epoch, logs={}):
        # Adding important values to lists in the class
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))

        # Less frequent logger
        if epoch > 0 and epoch % (EPOCH_CAP/40) == 0:
            self.log("Network reached epoch #" + str(epoch) + ". Loss: " + str(self.losses[-1])) 

        # Hard stop for slow validation error improvement
        if (len(self.losses) >= 10) and (self.losses[-2] / float(self.losses[-1]) < SLOW_IMPROVEMENT_THRESHOLD):
            # self.log("(" + str(epoch) + ") Slow improvement notice - Loss: " + str(self.losses[-1]))
            self.slow_improvement_count += 1
        else:
            self.slow_improvement_count = 0
        
        if (self.slow_improvement_count == 5):
            self.log('Ended (slow improvement) at epoch ' + str(epoch))
            self.end()

        # Hard training stop
        if epoch == EPOCH_CAP:
            self.log("Ending after final epoch (" + str(EPOCH_CAP) + " epochs complete)")
            self.end


def train(nn, nnid, inputs, labels):
    # train the model
    monitor = MonitorNN(nn, nnid, inputs, labels)
    history = nn.fit(x=inputs, y=labels, batch_size=10, epochs=EPOCH_CAP, verbose=1, callbacks=[monitor])
    # # save model to file
    nn.save("models/" + nnid + ".h5")
    return history

def load_nn(name):
    model = load_model("models/" + name + ".h5")
    return model

def evaluate(nn, inputs, labels):
     # evaluate the model
    return nn.evaluate(inputs, labels, verbose=0)

# Make a prediction
def predict(nn, inputs):
    return nn.predict(inputs, batch_size=None, verbose=0)

def plotTraining(history):
    pyplot.title('Learning Curves')
    pyplot.xlabel('Epoch')
    pyplot.ylabel('RMSE')
    pyplot.plot(history.history['loss'], label='training')
    #pyplot.plot(history.history['val_loss'], label='val')
    pyplot.legend()
    pyplot.show()


