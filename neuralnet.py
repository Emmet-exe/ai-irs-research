from config import *
from numpy.random import seed
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
from matplotlib import pyplot
import keras
import tensorflow

EPOCH_CAP = 400
NETWORK_SHAPE = [25, 19, 21, 13, 7]
SLOW_IMPROVEMENT_THRESHOLD = 1.00007
BATCH_SIZE = 32

# 1:1 softsign based input activation - scales down our numbers so that they are within domain of tanh activation
def custom_in_activation(x):
    return 2 * x / keras.backend.cast((10**6) + tensorflow.math.abs(x), keras.backend.floatx())
    # return tensorflow.math.sign(x) * 1.3 / keras.backend.cast(tensorflow.math.pow(x, -.9888), keras.backend.floatx())
    # return 1 * math.pow(10, 8) * ((1 / keras.backend.cast((1 + tensorflow.math.pow(math.e, -1 * 5 * x * math.pow(10, -8))), keras.backend.floatx())) - .5)
    # return tensorflow.math.sign(x) * 2 * tensorflow.math.pow(tensorflow.math.abs(x), keras.backend.cast(.95, keras.backend.floatx()))
    # return x * keras.backend.log(keras.backend.cast(tensorflow.math.abs(x), keras.backend.floatx())) / keras.backend.log(50000.0)
    # return x

# for generating labels
def scalar_activation(x):
    return x / float((10**6) + abs(x))
    # return 1 * math.pow(10, 8) * (1 / (1 + math.pow(math.e, -1 * 5 * x * math.pow(10, -8))) - .5)

# basically the piece-wise softsign inverse
def activation_inverse(x):
    if x < 0:
        return (10**6) * x / float(x + 1)
    return -1 * (10**6) * x / float(x - 1)

# no custom out activation as of rn
def custom_out_activation(x):
    return x

# create the neural network
def createNN(seedNum, inputNum):
    if (seedNum != 0):
        seed(seedNum)
        tensorflow.random.set_seed(seedNum)
    
    tensorflow.keras.utils.get_custom_objects().update({"customInAct": custom_in_activation, "customOutAct": custom_out_activation})

    model = Sequential()
    model.add(Dense(NETWORK_SHAPE[0], activation="customInAct", kernel_initializer="he_normal", input_shape=(inputNum,)))
    for i in range (1, len(NETWORK_SHAPE)):
        model.add(Dense(NETWORK_SHAPE[i], activation="tanh", kernel_initializer="he_normal"))
    model.add(Dense(1, activation = "tanh"))
    model.compile(loss="mean_squared_error", optimizer="adam")
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
        if epoch > 0 and epoch % 5 == 0:
            self.log("Network reached epoch #" + str(epoch) + ". Loss: " + str(self.losses[-1])) 

        # Hard stop for slow validation error improvement
        if (len(self.losses) >= 10) and (self.losses[-2] / float(self.losses[-1]) < SLOW_IMPROVEMENT_THRESHOLD):
            # self.log("(" + str(epoch) + ") Slow improvement notice - Loss: " + str(self.losses[-1]))
            self.slow_improvement_count += 1
        else:
            self.slow_improvement_count = 0
        
        if (self.slow_improvement_count >= 6):
            self.log('Ended (slow improvement) at epoch ' + str(epoch))
            self.end()

        # Hard training stop
        if epoch == EPOCH_CAP:
            self.log("Ending after final epoch (" + str(EPOCH_CAP) + " epochs complete)")
            self.end

# Train and save a neural network on the given training data
def train(nn, nnid, inputs, labels):
    # train the model
    monitor = MonitorNN(nn, nnid, inputs, labels)
    history = nn.fit(x=inputs, y=labels, batch_size=BATCH_SIZE, epochs=EPOCH_CAP, verbose=1, callbacks=[monitor], validation_split=0.25)
    # # save model to file
    nn.save("models/" + nnid + ".h5")
    return history

# load a model from /h5 file that is saved at the end of training
def load_nn(name):
    model = load_model("models/" + name + ".h5", custom_objects={"custom_in_activation": custom_in_activation, "custom_out_activation": custom_out_activation})
    return model

# evaluate the model on testing dataset
def evaluate(nn, inputs, labels):
    return nn.evaluate(inputs, labels, verbose=0)

# Make a prediction
def predict(nn, inputs):
    return nn.predict(inputs, batch_size=None, verbose=0)[0].item()

# displays a graph of training loss, given history object
def plotTraining(history):
    pyplot.title('Learning Curves')
    pyplot.xlabel('Epoch')
    pyplot.ylabel('MSE (Loss)')
    pyplot.plot(history.history['loss'], label='Training', color=MAIN_COLOR)
    pyplot.plot(history.history['val_loss'], label='Validation', color=SECONDARY_COLOR)
    pyplot.legend()
    pyplot.show()
