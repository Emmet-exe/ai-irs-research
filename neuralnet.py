from config import *
from numpy.random import seed
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
from matplotlib import pyplot
import keras
import tensorflow

EPOCH_CAP = 350
NETWORK_SHAPE = [24, 18, 21, 15, 8, 3]
SLOW_IMPROVEMENT_THRESHOLD = 1.0015
BATCH_SIZE = 32

# custom loss function - log-scaled MAPE as of right now
def custom_loss(y_true, y_pred):
    ten = tensorflow.constant(10.0)
    label = keras.backend.cast(y_true, keras.backend.floatx())
    return keras.backend.mean(keras.backend.abs(keras.backend.abs(label - y_pred) / label) * (tensorflow.keras.backend.log(keras.backend.abs(label)) / tensorflow.keras.backend.log(ten)))


# 1:1 softsign based input activation - scales down our numbers so that they are within domain of tanh activation
def custom_in_activation(x):
    return 2 * x / keras.backend.cast((1e6) + tensorflow.math.abs(x), keras.backend.floatx())

# for generating labels
def scalar_activation(x):
    return 2 * x / float((1e6) + abs(x))

# basically the piece-wise softsign inverse
def activation_inverse(x):
    if x < 0:
        return (1e6) * x / float(x + 1)
    return -1 * (1e6) * x / float(x - 1)

def activation_inverse_tensor(x):
    op = -1 * tensorflow.math.sign(x)
    return op * (1e6) * x / keras.backend.cast((x + (1 * op)), keras.backend.floatx())

# no custom out activation as of rn
def custom_out_activation(x):
    return activation_inverse_tensor(keras.backend.tanh(x))

# create the neural network
def createNN(seedNum, inputNum):
    if (seedNum != 0):
        seed(seedNum)
        tensorflow.random.set_seed(seedNum)
    
    tensorflow.keras.utils.get_custom_objects().update({"customInAct": custom_in_activation, "customOutAct": custom_out_activation, "customLoss": custom_loss})

    model = Sequential()
    model.add(Dense(NETWORK_SHAPE[0], activation="customInAct", kernel_initializer="he_normal", input_shape=(inputNum,)))
    for i in range (1, len(NETWORK_SHAPE)):
        model.add(Dense(NETWORK_SHAPE[i], activation="tanh", kernel_initializer="he_normal"))
    model.add(Dense(1, activation = "customOutAct"))
    model.compile(loss="customLoss", optimizer="adam")
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
        self.val_losses = []
        #Counter for epochs with slow improvement
        self.slow_improvement_count = 0
        self.overfitting_increase = 0
        

    # callback function that runs at the end of every epoch
    def on_epoch_end(self, epoch, logs={}):
        # Adding important values to lists in the class
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
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

        #Increment counter for overfitting
        if (len(self.val_losses) >= 2) and (self.val_losses[-1] - self.val_losses[-2] > 0) and (self.val_losses[-1] - self.losses[-1]) > (self.val_losses[-2] - self.losses[-2]):
            self.overfitting_increase += 1
        else:
            self.overfitting_increase = 0

        if self.overfitting_increase >= 5:
            self.log("Ended (overfitting) at epoch " + str(epoch))
            self.end()

        # Hard training stop
        if epoch == EPOCH_CAP:
            self.log("Ending after final epoch (" + str(EPOCH_CAP) + " epochs complete)")
            self.end()

# Train and save a neural network on the given training data
def train(nn, nnid, inputs, labels):
    # train the model
    monitor = MonitorNN(nn, nnid, inputs, labels)
    history = nn.fit(x=inputs, y=labels, batch_size=BATCH_SIZE, epochs=EPOCH_CAP, verbose=1, callbacks=[monitor], validation_split=0.5)
    # # save model to file
    nn.save("models/" + nnid + ".h5")
    return history

# load a model from /h5 file that is saved at the end of training
def load_nn(name):
    model = load_model("models/" + name + ".h5", custom_objects={"custom_in_activation": custom_in_activation, "custom_out_activation": custom_out_activation, "customLoss": custom_loss})
    return model

# evaluate the model on testing dataset
def evaluate(nn, inputs, labels):
    return nn.evaluate(inputs, labels, verbose=0)

# Make a prediction
def predict(nn, inputs):
    return nn.predict(inputs, batch_size=None, verbose=0)[0].item()

# percentage increase or decrease prediction
def pctpredict_old(nn, inputs, total):
    return (activation_inverse(predict(nn, inputs)) / float(total)) * 100

# percentage increase or decrease prediction
def pctpredict(nn, inputs, total):
    return (predict(nn, inputs)) / float(total) * 100

# displays a graph of training loss, given history object
def plotTraining(history):
    pyplot.title('Learning Curves')
    pyplot.xlabel('Epoch')
    pyplot.ylabel('MSE (Loss)')
    pyplot.plot(history.history['loss'], label='Training', color=MAIN_COLOR)
    pyplot.plot(history.history['val_loss'], label='Validation', color=SECONDARY_COLOR)
    pyplot.legend()
    pyplot.show()
