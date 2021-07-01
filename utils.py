import math

# Sigmoid function
def sigmoid(x):
  return 1 / (1 + math.exp(-x))

# Needed to JSON serialize a set
def serialize_set(obj):
    if isinstance(obj, set):
        return list(obj)
    return obj

# Parse a string to an integer, but empty string --> 0
def zeroint(s):
    if len(s) == 0:
        return 0
    return int(s)

# Format a float as a string to 3 decimal places
def strround(num):
    return str(round(num * 1000) / 1000.0)

def pos(num):
    if num >= 0:
        return num
    else:
        return 0