

import numpy as np

np.random.seed(42)

f = open("mnist_test.amat", "r")

test = []

for line in f:
    test.append([float(x) for x in line.split()])

test = np.array(test)


f = open("mnist_train.amat", "r")

trainval = []

for line in f:
    trainval.append([float(x) for x in line.split()])

trainval = np.array(trainval)

train = trainval[:10000, :].copy()
valid = trainval[10000:, :].copy()

np.savez("mnist_trainval", images=trainval[:, :-1].reshape(-1, 28, 28), labels=trainval[:, -1])
np.savez("mnist_test", images=test[:, :-1].reshape(-1, 28, 28), labels=test[:, -1])
np.savez("mnist_train", images=train[:, :-1].reshape(-1, 28, 28), labels=train[:, -1])
np.savez("mnist_valid", images=valid[:, :-1].reshape(-1, 28, 28), labels=valid[:, -1])

del train
del valid
del test

np.random.shuffle(trainval)

train = trainval[:10000, :].copy()
valid = trainval[10000:, :].copy()

np.savez("mnist_train_shuffled", images=train[:, :-1].reshape(-1, 28, 28), labels=train[:, -1])
np.savez("mnist_valid_shuffled", images=valid[:, :-1].reshape(-1, 28, 28), labels=valid[:, -1])

