from keras.models import Sequential
from keras.layers import Dense
import keras.metrics
import pandas as pd
import numpy as np
import math
from keras.callbacks import CSVLogger

#Training data:
#Import test data and training data
input = pd.read_csv("C:/Users/Kilby/Code/Waterloo/CS680/Project/human-data/FINAL_s.csv", header=None).to_numpy()
target = pd.read_csv("C:/Users/Kilby/Code/Waterloo/CS680/Project/human-data/FINAL_a.csv", header=None).to_numpy()
# new_input = []
# for i in input:
#         new_input.append(i[:6])
# new_input = np.asarray(new_input)
# input = new_input


train_input = input[:math.floor(len(input)*0.99)]
train_target = target[:math.floor(len(input)*0.99)]

test_input = input[math.floor(len(input)*0.99):]
test_target = target[math.floor(len(input)*0.99):]

nepochs = 500
input_dimensions = 6
output_dimensions = 4

nhidden = 20

model = Sequential()
model.add(Dense(nhidden, input_dim=input_dimensions,activation='sigmoid'))
model.add(Dense(nhidden, activation='sigmoid'))
model.add(Dense(output_dimensions,activation='linear'))
model.compile(loss="mean_squared_error", optimizer='adam')



#Train model
csv_logger = CSVLogger("nn_training.csv", append=True)

model.fit(train_input,train_target,epochs=nepochs, verbose=1, callbacks=[csv_logger])
pred = model.predict(test_input)


sum = 0
for i in range(len(pred)):
    print(pred[i])
    print(test_target[i])
    print("---------")
    a = np.argmax(pred[i])
    if (test_target[i][a]>=test_target[i]).all():
        sum += 1

print(sum)


model.save("C:/Users/Kilby/Code/Waterloo/CS680/Project/NN1.h5")