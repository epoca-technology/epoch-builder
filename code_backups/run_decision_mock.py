from pandas import read_csv, concat
from tensorflow import random as tfrandom
from keras import Sequential
from keras.layers import LSTM, Dense, Dropout, Dense
from keras.optimizers import adam_v2
from keras.losses import CategoricalCrossentropy
from keras.metrics import CategoricalAccuracy
from keras.callbacks import EarlyStopping

# Set the random seed
tfrandom.set_seed(212)



# Import the DataFrame
df = read_csv("decision_data/decision_data_dump.csv")



# Split the DF into train and test data
TRAIN_SIZE = 0.7

# Init the x data
train_x = df[:int(df.shape[0]*TRAIN_SIZE)]
test_x = df[int(df.shape[0]*TRAIN_SIZE):]

# Init the y data
train_y = concat([train_x.pop(x) for x in ['up', 'down']], axis=1)
test_y = concat([test_x.pop(x) for x in ['up', 'down']], axis=1)




# Initialize the model
model = Sequential([
    LSTM(100, input_shape=(train_x.shape[1],1,), return_sequences=True),
    LSTM(90, return_sequences=True),
    LSTM(80, return_sequences=True),
    Dropout(0.25),
    LSTM(70, return_sequences=True),  
    LSTM(60, return_sequences=False),
    Dense(2, activation='softmax')
])

# Compile the model
model.compile(optimizer=adam_v2.Adam(learning_rate=0.001),
              loss=CategoricalCrossentropy(),
              metrics=[CategoricalAccuracy()])


# Train the model
print("\n\nTRAINING:")
es = EarlyStopping(monitor='val_categorical_accuracy', mode='max', min_delta=0.001, patience=3)
history = model.fit(
    train_x,
    train_y,
    validation_split=0.2,
    batch_size=64,
    epochs=100,
    callbacks=[es]
)
print("\n\nTRAINING RESULTS:")
print("Loss: ", history.history['loss'])
print("Accuracy: ", history.history['categorical_accuracy'])
print("Val Loss: ", history.history['val_loss'])
print("Val Accuracy: ", history.history['val_categorical_accuracy'])



# Evaluate the model
print("\n\nEVALUATING:")
loss, accuracy = model.evaluate(test_x, test_y)
print("Accuracy: ", accuracy)



# Test predictions
print("\n\nPREDICTING:")
features = [
    [2, 0, 2, 1, 2, 2, 0, 1, 2, 2],
    [0, 1, 1, 1, 2, 0, 1, 2, 1, 0],
    [2, 0, 0, 1, 2, 1, 0, 2, 1, 1],
]
preds = model.predict(features)
for i in range(len(features)):
    print (f"\nInput: {features[i]}")
    print (f"Up: {preds[i][0]}%")
    print (f"Down: {preds[i][1]}%")