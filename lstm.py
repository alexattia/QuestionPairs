from __future__ import division, print_function
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
# Our features
import preprocessing_lstm
# Keras library
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, merge
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
# Garbage Collection
import gc

###############################################
# LSTM
###############################################
data_1, data_2, test_data_1, test_data_2, test_ids, labels, word_index = preprocessing_lstm.preprocessing()
leaks, test_leaks = preprocessing_lstm.load_leaky()

data = preprocessing_lstm.input_nn_data(data_1, data_2, test_data_1, test_data_2, test_ids, labels, leaks, test_leaks)
data_1_train = data["data_1_train"]
data_2_train = data["data_2_train"]
leaks_train = data["leaks_train"]
labels_train = data["labels_train"]
data_1_val = data["data_1_val"]
data_2_val = data["data_2_val"]
leaks_val = data["leaks_val"]
labels_val = data["labels_val"]
weight_val = data["weight_val"]
test = data["test"]

embedding_index = preprocessing_lstm.Glove_Indexing()
embedding_matrix = preprocessing_lstm.Words_Embedding(word_index, embedding_index)


EMBEDDING_DIM = 300
MAX_SEQUENCE_LENGTH = 144180
# words to vector in glovespace
embedding_layer = Embedding(20355,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)
_ = gc.collect()

act = 'relu'
test_ids = test.test_id
# seed
i = 5
np.random.seed(i)
print ('fold = ' + str(i))
print ('--> Initializing LSTM')

#initialize with random values every time with different seed
num_lstm = np.random.randint(175, 275)
num_dense = np.random.randint(100, 150)
rate_drop_lstm = 0.15 + np.random.rand() * 0.25
rate_drop_dense = 0.15 + np.random.rand() * 0.25
lstm_drp = 0.15 + np.random.rand() * 0.25
leak_drp = 0.15 + np.random.rand() * 0.25
gc.collect()

#----------------------------------------------------------------------------------
#model structure
lstm_layer = LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm)

sequence_1_input = Input(shape=(40,), dtype='int32')
embedded_sequences_1 = embedding_layer(sequence_1_input)
x1 = lstm_layer(embedded_sequences_1)
x1 = Dropout(lstm_drp)(x1)

sequence_2_input = Input(shape=(40,), dtype='int32')
embedded_sequences_2 = embedding_layer(sequence_2_input)
y1 = lstm_layer(embedded_sequences_2)
y1 = Dropout(lstm_drp)(y1)

leaks_input = Input(shape=(leaks_train.shape[1],))
leaks_dense = Dense(int(num_dense/2), activation=act)(leaks_input)
leaks_dense = Dropout(leak_drp)(leaks_dense)

merged = merge([x1, y1, leaks_dense],'concat')
merged = Dropout(rate_drop_dense)(merged)
merged = BatchNormalization()(merged)

merged = Dense(num_dense, activation=act)(merged)
merged = Dropout(rate_drop_dense)(merged)
merged = BatchNormalization()(merged)
preds = Dense(1, activation='sigmoid')(merged)

class_weight = None
weight_val = np.ones(len(labels_val))

print ('--> LSTM Model Created')

#----------------------------------------------------------------------------------
model = Model([sequence_1_input, sequence_2_input, leaks_input], preds)
model.compile(loss='binary_crossentropy',metrics=['acc'], optimizer='nadam')

early_stopping = EarlyStopping(monitor='val_loss', patience=6)
bst_model_path = 'fold' + str(i) + '.h5'
model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)
print('--> LSTM Model Compiled')

#----------------------------------------------------------------------------------
gc.collect()
hist = model.fit([data_1_train, data_2_train, leaks_train], labels_train,
                 validation_data=([data_1_val, data_2_val, leaks_val], labels_val, weight_val),
                 epochs=50, batch_size=2048, shuffle=True, verbose=2,
                 class_weight=class_weight, callbacks=[early_stopping, model_checkpoint])

#----------------------------------------------------------------------------------
print('--> Testing')
model.load_weights('fold' + str(i) + '.h5')
predictions = model.predict([test_data_1, test_data_2, test_leaks], batch_size=2000, verbose=0)
predictions += model.predict([test_data_1, test_data_2, test_leaks], batch_size=2000, verbose=0)
predictions /= 2

#----------------------------------------------------------------------------------
score = pd.DataFrame()
score["score"] = predictions.reshape(len(predictions))
score.to_csv("lstm_final.csv")
