import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, matthews_corrcoef
from keras.models import Model
import numpy as np
from keras import Sequential
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Bidirectional, Embedding, GRU
from keras.preprocessing import sequence
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from keras.models import model_from_json
from keras.models import load_model
from keras.optimizers import Adam
from keras import backend as K

##################################

#         Definir metrics

##################################

class Metrics(Callback):
  def on_train_begin(self, logs={}):
    self.val_f1s = []
    self.val_recalls = []
    self.val_precisions = []
    self.val_mccs = []
  
  def on_epoch_end(self, epoch, logs):
    val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
    val_targ = self.validation_data[1]
    _val_f1 = f1_score(val_targ, val_predict)
    _val_recall = recall_score(val_targ, val_predict)
    _val_precision = precision_score(val_targ, val_predict)
    _val_mcc = matthews_corrcoef(val_targ, val_predict)
    self.val_f1s.append(_val_f1)
    self.val_recalls.append(_val_recall)
    self.val_precisions.append(_val_precision)
    self.val_mccs.append(_val_mcc)
    logs['val_f1s'] = _val_f1
    logs['val_prec'] = _val_precision
    logs['val_rec'] = _val_recall
    logs['val_mcc'] = _val_mcc
    print (" — val_f1: %f — val_precision: %f — val_recall %f — val_mcc %f" %(_val_f1, _val_precision, _val_recall, _val_mcc))
    return


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

##################################

metrics = Metrics()

batch_shape = (512, 30)
batch_unl = 96
power_iteration = 1


embedding_dimension = '200'
trait = 'NEU'
archi = 'LSTM'
cells = 768       # Nbre de cellules LSTM
dense_units = 60  # Nbre de cellules couche cachée
epochs = 20       
max_features = 20000
maxlen = 30       #  Taille des phrases
batch_size = 512

pathsave = 'gdrive/My Drive/MyPersonality/lstm_train/'
data='gdrive/My Drive/MyPersonality/'

# Charger matrice embedding
embedding_matrix = np.load( "{}".format(data)+"embedding_matrix_200d.npy",allow_pickle=True )

# Charger corpus

xtrain = np.load( "{}".format(data)+trait+"_xtrain.npy",allow_pickle=True)
ytrain = np.load( "{}".format(data)+trait+"_ytrain.npy".format(data) ,allow_pickle=True)
xtest = np.load( "{}".format(data)+trait+"_xtest.npy".format(data) ,allow_pickle=True)
ytest = np.load( "{}".format(data)+trait+"_ytest.npy".format(data),allow_pickle=True )

x_test = sequence.pad_sequences(xtest, maxlen = maxlen)
y_test = np.reshape(ytest, newshape=(ytest.shape[0], 1))
x = sequence.pad_sequences(xtrain, maxlen = maxlen)
ytrain = np.reshape(ytrain, newshape=(ytrain.shape[0], 1))





print('Build model...')
model = Sequential()
model.add(Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],input_length = maxlen,
										 weights=[embedding_matrix],trainable=False, name='Embedding'))
model.add(Dropout(rate=0.5, seed=91, name='Dropout'))
model.add(LSTM(cells, dropout=0.2, recurrent_dropout=0.2))

model.add(Dense(dense_units, activation='relu', name='Dense'))
model.add(Dense(1, activation='sigmoid', name='Prob'))
opt = Adam(learning_rate=0.0001)

model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy', f1_m, precision_m, recall_m]
              )

print('Train...')
logfile = pathsave + archi +'_'+str(cells)+'_'+str(dense_units)+'_'+trait+'_200d.log'
history = model.fit(x, ytrain,
          batch_size=batch_shape[0],
          epochs=epochs,
          validation_data=(x_test,y_test),
          callbacks=[CSVLogger(logfile), metrics]
          )




#########################################

#           Sauvegarde modele

#########################################

print("Saving model")
model_json = model.to_json()
with open(pathsave + archi+'_'+str(cells)+'_'+str(dense_units)+'_'+trait+'_200d.json', "w") as json_file:
    json_file.write(model_json)
model.save_weights(pathsave + archi+'_'+str(cells)+'_'+str(dense_units)+'_'+trait+'_200d.h5')
print("Saved model to disk")
print('loading model')
json_file = open(pathsave + archi+'_'+str(cells)+'_'+str(dense_units)+'_'+trait+'_200d.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model_test = model_from_json(loaded_model_json)
model_test.load_weights(pathsave + archi+'_'+str(cells)+'_'+str(dense_units)+'_'+trait+'_200d.h5')
print("Loaded model from disk")

########################################

#           Phase de test

#######################################


test_loss, test_accuracy, test_fscore, test_precision, test_recall = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
results = open(logfile,"a")
results.write('\n\n')
results.write('----------------------------------------TEST-------------------------------\n')
results.write('Test loss:  '+str(test_loss)+'\n\n')
results.write('Test precision:  '+str(test_precision)+'\n\n')
results.write('Test recall:  '+str(test_recall)+'\n\n')
results.write('Test fscore:  '+str(test_fscore)+'\n\n')
results.write('Test acc:  '+str(test_accuracy)+'\n\n')
results.close()

print('Test score:', test_fscore)
print('Test accuracy:', test_accuracy)

print(model.metrics_names)