from keras import backend as K
from keras.models import Model
import numpy as np
from keras import Sequential
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Bidirectional, Embedding, GRU, Input, Conv1D, GlobalMaxPooling1D, concatenate
from keras.preprocessing import sequence
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from keras.models import load_model
from keras.models import model_from_json

##################################

#         Definir metrics

##################################
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

#         Definir Architecture

##################################
def ConvNet(embeddings, max_sequence_length):
 
    embedding_layer = Embedding(embeddings.shape[0], 
                                embeddings.shape[1],
                                input_length = max_sequence_length,
										            weights=[embeddings],
                                trainable=False,
                                name='emb')
    
    sequence_input = Input(shape=(max_sequence_length,), dtype='int32',name='inp')
    embedded_sequences = embedding_layer(sequence_input)
    convs = []
    filter_sizes = [1,2,4,5,6]
    for filter_size in filter_sizes:
        l_conv = Conv1D(filters=200, 
                        kernel_size=filter_size, 
                        activation='relu')(embedded_sequences)
        l_pool = GlobalMaxPooling1D()(l_conv)
        
        convs.append(l_pool)
    l_merge = concatenate(convs, axis=1)
    x = Dropout(0.1)(l_merge)  
    x = Dense(units=64,  activation='relu')(x)
    x = Dropout(0.2)(x)
    preds = Dense(1, activation='sigmoid')(x)
    model = Model(sequence_input, preds)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy', f1_m, precision_m, recall_m]
                  )
    return model



batch_shape = (512, 30)
batch_unl = 96
power_iteration = 1
embedding_dimension = '200'
trait = 'EXT'
archi = 'CNN'
cells = 256
dense_units = 30
epochs = 10
pathsave = 'gdrive/My Drive/'+'cnn2/'
data='gdrive/My Drive/'



max_features = 20000
maxlen = 30  # Taille des phrases
batch_size = 64

# Charger matrice d'embeddings
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

logfile = pathsave + archi+'_'+str(128)+'_'+trait+'_200d.log'

model = ConvNet(embedding_matrix, 
                maxlen)

history = model.fit(x, ytrain, 
                 epochs=epochs, 
                  validation_split=0.2,
                 batch_size=batch_size,
                 callbacks=[CSVLogger(logfile)]
                )

########################################

#           Sauvegarde modele

#######################################
print("Saving model")
model_json = model.to_json()
with open(pathsave + archi+'_'+str(128)+'_'+trait+'_200d.json', "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(pathsave + archi+'_'+str(128)+'_'+trait+'_200d.h5')
print("Saved model to disk")
print('loading model')
json_file = open(pathsave + archi+'_'+str(128)+'_'+trait+'_200d.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model_test = model_from_json(loaded_model_json)
model_test.load_weights(pathsave + archi+'_'+str(128)+'_'+trait+'_200d.h5')
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