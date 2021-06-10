import tensorflow as tf
from tensorflow import keras as K
import numpy as np
import argparse
import h5py

from progressbar import ProgressBar
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
from keras import backend as b
from sklearn.metrics import confusion_matrix


savepath = './adv_train/'
maxlen=30
#initialiser tensorflow
init_op = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init_op)

class Network:
    def __init__(self, session, dict_weight, lr ,dropout=0.2, lstm_units=256, dense_units=70):

        #Creer reseau
        self.sess = session
        K.backend.set_session(self.sess)

        # definir couches
        dict_shape = dict_weight.shape
        self.emb = K.layers.Embedding(dict_shape[0], dict_shape[1], weights=[dict_weight], trainable=False, name='embedding')
        self.drop = K.layers.Dropout(rate=dropout, seed=91, name='dropout')
        self.lstm = K.layers.LSTM(lstm_units, stateful=False, return_sequences=False, name='lstm')
        self.dense = K.layers.Dense(dense_units, activation='relu', name='dense')
        self.p = K.layers.Dense(1, activation='sigmoid', name='p')

        # Definir optimisation
        self.optimizer = tf.train.AdamOptimizer(learning_rate=lr)


    """
    Fonction qui retourne la sortie du réseau et la sortie de la couche d'embedding
    """
    def __call__(self, batch, perturbation=None):
        embedding = self.emb(batch) 
        drop = self.drop(embedding)
        if (perturbation is not None):
            drop += perturbation
        lstm = self.lstm(drop)
        dense = self.dense(lstm)
        return self.p(dense), embedding
    
    """
    Fonction qui genere les mini-batchs
    """
    def get_minibatch(self, x, y, ul, batch_shape=(512, 30)):
        x = K.preprocessing.sequence.pad_sequences(x, maxlen=batch_shape[1])
        permutations = np.random.permutation( len(y) )
        ul_permutations = None
        len_ratio = None
        if (ul is not None):
            ul = K.preprocessing.sequence.pad_sequences(ul, maxlen=batch_shape[1])
            ul_permutations = np.random.permutation( len(ul) )
            len_ratio = len(ul)/len(y)
        for s in range(0, len(y), batch_shape[0]):
            perm = permutations[s:s+batch_shape[0]]
            minibatch = {'x': x[perm], 'y': y[perm]}
            if (ul is not None):
                ul_perm = ul_permutations[int(np.floor(len_ratio*s)):int(np.floor(len_ratio*(s+batch_shape[0])))]
                minibatch.update( {'ul': np.concatenate((ul[ul_perm], x[perm]), axis=0)} )
            yield minibatch

    """
        Fonction qui retourne l'erreur et la sortie de la couche d'embedding 
    """        
    def get_loss(self, batch, labels):
        pred, emb = self(batch)
        loss = K.losses.binary_crossentropy(labels, pred)
        return tf.reduce_mean( loss ), emb
    
    """
        Fonction qui calcule et retourne l'erreur adversaire
    """  
    def get_adv_loss(self, batch, labels, loss, emb, p_mult):
        gradient = tf.gradients(loss, emb, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)[0]
        p_adv = p_mult * tf.nn.l2_normalize(tf.stop_gradient(gradient), dim=1)
        adv_loss = K.losses.binary_crossentropy(labels, self(batch, p_adv)[0])
        return tf.reduce_mean( adv_loss )

    """
        Fonction qui calcule et retourne l'erreur adversaire virtuelle
    """  
    def get_v_adv_loss(self, ul_batch, p_mult, power_iterations=1):
        bernoulli = tf.distributions.Bernoulli
        prob, emb = self(ul_batch)
        prob = tf.clip_by_value(prob, 1e-7, 1.-1e-7)
        prob_dist = bernoulli(probs=prob)
        # Generation de la perturbation adv virt
        d = tf.random_uniform(shape=tf.shape(emb), dtype=tf.float32)
        for _ in range( power_iterations ):
            d = (0.02) * tf.nn.l2_normalize(d, dim=1)
            p_prob = tf.clip_by_value(self(ul_batch, d)[0], 1e-7, 1.-1e-7)
            kl = tf.distributions.kl_divergence(prob_dist, bernoulli(probs=p_prob), allow_nan_stats=False)
            gradient = tf.gradients(kl, [d], aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)[0]
            d = tf.stop_gradient(gradient)
        d = p_mult * tf.nn.l2_normalize(d, dim=1)
        tf.stop_gradient(prob)
        # Calcul de l'erreur
        p_prob = tf.clip_by_value(self(ul_batch, d)[0], 1e-7, 1.-1e-7)
        v_adv_loss = tf.distributions.kl_divergence(prob_dist, bernoulli(probs=p_prob), allow_nan_stats=False)
        return tf.reduce_mean( v_adv_loss )

    """
        Etape de validation
    """
    def validation(self, x, y, trait, batch_shape=(512, 30)):
        print( 'Validation...' )
        
        labels = tf.placeholder(tf.float32, shape=(None, 1), name='validation_labels')
        batch = tf.placeholder(tf.float32, shape=(None, batch_shape[1]), name='validation_batch')

        accuracy = tf.reduce_mean( K.metrics.binary_accuracy(labels, self(batch)[0]) )
        y_true=labels
        y_pred=self(batch)[0]

        true_positives = b.sum(b.round(b.clip(y_true * y_pred, 0, 1)))
        possible_positives = b.sum(b.round(b.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + b.epsilon())
        true_positives = b.sum(b.round(b.clip(y_true * y_pred, 0, 1)))
        predicted_positives = b.sum(b.round(b.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + b.epsilon())
        fscore= 2*((precision*recall)/(precision+recall+b.epsilon()))

        fscores = list()
        accuracies = list()
        minibatch = self.get_minibatch(x, y, ul=None, batch_shape=batch_shape)
        for val_batch in minibatch:
            fd = {batch: val_batch['x'], labels: val_batch['y'], K.backend.learning_phase(): 0} #test mode
            accuracies.append( self.sess.run(accuracy, feed_dict=fd) )
            fscores.append(self.sess.run(fscore, feed_dict=fd))
        logfile = savepath+'lstm_adv'+trait+'.log'
        log_msg = " val accuracy is {:.3f} (train) -- val f1score is {:.3f}"
        results = open(logfile,"a")
        results.write('\n\n')
        results.write(log_msg.format(np.asarray(accuracies).mean(),  np.asarray(fscores).mean())+'\n\n')
        results.close()
        print( "Average accuracy on validation is {:.3f}".format(np.asarray(accuracies).mean()) )



    """
       Training 
    """
    def train(self,dataset, trait, dict_weight,batch_shape=(512, 30), epochs=25, loss_type='none', p_mult=0.02, init=None, save=None, dropout=0.2, lstm_units=256, dense_units=70,saving=None):
        
        
        print( 'Training...' )
        xtrain = np.load( "{}".format(dataset)+trait+"_xtrain.npy",allow_pickle=True)
        ytrain = np.load( "{}".format(dataset)+trait+"_ytrain.npy" ,allow_pickle=True)
        ultrain = np.load( "{}".format(dataset)+trait+"_ultrain.npy",allow_pickle=True) if (loss_type == 'v_adv') else None
        
        
        # Definition ens val
        xval = list()
        yval = list()
        for _ in range( int(len(ytrain)*0.025) ):
            xval.append( xtrain[0] ); xval.append( xtrain[-1] )
            yval.append( ytrain[0] ); yval.append( ytrain[-1] )
            xtrain = np.delete(xtrain, 0); xtrain = np.delete(xtrain, -1)
            ytrain = np.delete(ytrain, 0); ytrain = np.delete(ytrain, -1)
        xval = np.asarray(xval)
        yval = np.asarray(yval)
        print( '{} elements in validation set'.format(len(yval)) )

        yval = np.reshape(yval, newshape=(yval.shape[0], 1))
        ytrain = np.reshape(ytrain, newshape=(ytrain.shape[0], 1))
        
        labels = tf.placeholder(tf.float32, shape=(None, 1), name='train_labels')
        batch = tf.placeholder(tf.float32, shape=(None, batch_shape[1]), name='train_batch')
        ul_batch = tf.placeholder(tf.float32, shape=(None, batch_shape[1]), name='ul_batch')
        
        accuracy = tf.reduce_mean( K.metrics.binary_accuracy(labels, self(batch)[0]) )


        y_true=labels
        y_pred=self(batch)[0]

        true_positives = b.sum(b.round(b.clip(y_true * y_pred, 0, 1)))
        possible_positives = b.sum(b.round(b.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + b.epsilon())
        true_positives = b.sum(b.round(b.clip(y_true * y_pred, 0, 1)))
        predicted_positives = b.sum(b.round(b.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + b.epsilon())
        fscore= 2*((precision*recall)/(precision+recall+b.epsilon()))

        loss, emb = self.get_loss(batch, labels)
        if (loss_type == 'adv'):
            loss += self.get_adv_loss(batch, labels, loss, emb, p_mult)
        elif (loss_type == 'v_adv'):
            loss += self.get_v_adv_loss(ul_batch, p_mult)

        opt = self.optimizer.minimize( loss )

        # Initialiser parametres
        if (init is None):
            self.sess.run( [var.initializer for var in tf.global_variables() if not('embedding' in var.name)] )
            print( 'Random initialization' )
        else:
            saver = tf.train.Saver()
            saver.restore(self.sess, init)
            print( 'Restored value' )
        
        _losses = list()
        _accuracies = list()
        _fscores = list()
        list_ratio = (len(ultrain)/len(ytrain)) if (ultrain is not None) else None



        # Boucle d'iter
        for epoch in range(epochs):
            losses = list()
            accuracies = list()
            validation = list()
            fscores = list()

            bar = ProgressBar(max_value=np.floor(len(ytrain)/batch_shape[0]).astype('i'))
            minibatch = enumerate(self.get_minibatch(xtrain, ytrain, ultrain, batch_shape=batch_shape))
            for i, train_batch in minibatch:
                fd = {batch: train_batch['x'], labels: train_batch['y'], K.backend.learning_phase(): 1} #training mode
                if (loss_type == 'v_adv'):
                    fd.update( {ul_batch: train_batch['ul']} )
                
                _, acc_val, loss_val = self.sess.run([opt, accuracy, loss], feed_dict=fd)
                accuracies.append( acc_val )
                losses.append( loss_val )
                fscores.append(self.sess.run(fscore, feed_dict=fd))

                bar.update(i)
            
            # Sauvegarde metrics
            _accuracies.append( accuracies )
            _losses.append(losses)
            logfile = savepath+'lstm_adv'+trait+'.log'
            log_msg = "\nEpoch {} of {} -- average accuracy is {:.3f} (train) -- average loss is {:.3f} -- average f1score is {:.3f}"
            results = open(logfile,"a")
            results.write('\n\n')
            results.write(log_msg.format(epoch+1, epochs, np.asarray(accuracies).mean(), np.asarray(losses).mean(), np.asarray(fscores).mean())+'\n\n')
            results.close()
            
            log_msg = "\nEpoch {} of {} -- average accuracy is {:.3f} (train) -- average loss is {:.3f} -- average f1score is {:.3f}"
            print( log_msg.format(epoch+1, epochs, np.asarray(accuracies).mean(), np.asarray(losses).mean(), np.asarray(fscores).mean()) )
            
            # validation
            self.validation(xval, yval, trait, batch_shape=batch_shape)
            
        ##################################

        #         Sauvegarde modele 

        ##################################

        if saving :
              print('saving model')
              dict_shape = dict_weight.shape
              clone_model = K.Sequential (
                  [
                   K.layers.Embedding(dict_shape[0], dict_shape[1], weights=[dict_weight], trainable=False, name='embedding'),
                   K.layers.Dropout(rate=dropout, seed=91, name='dropout'),
                   K.layers.LSTM(lstm_units, stateful=False, return_sequences=False, name='lstm'),
                   K.layers.Dense(dense_units, activation='relu', name='dense'),
                   K.layers.Dense(1, activation='sigmoid', name='p')
                  ]
              )
              clone_model.layers[0].set_weights(self.emb.get_weights())
              clone_model.layers[1].set_weights(self.drop.get_weights())
              clone_model.layers[2].set_weights(self.lstm.get_weights())
              clone_model.layers[3].set_weights(self.dense.get_weights())
              clone_model.layers[4].set_weights(self.p.get_weights())
              clone_model.save(savepath+'lstm_adv'+trait+'.h5')

        
        # Graphe d'erreur et acc
        plt.plot([np.asarray(l).mean() for l in _losses], color='red', linestyle='solid', marker='o', linewidth=2)
        plt.plot([np.asarray(a).mean() for a in _accuracies], color='blue', linestyle='solid', marker='o', linewidth=2)
        plt.savefig(savepath+'train_{}_e{}_m{}_l{}.png'.format(loss_type, epochs, batch_shape[0], batch_shape[1]))

        ##################################

        #         Phase de tests

        ##################################

    def test(self, dataset, trait ,batch_shape=(256, 400)):
        print( 'Test...' )
        xtest = np.load( "{}".format(dataset)+trait+"_xtest.npy" ,allow_pickle=True)
        ytest = np.load( "{}".format(dataset)+trait+"_ytest.npy",allow_pickle=True )
        ytest = np.reshape(ytest, newshape=(ytest.shape[0], 1))
        
        labels = tf.placeholder(tf.float32, shape=(None, 1), name='test_labels')
        batch = tf.placeholder(tf.float32, shape=(None, batch_shape[1]), name='test_batch')

        accuracy = tf.reduce_mean( K.metrics.binary_accuracy(labels, self(batch)[0]) )
        y_true=labels
        y_pred=self(batch)[0]

        true_positives = b.sum(b.round(b.clip(y_true * y_pred, 0, 1)))
        possible_positives = b.sum(b.round(b.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + b.epsilon())
        
        
        true_positives = b.sum(b.round(b.clip(y_true * y_pred, 0, 1)))
        predicted_positives = b.sum(b.round(b.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + b.epsilon())
        
        
        fscore= 2*((precision*recall)/(precision+recall+b.epsilon()))


        accuracies = list()
        fscores=list()
        recalls=list()
        precisions=list()
        bar = ProgressBar(max_value=np.floor(len(ytest)/batch_shape[0]).astype('i'))
        minibatch = enumerate(self.get_minibatch(xtest, ytest, ul=None, batch_shape=batch_shape))
        for i, test_batch in minibatch:
            fd = {batch: test_batch['x'], labels: test_batch['y'], K.backend.learning_phase(): 0} #test mode
            accuracies.append( self.sess.run(accuracy, feed_dict=fd) )
            fscores.append(self.sess.run(fscore, feed_dict=fd))
            recalls.append(self.sess.run(recall, feed_dict=fd))
            precisions.append(self.sess.run(precision, feed_dict=fd))
            bar.update(i)

        # Metrics
        logfile = savepath+'lstm_adv'+trait+'.log'
        print( "\nAverage accuracy is {:.3f}".format(np.asarray(accuracies).mean()) )
        print( "\nF1-scores is ",fscores )
        print( "\nRecalls is ",recalls )
        print( "\nPrecisions is ",precisions )
        results = open(logfile,"a")
        results.write('\n')
        results.write('----------------------------------------TEST-------------------------------\n')
        results.write('Test score:  '+str(np.asarray(fscores).mean())+'\n\n')
        results.write('Test acc:  '+str(np.asarray(accuracies).mean())+'\n\n')
        results.close()
        
        
        
def main(data, n_epochs, n_ex, ex_len, lt, pm):
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    config = tf.ConfigProto(log_device_placement=True)
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    embedding_weights = np.load( "{}embedding_matrix_200d.npy".format(data),allow_pickle=True )
    trait = 'CON'
    net = Network(session, embedding_weights, lr=0.0001, dropout=0.2, lstm_units=1024, dense_units=30)
    net.train(data, trait ,embedding_weights , batch_shape=(n_ex, ex_len), 
              epochs=25, loss_type=lt, p_mult=pm, init=None, save=None, 
              dropout=0.2, lstm_units=1024, dense_units=30,saving=True)
    net.test(data, trait, batch_shape=(n_ex, ex_len))
    K.backend.clear_session()



main(data='./', n_epochs=20, n_ex=512, ex_len=30, lt='v_adv', pm=0.02)