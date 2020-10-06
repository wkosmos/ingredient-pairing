import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import SGD, Adam
from sklearn.model_selection import train_test_split
import pickle

def make_MLP_model(X, nn_hl, activ='relu', opt='adam'):

    num_coef = 2

    model = Sequential()

    model.add(Dense(units=nn_hl,
                    input_shape=(num_coef,),
                    activation=activ, 
                    use_bias=True, 
                    kernel_initializer='glorot_uniform', 
                    bias_initializer='zeros', 
                    kernel_regularizer=None, 
                    bias_regularizer=None, 
                    activity_regularizer=None, 
                    kernel_constraint=None, 
                    bias_constraint=None))
    model.add(Dense(units=nn_hl,
                    input_shape=(num_coef,),
                    activation=activ, 
                    use_bias=True, 
                    kernel_initializer='glorot_uniform', 
                    bias_initializer='zeros', 
                    kernel_regularizer=None, 
                    bias_regularizer=None, 
                    activity_regularizer=None, 
                    kernel_constraint=None, 
                    bias_constraint=None))                
    model.add(Dense(units=1,
                    activation='linear', 
                    use_bias=True, 
                    kernel_initializer='glorot_uniform')) 
    
    model.compile(loss="mean_squared_error", optimizer=opt, metrics=["mse"] )

    return model


if __name__ == "__main__":
    
    # load data
    with open('data/existing_combs.pickle', 'rb') as data:
        existing_combs = pickle.load(data)
        X = np.asarray(existing_combs)
        
    
    with open('data/existing_combs_counts.pickle', 'rb') as data:
        existing_combs_counts = pickle.load(data)
        temp = [i[1] for i in existing_combs_counts]
        y = np.asarray(temp)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    sgd = SGD(lr=1.0, decay=1e-7, momentum=.9) # using stochastic gradient descent

    nn_hl = 16 
    num_epochs = 1000 
    batch_size = 32 
    mlp = make_MLP_model(X_train, nn_hl)

    checkpoint_filepath = '/data/checkpoint'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_r2',
        mode='max',
        save_best_only=True)
    mlp.fit(X_train, y_train, batch_size=batch_size, epochs=num_epochs, verbose=1, shuffle=True)
    y_pred = mlp.predict(X_test)
    np.savetxt('y_pred_known_combs.csv', y_pred, delimiter=',')


    mlp.save('src/trained_MLP')
    