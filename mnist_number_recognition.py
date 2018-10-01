import numpy as np
import keras
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten
from keras.models import Model
from keras.datasets import mnist
from scipy.io import loadmat

class NumberRecognition:
    '''
    Implementing a model that will train to classify whether two MNIST digits
    are the same or different
    '''
    def __init__(self, model_name='sample_number_recognition_model', epochs=5, batch_size=32):
        '''

        :param model_name: name of the model
        :param epochs: number of epochs
        :param batch_size: size of batches
        '''
        self._model_name = model_name
        self._epochs = epochs
        self._batch_size = batch_size
        self._train = {}
        self._test = {}
        self._classification_model = None
        self._create_data_set()
        self._define_model()
        return

    def _create_data_set(self):
        '''
        Create data set for pairs of images and whether the two digits match or not
        '''
        def one_hot_encode(labels):
            '''
            converts 0-9 number to one hot encoded vector
            '''

            one_hot = np.zeros((labels.shape[0], 10))
            one_hot[np.arange(labels.shape[0]), labels] = 1
            return one_hot
        #load MNIST data
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        #create pairs of images with label
        self._train['images'] = np.reshape(x_train, (-1, 28, 28, 1))
        self._train['labels'] = one_hot_encode(y_train)
        self._test['images'] = np.reshape(x_test, (-1, 28, 28, 1))
        self._test['labels'] = one_hot_encode(y_test)
        return

    def _define_model(self):
        '''
        define the model to be trained
        Original code taken from:
        https://keras.io/getting-started/functional-api-guide/
        '''
        # First, define the vision modules
        digit_input = Input(shape=(28, 28, 1))
        hidden_layer = Flatten()(digit_input)
        hidden_layer = Dense(784, activation='relu')(hidden_layer)
        out = Dense(10, activation='relu')(hidden_layer)

        self._classification_model = Model(digit_input, out)
        self._classification_model.compile(optimizer='rmsprop',
                                           loss='mean_squared_error',
                                           metrics=['binary_accuracy']
                                          )
        return

    def train_model(self):
        '''
        train the model
        '''
        self._classification_model.fit(self._train['images'],
                                       self._train['labels'], epochs=self._epochs, batch_size=self._batch_size,
                                       validation_data=(self._test['images'], self._test['labels']))
        return

    def evaluate_model(self):
        '''
        Evaluate model using test set data
        :return:
        '''
        score = self._classification_model.evaluate(self._test['images'], self._test['labels'])
        print('Test loss: {}'.format(score[0]))
        print('Test accuracy: {}'.format(score[1]))
        return

    def save_model(self, save_path=''):
        '''
        save the model

        :param save_path: path to save the model at
        :return:
        '''
        self._classification_model.save(save_path+self._model_name+'.h5')
        print('Model saved to '+save_path+self._model_name+'.h5')
        return

    def load_model(self, model_path):
        '''
        loads a model

        :param model_path: path to the model
        :return:
        '''
        self._classification_model = keras.models.load_model(model_path)
        return

if __name__ == '__main__':
    NumberRecognitionInstance = NumberRecognition()
    NumberRecognitionInstance.train_model()
    NumberRecognitionInstance.evaluate_model()
    NumberRecognitionInstance.save_model()
