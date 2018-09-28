import numpy as np
import keras
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten
from keras.models import Model
from keras.utils import to_categorical
from scipy.io import loadmat

class SharedVisionModel:
    '''
    Implementing a model that will train to classify whether two MNIST digits
    are the same or different
    '''
    def __init__(self, data_path, model_name):
        self._model_name = model_name
        self.train = None
        self.test = None
        self._classification_model = None
        self._create_dataset(data_path)
        self._define_model()
        return

    @staticmethod
    def load_data(data_path):
        '''
        load MNIST img and labels
        '''
        # Load the dataset
        MNIST_data = loadmat(data_path)
        d_imgs = {}
        d_labels = {}
        d_imgs['train'] = np.zeros((0, 784))
        d_labels['train'] = np.zeros((0, 1))
        d_imgs['test'] = np.zeros((0, 784))
        d_labels['test'] = np.zeros((0, 1))

        #loop through integers 0 to 9
        for i in range(10):
            num_train_img = MNIST_data['train'+str(i)].shape[0]
            num_test_img = MNIST_data['test'+str(i)].shape[0]
            d_imgs['train'] = np.concatenate((d_imgs['train'], MNIST_data['train'+str(i)]))
            d_labels['train'] = np.concatenate((d_labels['train'], np.ones((num_train_img, 1))*i))
            d_imgs['test'] = np.concatenate((d_imgs['test'], MNIST_data['test'+str(i)]))
            d_labels['test'] = np.concatenate((d_labels['test'], np.ones((num_test_img, 1))*i))
        d_imgs['train'] = np.reshape(d_imgs['train'], (-1, 28, 28, 1))
        d_imgs['test'] = np.reshape(d_imgs['test'], (-1, 28, 28, 1))
        return d_imgs, d_labels

    def _create_dataset(self, data_path):
        '''
        Create dataset for pairs of images and whether the two digits match or not
        '''
        def get_img_pairs_and_labels(imgs, labels):
            '''
            creates image pairs with appropriate label vector
            '''
            data = {}
            num_img = imgs.shape[0]
            #get num_img random digits between 0 and num_img-1
            digit_a_idx = np.random.choice(num_img-1, num_img)
            digit_b_idx = np.random.choice(num_img-1, num_img)
            data['digit_a'] = np.take(imgs, digit_a_idx, axis=0)
            data['digit_b'] = np.take(imgs, digit_b_idx, axis=0)

            #get labels as 0 when digits don't match, 1 otherwise
            labels_a = np.take(labels, digit_a_idx, axis=0)
            labels_b = np.take(labels, digit_b_idx, axis=0)
            data['labels'] = (labels_a == labels_b).astype(int)
            return data

        #first load MNIST data
        d_imgs, d_labels = self.load_data(data_path)
        self._train = get_img_pairs_and_labels(d_imgs['train'], d_labels['train'])
        self._test = get_img_pairs_and_labels(d_imgs['test'], d_labels['test'])
        return

    def _define_model(self):
        '''
        define the model to be trained
        Original code taken from:
        https://keras.io/getting-started/functional-api-guide/
        '''
        # First, define the vision modules
        digit_input = Input(shape=(28, 28, 1))
        hidden_layer = Conv2D(64, (3, 3))(digit_input)
        hidden_layer = Conv2D(64, (3, 3))(hidden_layer)
        hidden_layer = MaxPooling2D((2, 2))(hidden_layer)
        out = Flatten()(hidden_layer)

        vision_model = Model(digit_input, out)

        # Then define the tell-digits-apart model
        digit_a = Input(shape=(28, 28, 1))
        digit_b = Input(shape=(28, 28, 1))

        # The vision model will be shared, weights and all
        out_a = vision_model(digit_a)
        out_b = vision_model(digit_b)

        concatenated = keras.layers.concatenate([out_a, out_b])
        out = Dense(1, activation='sigmoid')(concatenated)

        self._classification_model = Model([digit_a, digit_b], out)
        self._classification_model.compile(optimizer='rmsprop',
                                           loss='binary_crossentropy',
                                           metrics=['accuracy']
                                          )
        return

    def train_model(self):
        '''
        train the model
        '''
        print(self._train['labels'].shape)
        self._classification_model.fit([self._train['digit_a'], self._train['digit_b']],
                                       self._train['labels'], epochs=1)
        return

    def save_model(self, save_path):
        '''
        save the model
        '''
        self._classification_model.save(save_path+self._model_name+'.h5')
        print('Model saved to '+save_path+self._model_name+'.h5')
        return

    def load_model(self, model_path):
        '''
        load a model
        '''
        self._classification_model = keras.models.load_model(model_path)
        return

if __name__ == '__main__':
    MODEL_NAME = 'sample_model'
    DATA_PATH = 'mnist_all.mat'
    SAVE_MODEL_PATH = ''
    SharedVisionClassifier = SharedVisionModel(DATA_PATH, MODEL_NAME)
    SharedVisionClassifier.train_model()
    SharedVisionClassifier.save_model(SAVE_MODEL_PATH)
