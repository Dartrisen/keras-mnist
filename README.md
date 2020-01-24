# keras-mnist

Classifier for handwritten digits from the famous MNIST dataset

## Instructions

1. Setup your environment.
Let’s first download some packages we’ll need:

``pip install keras tensorflow numpy matplotlib``
Note: We need to install tensorflow because we’re going to run Keras on a TensorFlow backend.
2. Clone or download this project
3. Run ``python cnn.py`` to train your CNN.
5. You could use pre-trained model very simply (make sure you have installed h5py):

``from keras.models import load_model

model = load_model('model.h5')``
