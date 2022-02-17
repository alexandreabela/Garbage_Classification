from tensorflow.keras.optimizers import Adam

# General data parameters
DATASET_PATH = 'dataset-resized'
SHUFFLE_DATA = True

# Data generator parameters
TRAINING_BATCH_SIZE = 16
TRAINING_IMAGE_SIZE = (224, 224)
VALIDATION_BATCH_SIZE = 16
VALIDATION_IMAGE_SIZE = (224, 224)
TESTING_BATCH_SIZE = 16
TESTING_IMAGE_SIZE = (224, 224)
NUMBER_OF_CHANNELS = 3
TRANSFORM = True

## Model parameters
loss_function = 'categorical_crossentropy'
no_classes = 6
no_epochs = 50
optimizer = Adam()
verbosity = 1