import tensorflow as tf
from keras.applications import EfficientNetB3, EfficientNetB4, EfficientNetB5, InceptionV3
from keras.layers import GlobalAveragePooling2D, Dense, Concatenate, Reshape, Multiply
from keras.models import Model, Sequential
from keras.losses import CategoricalCrossentropy
from keras.metrics import CategoricalAccuracy
import keras.backend as K
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import pathlib



directory = pathlib.Path('test_data').with_suffix('')

img_size = 150


datagen_train = ImageDataGenerator(
    rescale = 1./255,
    rotation_range=20,  # rotate images randomly within the range of 20 degrees
    zoom_range=0.2,  # apply zoom augmentation with a range of 0.2
    horizontal_flip=True  # randomly flip the images horizontally
)

datagen_val = ImageDataGenerator(
    rescale = 1./255
)

train = tf.keras.utils.image_dataset_from_directory(
    directory,
    subset="training",
    validation_split=0.2,
    image_size=(img_size, img_size),
    seed=1415,
    batch_size=128,)


valid = tf.keras.utils.image_dataset_from_directory(
    directory,
    subset="validation",
    seed=1415,
    validation_split=0.2,
    image_size=(img_size, img_size),
    batch_size=128,)


train= train.map(lambda x, y: (x, tf.one_hot(y, 21)))

valid = valid.map(lambda x, y: (x, tf.one_hot(y, 21)))
# Define the backbone models
backbone_models = {
    'EfficientNetB3': EfficientNetB3,
    'EfficientNetB4': EfficientNetB4,
    'EfficientNetB5': EfficientNetB5,
    'InceptionV4': InceptionV3
}

# Define the input shape and backbone model to use
input_shape = (img_size, img_size, 3)
backbone_model_name = 'EfficientNetB5'

# Create the backbone model
backbone = backbone_models[backbone_model_name](input_shape=input_shape, include_top=False, weights='imagenet')
backbone.trainable = False
# Define the local features extraction layers
block4_features = backbone.get_layer('block4a_project_bn').output
block5_features = backbone.get_layer('block5a_project_bn').output
block6_features = backbone.get_layer('block6a_project_bn').output

def osme_block(x, reduction_ratio=16):
    filters = K.int_shape(x)[-1]
    squeeze = GlobalAveragePooling2D()(x)
    excitation = Dense(filters // reduction_ratio, activation='relu')(squeeze)
    excitation = Dense(filters, activation='sigmoid')(excitation)
    excitation = Reshape((1, 1, filters))(excitation)
    scaled_features = Multiply()([x, excitation])
    return scaled_features

block4_features = osme_block(block4_features)
block5_features = osme_block(block5_features)
block6_features = osme_block(block6_features)

block4_pool = GlobalAveragePooling2D()(block4_features)
block5_pool = GlobalAveragePooling2D()(block5_features)
block6_pool = GlobalAveragePooling2D()(block6_features)

# Concatenate the local features
local_features = Concatenate()([block4_pool, block5_pool, block6_pool])

# Define the "head" layers
head = Dense(512, activation='relu')(local_features)
head_output = Dense(21, activation='softmax')(head)  # Replace `num_classes` with the number of output classes

# Create the final model
model1 = Model(inputs=backbone.input, outputs=head_output)
model2 = Sequential()
model2.add(backbone)
# Set up the loss function
loss_function = CategoricalCrossentropy()

# Set up the optimizer
optimizer = Adam()

# Compile the model
model1.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])

model1.fit(
  train,
  validation_data=valid,
  epochs=30
)
model1.save('my_model.h5')
