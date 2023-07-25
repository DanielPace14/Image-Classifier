import tensorflow as tf
from keras.layers import Input, GlobalAveragePooling2D, Dense
from keras.models import Model
from keras.applications import DenseNet121, InceptionResNetV2, ResNet50,EfficientNetB5
import pathlib
directory = pathlib.Path('Data').with_suffix('')

img_size = 224

train = tf.keras.utils.image_dataset_from_directory(
    directory,
    validation_split=0.2,
    subset="training",
    image_size=(img_size, img_size),
    seed=1415,
    batch_size=64,)


valid = tf.keras.utils.image_dataset_from_directory(
    directory,
    validation_split=0.2,
    subset="validation",
    seed=1415,
    image_size=(img_size, img_size),
    batch_size=64,)


model_type = 'EfficientNetB5'
is_backbone_freezed = False
num_categories = 0
use_bias = True
reinitialize_classifier = False

# Input shape
input_shape = (224, 224, 3)

# Define the backbone model
backbone_models = {
    'densenet121': DenseNet121,
    'inceptionresnetv2': InceptionResNetV2,
    'resnet50': ResNet50,
    'EfficientNetB5': EfficientNetB5,
}

backbone_model = backbone_models[model_type](include_top=False, weights='imagenet', input_shape=input_shape)

# Freeze the backbone layers if required
if is_backbone_freezed:
    backbone_model.trainable = False

# Define the input layer
inputs = Input(shape=input_shape)

# Pass the inputs through the backbone model
x = backbone_model(inputs)

# Global average pooling
x = GlobalAveragePooling2D()(x)

# Define the classifier head
if num_categories > 0:
    x = Dense(num_categories, use_bias=use_bias)(x)

# Create the model
model = Model(inputs=inputs, outputs=x)

# Reinitialize the classifier if required
if reinitialize_classifier:
    model.get_layer('dense').kernel_initializer = tf.keras.initializers.RandomNormal(stddev=0.01)
    if use_bias:
        model.get_layer('dense').bias_initializer = 'zeros'

import tensorflow as tf
from keras.optimizers import Adam
import numpy as np
# ----- Stage 1 -----
num_epochs_stage1 = 75
batch_size_stage1 = 2048
gradient_accumulation_steps_stage1 = 4  # Accumulate gradients every 4 steps
learning_rate_stage1 = 0.001
warmup_epochs_stage1 = 5

steps_per_epoch = len(train)//64
total_steps_stage1 = num_epochs_stage1 * steps_per_epoch


# Define the optimizer with weight decay (AdamW)
optimizer_stage1 = Adam(learning_rate=learning_rate_stage1, decay=0.01)

# Define the learning rate schedule (cosine decay with linear warmup)
lr_schedule_stage1 = tf.keras.optimizers.schedules.CosineDecay(learning_rate_stage1, total_steps_stage1)
#lr_schedule_stage1 = tf.keras.optimizers.schedules.WarmUp(lr_schedule_stage1, warmup_epochs_stage1, initial_learning_rate=0.0)

# Compile the model with the optimizer and loss function
model.compile(optimizer=optimizer_stage1, loss='categorical_crossentropy', metrics=['accuracy'])

def create_balanced_dataset(train_dataset, num_samples):
    balanced_dataset = []
    num_classes = len(train_dataset.class_names)

    for class_index in range(num_classes):
        class_indices = np.where(train_dataset.labels == class_index)[0]
        class_samples = np.random.choice(class_indices, size=num_samples, replace=False)
        class_samples = [(train_dataset.images[i], train_dataset.labels[i]) for i in class_samples]
        balanced_dataset.extend(class_samples)

    # Shuffle the balanced dataset
    np.random.shuffle(balanced_dataset)

    return balanced_dataset

# Create a balanced dataset with randomly sampled images per category
# Replace `train_dataset` with your actual training dataset
balanced_train_dataset_stage1 = create_balanced_dataset(train, num_samples=75)

# Train the model for the specified number of epochs
model.fit(balanced_train_dataset_stage1, epochs=num_epochs_stage1, steps_per_epoch=steps_per_epoch)

# ----- Stage 2 -----
num_epochs_stage2 = 105
batch_size_stage2 = 2048
gradient_accumulation_steps_stage2 = 4  # Accumulate gradients every 4 steps
learning_rate_stage2 = 0.001
warmup_epochs_stage2 = 0
total_steps_stage2 = num_epochs_stage2 * steps_per_epoch

# Define the optimizer with weight decay (AdamW)
optimizer_stage2 = Adam(learning_rate=learning_rate_stage2, weight_decay=0.01)

# Define the learning rate schedule (cosine decay with linear warmup)
lr_schedule_stage2 = tf.keras.optimizers.schedules.CosineDecay(learning_rate_stage2, total_steps_stage2)
lr_schedule_stage2 = tf.keras.optimizers.schedules.WarmUp(lr_schedule_stage2, warmup_epochs_stage2, initial_learning_rate=0.0)

# Compile the model with the optimizer and loss function
model.compile(optimizer=optimizer_stage2, loss='categorical_crossentropy', metrics=['accuracy'])

# Create a balanced dataset with randomly sampled images per category
# Replace `train_dataset` with your actual training dataset
balanced_train_dataset_stage2 = create_balanced_dataset(train, num_samples=25)

# Train the model for the specified number of epochs
model.fit(balanced_train_dataset_stage2, epochs=num_epochs_stage2, steps_per_epoch=steps_per_epoch)
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

# Print the model summary
model.fit(
  train,
  validation_data=valid,
  epochs=5
)
