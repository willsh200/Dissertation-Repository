import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers
from tensorflow.keras.optimizers import Adam
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args

# Load CIFAR-100 dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar100.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

param_space = [
    Real(1e-6, 0.1, name='learning_rate', prior='log-uniform'),
    Real(1e-6, 1e-3, name='weight_decay', prior='log-uniform')
]

# Open a log file to record hyperparameters and accuracies
with open('hyperparameter_optimization_log.txt', 'w') as f:
   f.write("Trial\tLearning Rate\tWeight Decay\tValidation Accuracy\n")


def resnet_block(input_layer, filters, kernel_size=3, strides=1, activation='relu'):
    x = layers.Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same')(input_layer)
    x = layers.Activation(activation)(x)
    x = layers.Conv2D(filters, kernel_size=kernel_size, strides=1, padding='same')(x)
    return layers.Add()([x, input_layer])

# Define the custom ResNet model without batch normalization
def create_custom_resnet_model(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, kernel_size=7, strides=2, padding='same')(inputs)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    for filters in [64, 128, 256, 512]:
        for _ in range(3):
            x = resnet_block(x, filters)
        x = layers.Conv2D(filters * 2, kernel_size=1, strides=2, padding='same')(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=x)
    return model

@use_named_args(param_space)
def objective(learning_rate, weight_decay):
    # Create the model
    model = create_custom_resnet_model(
        input_shape=(32, 32, 3),
        num_classes=100
    )
    
    # Compile the model
    optimizer = Adam(learning_rate=learning_rate, weight_decay = weight_decay)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    history = model.fit(train_images, train_labels,
                        epochs=50,
                        validation_data=(val_images, val_labels),
                        verbose=2)

    # Get the validation accuracy
    val_acc = history.history['val_accuracy'][-1]
    
 # Log the hyperparameters and accuracy
    with open('hyperparameter_optimization_log.txt','w') as f:
        f.write(f"{objective.iteration}\t{learning_rate}\t{weight_decay}\t{val_acc}\n")
    objective.iteration += 1
    
    # We want to maximize accuracy, so we return the negative accuracy as the objective function's value
    return -val_acc

# Initialize iteration counter
objective.iteration = 1

result = gp_minimize(objective,
                     dimensions=param_space,
                     n_calls=40,
                     random_state=42)

# Get the best hyperparameters
best_hyperparams = result.x
best_score = -result.fun

with open('best_hyperparameters.txt','w') as f:
   f.write(f"The optimal hyperparameters are:\n- Learning rate: {best_hyperparams[0]}\n- Weight Decay: {best_hyperparams[1]}")
