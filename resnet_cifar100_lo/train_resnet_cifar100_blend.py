import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers
from tensorflow.keras.optimizers import Adam
from sklearn.calibration import calibration_curve
import numpy as np
import matplotlib.pyplot as plt
import math

# Load CIFAR-100 dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar100.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Custom layer to blend outputs from Batch Normalized and non-Batch Normalized paths
class BlendBNLayer(layers.Layer):
    def __init__(self, total_epochs, **kwargs):
        super(BlendBNLayer, self).__init__(**kwargs)
        self.total_epochs = total_epochs
        self.bn = layers.BatchNormalization(**kwargs)

    def call(self, inputs, training=None, current_epoch=0):
        alpha = 1.0 - math.log(1 + current_epoch) / math.log(1 + self.total_epochs)
        alpha = tf.cast(alpha, tf.float32)

        # Path with Batch Normalization
        bn_output = self.bn(inputs, training=training)
        # Path without Batch Normalization
        non_bn_output = inputs

        # Blend the outputs
        return alpha * bn_output + (1 - alpha) * non_bn_output

# Define a custom ResNet block with blending
def resnet_block(input_layer, filters, kernel_size=3, strides=1, total_epochs=100, current_epoch=0):
    x = layers.Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same')(input_layer)
    x = BlendBNLayer(total_epochs=total_epochs)(x, training=True, current_epoch=current_epoch)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters, kernel_size=kernel_size, strides=1, padding='same')(x)
    return layers.Add()([x, input_layer])

# Define the custom ResNet model with blending
def create_custom_resnet_model(input_shape, num_classes, total_epochs=100):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, kernel_size=7, strides=2, padding='same')(inputs)
    x = BlendBNLayer(total_epochs=total_epochs)(x, training=True, current_epoch=0)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    current_epoch = 0
    for filters in [64, 128, 256, 512]:
        for _ in range(3):
            x = resnet_block(x, filters, total_epochs=total_epochs, current_epoch=current_epoch)
        x = layers.Conv2D(filters * 2, kernel_size=1, strides=2, padding='same')(x)
        current_epoch += 1

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=x)
    return model

# Total number of epochs
total_epochs = 500

model = create_custom_resnet_model(input_shape=(32, 32, 3), num_classes=100, total_epochs=total_epochs)

# Compile the model
model.compile(optimizer=Adam(learning_rate=2.4e-4, weight_decay=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

# List to store alpha values over epochs
alpha_values = []

# Custom callback to track and store alpha values during training
class AlphaCallback(tf.keras.callbacks.Callback):
    def __init__(self, total_epochs):
        super(AlphaCallback, self).__init__()
        self.total_epochs = total_epochs

    def on_epoch_end(self, epoch, logs=None):
        alpha = 1.0 - math.log(1 + epoch) / math.log(1 + self.total_epochs)
        alpha_values.append(alpha)

# Train the model
history = model.fit(train_images, train_labels,
    epochs=total_epochs,
    validation_data=(test_images, test_labels),
    callbacks=[AlphaCallback(total_epochs)],
    verbose=2)

# Save the model
model.save(f'resnet_blend_bn_cifar100_logalpha_500.h5')

# Plot training & validation accuracy values
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title(f'Model accuracy with Blending Batch Normalization')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig(f'accuracy_plot_blend_bn_logalpha_500.png')

# Plot alpha values over epochs
plt.figure()
plt.plot(range(total_epochs), alpha_values)
plt.title('Alpha values over epochs')
plt.xlabel('Epoch')
plt.ylabel('Alpha')
plt.savefig('alpha_values_over_epochs_log_500.png')

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

# Save the test accuracy to a file
with open('model_metrics_blend_bn_logalpha_500.txt', 'w') as f:
    f.write(f"Test accuracy: {test_acc}\n")

# Get the predicted probabilities
y_pred = model.predict(test_images)
y_true = test_labels

# Convert class indices to one-hot encodings
y_true_one_hot = tf.keras.utils.to_categorical(y_true, num_classes=100)

# Compute calibration curve
prob_true, prob_pred = calibration_curve(y_true_one_hot.ravel(), y_pred.ravel(), n_bins=10)

# Plot calibration curve
plt.figure()
plt.plot(prob_pred, prob_true, marker='o', label='ResNet Model with Blended BN')
plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
plt.xlabel('Mean predicted probability')
plt.ylabel('Fraction of positives')
plt.title('Calibration plot with Blended Batch Normalization')
plt.legend()
plt.savefig('calibration_plot_blend_bn_logalpha_500.png')

# Calculate calibration error (ECE - Expected Calibration Error)
def expected_calibration_error(prob_true, prob_pred):
    return np.sum(np.abs(prob_pred - prob_true) * np.histogram(prob_pred, bins=10, range=(0, 1))[0]) / len(prob_true)

ece = expected_calibration_error(prob_true, prob_pred)

# Save the Expected Calibration Error (ECE) to a file
with open('model_metrics_blend_bn_logalpha_500.txt', 'a') as f:
    f.write(f"Expected Calibration Error (ECE): {ece}\n")

