import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import KFold
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import numpy as np

# Load CIFAR-100 dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar100.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Define a custom ResNet block without batch normalization
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

model = create_custom_resnet_model(input_shape=(32, 32, 3), num_classes=100)

# Compile the model
model.compile(optimizer=Adam(learning_rate=3.78e-5, weight_decay=1e-6),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

# Train the model
history = model.fit(train_images, train_labels,
    epochs=500,
    validation_data=(test_images, test_labels),
    verbose=2)
    
# Save the model
model.save(f'resnet_no_bn_cifar100_500.h5')
    
# Plot training & validation accuracy values
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title(f'Model accuracy without Batch Normalization')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig(f'accuracy_plot_no_bn_500.png')
    
# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

# Save the test accuracy to a file
with open('model_metrics_no_bn_500.txt', 'w') as f:
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
plt.plot(prob_pred, prob_true, marker='o', label='ResNet Model without BN')
plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
plt.xlabel('Mean predicted probability')
plt.ylabel('Fraction of positives')
plt.title('Calibration plot without Batch Normalization')
plt.legend()
plt.savefig('calibration_plot_no_bn_500.png')

# Calculate calibration error (ECE - Expected Calibration Error)
def expected_calibration_error(prob_true, prob_pred):
    return np.sum(np.abs(prob_pred - prob_true) * np.histogram(prob_pred, bins=10, range=(0, 1))[0]) / len(prob_true)

ece = expected_calibration_error(prob_true, prob_pred)

# Save the Expected Calibration Error (ECE) to a file
with open('model_metrics_no_bn_500.txt', 'a') as f:
    f.write(f"Expected Calibration Error (ECE): {ece}\n")

