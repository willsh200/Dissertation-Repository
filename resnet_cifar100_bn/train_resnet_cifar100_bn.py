import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers
from tensorflow.keras.optimizers import Adam
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

# Load CIFAR-100 dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar100.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Define the ResNet model
def create_resnet_model():
    base_model = tf.keras.applications.ResNet50(
        weights=None, 
        input_shape=(32, 32, 3), 
        classes=100
    )
    model = models.Sequential()
    model.add(base_model)
    return model

model = create_resnet_model()

# Compile the model
model.compile(optimizer=Adam(learning_rate=2.4e-4, weight_decay = 0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model with the augmented data
history = model.fit(train_images,
                    train_labels,
                    epochs=500,
                    validation_data=(test_images, test_labels))


# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

# Save the model
model.save('resnet_cifar100_model_500.h5')

# Save the test accuracy to a file
with open('model_metrics_500.txt', 'w') as f:
    f.write(f"Test accuracy: {test_acc}\n")

# Plot training & validation accuracy values
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('accuracy_plot_500.png')

# Get the predicted probabilities
y_pred = model.predict(test_images)
y_true = test_labels

# Convert class indices to one-hot encodings
y_true_one_hot = tf.keras.utils.to_categorical(y_true, num_classes=100)

# Compute calibration curve
prob_true, prob_pred = calibration_curve(y_true_one_hot.ravel(), y_pred.ravel(), n_bins=10)

# Plot calibration curve
plt.figure()
plt.plot(prob_pred, prob_true, marker='o', label='ResNet Model')
plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
plt.xlabel('Mean predicted probability')
plt.ylabel('Fraction of positives')
plt.title('Calibration plot')
plt.legend()
plt.savefig('calibration_plot_500.png')

# Calculate calibration error (ECE - Expected Calibration Error)
def expected_calibration_error(prob_true, prob_pred):
    return np.sum(np.abs(prob_pred - prob_true) * np.histogram(prob_pred, bins=10, range=(0, 1))[0]) / len(prob_true)

ece = expected_calibration_error(prob_true, prob_pred)

# Save the Expected Calibration Error (ECE) to a file
with open('model_metrics_500.txt', 'a') as f:
    f.write(f"Expected Calibration Error (ECE): {ece}\n")
