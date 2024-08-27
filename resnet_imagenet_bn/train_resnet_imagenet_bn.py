import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

TRAIN_DIR = '/home/uczcwgt/Scratch/ILSVRC/Data/CLS-LOC/train'
VAL_DIR = '/home/uczcwgt/Scratch/ILSVRC/Data/CLS-LOC/test'

# Preprocess the data
IMG_SIZE = 224
BATCH_SIZE = 64

train_datagen = ImageDataGenerator(rescale=1.0/255)

val_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='sparse'
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='sparse'
)

# Define the ResNet model
def create_resnet_model():
    base_model = tf.keras.applications.ResNet50(
        weights='imagenet',
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        classes=1000
    )
    model = models.Sequential()
    model.add(base_model)
    return model

model = create_resnet_model()

# Compile the model
model.compile(optimizer=Adam(learning_rate=1e-5),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_generator,
                    epochs=10,
                    validation_data=val_generator)

# Evaluate the model
test_loss, test_acc = model.evaluate(val_generator, verbose=2)

# Save the model
model.save('resnet_imagenet_model.h5')

# Save the test accuracy to a file
with open('model_metrics.txt', 'w') as f:
    f.write(f"Test accuracy: {test_acc}\n")

# Plot training & validation accuracy values
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('accuracy_plot.png')

# Get the predicted probabilities
val_generator.reset()
y_pred = model.predict(val_generator)
y_true = np.concatenate([y for x, y in val_generator], axis=0)

# Convert class indices to one-hot encodings
y_true_one_hot = tf.keras.utils.to_categorical(y_true, num_classes=1000)

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
plt.savefig('calibration_plot.png')

# Calculate calibration error (ECE - Expected Calibration Error)
def expected_calibration_error(prob_true, prob_pred):
    return np.sum(np.abs(prob_pred - prob_true) * np.histogram(prob_pred, bins=10, range=(0, 1))[0]) / len(prob_true)

ece = expected_calibration_error(prob_true, prob_pred)

# Save the Expected Calibration Error (ECE) to a file
with open('model_metrics.txt', 'a') as f:
    f.write(f"Expected Calibration Error (ECE): {ece}\n")

