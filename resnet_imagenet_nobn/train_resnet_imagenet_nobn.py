import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models, optimizers
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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

# Define the model
num_classes = 1000  # ImageNet has 1000 classes
model = create_custom_resnet_model(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=num_classes)

# Compile the model
model.compile(optimizer=optimizers.Adam(learning_rate=2.75e-5),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_generator, epochs=10, validation_data=val_generator, verbose=2)

# Save the model
model.save('resnet_no_bn_imagenet.h5')

# Plot training & validation accuracy values
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy without Batch Normalization')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('accuracy_plot_no_bn_imagenet.png')

# Evaluate the model
test_loss, test_acc = model.evaluate(val_generator, verbose=2)

# Save the test accuracy to a file
with open('model_metrics_no_bn_imagenet.txt', 'w') as f:
    f.write(f"Test accuracy: {test_acc}\n")

# Get the predicted probabilities
y_pred = model.predict(val_generator.map(lambda x, y: x))
y_true = np.concatenate([y for x, y in val_generator], axis=0)

# Convert class indices to one-hot encodings
y_true_one_hot = tf.keras.utils.to_categorical(y_true, num_classes=num_classes)

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
plt.savefig('calibration_plot_no_bn_imagenet.png')

# Calculate calibration error (ECE - Expected Calibration Error)
def expected_calibration_error(prob_true, prob_pred):
    return np.sum(np.abs(prob_pred - prob_true) * np.histogram(prob_pred, bins=10, range=(0, 1))[0]) / len(prob_true)

ece = expected_calibration_error(prob_true, prob_pred)

# Save the Expected Calibration Error (ECE) to a file
with open('model_metrics_no_bn_imagenet.txt', 'a') as f:
    f.write(f"Expected Calibration Error (ECE): {ece}\n")

