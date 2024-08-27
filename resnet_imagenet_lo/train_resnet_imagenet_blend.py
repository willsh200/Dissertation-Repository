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

class BlendBNLayer(layers.Layer):
    def __init__(self, total_epochs, **kwargs):
        super(BlendBNLayer, self).__init__(**kwargs)
        self.total_epochs = total_epochs
        self.bn = layers.BatchNormalization(**kwargs)

    def call(self, inputs, training=None, current_epoch=0):
        alpha = 1.0 - (current_epoch / self.total_epochs)
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
total_epochs = 10

# ImageNet specifics
num_classes = 1000  # ImageNet has 1000 classes
input_shape = (224, 224, 3)  # Adjusted input shape for ImageNet

# Create the model
model = create_custom_resnet_model(input_shape=input_shape, num_classes=num_classes, total_epochs=total_epochs)

# Compile the model
model.compile(optimizer=Adam(learning_rate=2.5e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

# List to store alpha values over epochs
alpha_values = []

# Custom callback to track and store alpha values during training
class AlphaCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        alpha = 1.0 - (epoch / total_epochs)
        alpha_values.append(alpha)

# Train the model
history = model.fit(train_generator,
    epochs=total_epochs,
    validation_data=val_generator,
    callbacks=[AlphaCallback()],
    verbose=2)

# Save the model
model.save(f'resnet_blend_bn_imagenet.h5')

# Plot training & validation accuracy values
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title(f'Model accuracy with Blending Batch Normalization')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig(f'accuracy_plot_blend_bn.png')

# Plot alpha values over epochs
plt.figure()
plt.plot(range(total_epochs), alpha_values)
plt.title('Alpha values over epochs')
plt.xlabel('Epoch')
plt.ylabel('Alpha')
plt.savefig('alpha_values_over_epochs.png')

# Evaluate the model on validation data
val_loss, val_acc = model.evaluate(val_generator, verbose=2)

# Save the validation accuracy to a file
with open('model_metrics_blend_bn.txt', 'w') as f:
    f.write(f"Validation accuracy: {val_acc}\n")

# Get the predicted probabilities
y_pred = model.predict(val_generator)
y_true = np.concatenate([y for x, y in val_generator], axis=0)

# Convert class indices to one-hot encodings
y_true_one_hot = tf.keras.utils.to_categorical(y_true, num_classes=num_classes)

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
plt.savefig('calibration_plot_blend_bn.png')

# Calculate calibration error (ECE - Expected Calibration Error)
def expected_calibration_error(prob_true, prob_pred):
    bin_centers = np.linspace(0, 1, len(prob_pred))
    ece = np.sum(np.abs(prob_pred - prob_true) * np.histogram(prob_pred, bins=10, range=(0, 1))[0]) / len(prob_true)
    return ece

ece = expected_calibration_error(prob_true, prob_pred)

# Save the Expected Calibration Error (ECE) to a file
with open('model_metrics_blend_bn.txt', 'a') as f:
    f.write(f"Expected Calibration Error (ECE): {ece}\n")
