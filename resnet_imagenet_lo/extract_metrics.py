import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

class BlendBNLayer(tf.keras.layers.Layer):
    def __init__(self, total_epochs, **kwargs):
        super(BlendBNLayer, self).__init__(**kwargs)
        self.total_epochs = total_epochs
        self.bn = tf.keras.layers.BatchNormalization(**kwargs)

    def call(self, inputs, training=None, current_epoch=0):
        alpha = 1.0 - (current_epoch / self.total_epochs)
        alpha = tf.cast(alpha, tf.float32)

        # Path with Batch Normalization
        bn_output = self.bn(inputs, training=training)
        # Path without Batch Normalization
        non_bn_output = inputs

        # Blend the outputs
        return alpha * bn_output + (1 - alpha) * non_bn_output

# Load model
model = load_model('resnet_blend_bn_imagenet.h5', custom_objects={"BlendBNLayer": BlendBNLayer})

# Directory containing the images
VAL_DIR = '/home/uczcwgt/Scratch/ILSVRC/Data/CLS-LOC/test'

# Preprocess the data
IMG_SIZE = 224
BATCH_SIZE = 64

def preprocess_image(image_path):
    image = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    image = img_to_array(image)
    image /= 255.0  # Rescale image
    return image

# Get all image paths
image_paths = [os.path.join(VAL_DIR, fname) for fname in os.listdir(VAL_DIR) if fname.endswith('.jpeg')]

# Create a dataset
def load_and_preprocess_image(path):
    return preprocess_image(path)

# Create dataset for evaluation
dataset = tf.data.Dataset.from_tensor_slices(image_paths)
dataset = dataset.map(lambda x: tf.numpy_function(load_and_preprocess_image, [x], tf.float32))
dataset = dataset.batch(BATCH_SIZE)

# Evaluate the model on validation data
# Create labels array for evaluation purposes
num_samples = len(image_paths)
dummy_labels = np.zeros((num_samples,))  # Assuming a single class for the example, change as needed
label_dataset = tf.data.Dataset.from_tensor_slices(dummy_labels).batch(BATCH_SIZE)
val_dataset = tf.data.Dataset.zip((dataset, label_dataset))

val_loss, val_acc = model.evaluate(val_dataset, verbose=2)

# Save the validation accuracy to a file
with open('model_metrics_blend_bn.txt', 'w') as f:
    f.write(f"Validation accuracy: {val_acc}\n")

# Get the predicted probabilities
y_pred = model.predict(val_dataset)
y_true = np.concatenate([y for x, y in val_dataset], axis=0)

# Convert class indices to one-hot encodings
num_classes = model.output_shape[-1]  # Get number of classes from model output shape
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

