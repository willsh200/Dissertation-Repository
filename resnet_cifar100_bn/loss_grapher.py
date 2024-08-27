import re
import matplotlib.pyplot as plt

# File path (update with the correct path to your file)
file_path = 'ResNet_CIFAR100_bn.o827992'

# Lists to store extracted data
epochs = []
loss_values = []

# Regular expression pattern to match the required data
pattern = r'Epoch (\d+)/\d+\n.*? - loss: ([\d.]+)'

# Read and parse the file
with open(file_path, 'r') as file:
    content = file.read()
    matches = re.findall(pattern, content)
    
    for match in matches:
        epoch = int(match[0])
        loss = float(match[1])
        
        epochs.append(epoch)
        loss_values.append(loss)

# Plotting the data
plt.figure(figsize=(10, 6))
plt.plot(epochs, loss_values, marker='o', linestyle='-', color='b')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig('loss_vs_epochs.png')  # Save the plot as a file
plt.show()

