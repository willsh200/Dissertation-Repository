import matplotlib.pyplot as plt
import math

class AlphaDecay:
    def __init__(self, total_epochs):
        self.total_epochs = total_epochs

    def get_alpha(self, current_epoch):
        if current_epoch >= self.total_epochs:
            return 0.0
        alpha = 1.0 - math.log(1 + current_epoch) / math.log(1 + self.total_epochs)
        return alpha

# Parameters
total_epochs = 100
decay = AlphaDecay(total_epochs)

# Calculate alpha values
epochs = list(range(total_epochs + 1))
alpha_values = [decay.get_alpha(epoch) for epoch in epochs]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(epochs, alpha_values, label="Alpha Decay (Inverse Log)", color="blue")
plt.xlabel("Epoch")
plt.ylabel("Alpha")
plt.title("Alpha Decay Over 100 Epochs")
plt.grid(True)
plt.legend()

# Save the plot to a file
plt.savefig("alpha_decay_plot.png", dpi=300, bbox_inches='tight')

# Display the plot
plt.show()

