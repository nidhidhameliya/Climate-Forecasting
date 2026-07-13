import numpy as np
import matplotlib.pyplot as plt

preds = np.load("outputs/predictions/test_predictions.npy")
truth = np.load("data/processed/tensors/test_y.npy")

plt.figure()
plt.imshow(preds[0,0,:,:,0], cmap="coolwarm")
plt.colorbar()
plt.title("Predicted")
plt.savefig("outputs/visualizations/predicted.png")

plt.figure()
plt.imshow(truth[0,0,:,:,0], cmap="coolwarm")
plt.colorbar()
plt.title("Ground Truth")
plt.savefig("outputs/visualizations/ground_truth.png")

print("Plots saved.")