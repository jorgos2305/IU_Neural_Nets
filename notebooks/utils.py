import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from tensorflow import keras

from typing import Optional, List

class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]
label_names = {k : v for k, v in zip(range(len(class_names)), class_names)}

def visualize_fmnist(image_data:np.ndarray, label:int, ax:Optional[Axes]=None):

    if ax is not None:
        ax.imshow(image_data, cmap="binary")
        ax.set_title(f"Class name: {label_names[label]}\nLabel: {label}")
        ax.set_axis_off()

    else:
        raise NotImplementedError("At the moment it is only possible to use this function by passing an Axes object.")

def plot_train_and_val_error(df:pd.DataFrame, axs:List[Axes]):

    metric = None
    for col in df.columns.tolist():
        if "accuracy" in col and not col.startswith("val"):
            metric = col

    if not metric:
        raise ValueError("No metric for accuracy was detected")
    axs[0].set_title("Trainig and Validation Acurracy")
    axs[0].plot(df.index, df[metric], label="Train accuracy")
    axs[0].plot(df.index, df[f"val_{metric}"], linestyle="--", label="Val accuracy")
    axs[0].set_ylim([0,1.1])
    
    axs[1].set_title("Trainig and Validation Loss")
    axs[1].plot(df.index, df["loss"], label="Train loss")
    axs[1].plot(df.index, df["val_loss"], linestyle="--", label="Val loss")

    for ax in axs:
        ax.grid(True, linewidth=0.5, linestyle=":")
        ax.legend(loc="upper right")

# creating a callback to get the activations
class ActivationLogger(keras.callbacks.Callback):
    """
    After each epoch (on_epoch_end), this Logger collects the intermediate activation values
    of the neural network using the weight values of the current training step.

    This is done for a specific size of the training set, it is not necessary to do it with all samples
    """

    def __init__(self, X_probe, activation_model, batch_size=32, every_n_epochs=10):
        super().__init__()
        self.X_probe = X_probe
        self.batch_size = batch_size
        self.every_n_epochs = every_n_epochs
        self.activations = {}
        self.activation_model = activation_model

    def set_init_activations(self, value):
        self.activations[0] = value

    def collect(self):
        epoch_activations = []

        for i in range(0, len(self.X_probe), self.batch_size):

            X_batch = self.X_probe[i:i+self.batch_size]
            outputs = self.activation_model(X_batch, training=True)
            epoch_activations.append(
                [x.numpy() for x in outputs]
            )
        return epoch_activations

    def on_epoch_end(self, epoch, logs=None):
        """
        Collect the value of the current activations after each number of epochs.
        """
        epoch_number = epoch + 1
        if epoch_number % self.every_n_epochs == 0:
            self.activations[epoch_number] = self.collect()