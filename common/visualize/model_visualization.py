import matplotlib.pyplot as plt
import numpy as np

def print_dataset_info(title, X, Y):
  """
  Prints a brief summary about the length and shape of the dataset
  Arguments: 
    title: the title of the dataset, e.g., training or testing
    X: the examples matrix
    Y: the labels vector
  """
  print(f"Loading {len(X)} {title} images each with shape {X[0].shape}")
  print(f"Loading {len(Y)} associated labels")

def plot_loss(history, loss_type):
  """Given the history object this function draw the loss Vs. epochs plot"""
  loss = history[loss_type]
  plt.plot(np.arange(len(loss)), loss)
  plt.xlabel("Iteations")
  plt.ylabel(loss_type)

def plot_losses(history):
  """Given the history object this function draw both the train_loss and val_loss Vs. epochs plot"""
  plt.figure(figsize=(15, 10))
  epochs = np.arange(len(history["val_loss"]))
  plt.plot(epochs, history["val_loss"], "r-", label="Validation Loss" )
  plt.plot(epochs, history["loss"], "b-", label="Training Loss")
  plt.legend(loc="best")
  plt.show()

def plot_history(history):
  """Given the history object this function draw train loss, train accuracy, validation loss, and validation accuracy over epochs"""
  plt.figure(figsize=(20, 10))
  epochs = np.arange(len(history["val_loss"]))
  plt.plot(epochs, history["val_loss"], "r-", label="Validation Loss" )
  plt.plot(epochs, history["loss"], "b-", label="Training Loss")
  plt.legend(loc="best")
  plt.show()
  plt.figure(figsize=(20, 10))
  plt.plot(epochs, np.array(history["val_accuracy"]) * 100, "g-", label="Validation Accuracy")
  plt.plot(epochs, np.array(history["accuracy"]) * 100, "c-", label="Training Accuracy")
  plt.legend(loc="best")
  plt.show()