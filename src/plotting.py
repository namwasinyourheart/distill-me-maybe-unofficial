import matplotlib.pyplot as plt

def plot_metrics(metrics):
    epochs = range(1, len(metrics["accuracy"]) + 1)
    plt.plot(epochs, metrics["accuracy"], label="Accuracy")
    plt.plot(epochs, metrics["f1"], label="F1 Score")
    plt.xlabel("Epochs")
    plt.ylabel("Score")
    plt.legend()
    plt.show()
