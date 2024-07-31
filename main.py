import matplotlib.pyplot as plt
from src.data.preprocessing import create_spiral_data
from src.models.multilayer_perceptron import build_model


if __name__ == "__main__":
    X, y = create_spiral_data(samples=100, classes=3)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="brg")
    plt.show()

    build_model(X, y)
