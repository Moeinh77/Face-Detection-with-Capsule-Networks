import matplotlib.pyplot as plt


def peek(dataset, num_samples=5):

    # Get the first num_samples iterms from the dataset
    X, y = next(dataset.batch(num_samples))

    plt.figure(figsize=(num_samples * 2, 3))

    for index in range(num_samples):
        plt.subplot(1, num_samples, index + 1)

        sample, label = X[index], y[index]
        size_x, size_y, _ = sample.shape

        plt.imshow(sample.reshape(size_x, size_y), cmap="binary")
        plt.title(label)
        plt.axis("off")

    plt.show()
