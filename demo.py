import mnist_loader
import network
import time
import random
import numpy as np

if __name__ == "__main__":
    seed = 1
    random.seed(seed)
    np.random.seed(seed)

    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net = network.Network([784, 30, 10])

    t0 = time.time()
    net.SGD(
        training_data=training_data,
        epochs=30,
        mini_batch_size=10,
        learning_rate=1.8,
        test_data=test_data,
        # full_batch=True
    )
    print("Training completed in {} seconds".format(int(time.time() - t0)))
