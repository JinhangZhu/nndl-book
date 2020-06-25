import random
import time

import numpy as np

import mnist_loader
import network
import network2
import utils

if __name__ == "__main__":
    # Read the training data and set random seed
    seed = 1
    random.seed(seed)
    np.random.seed(seed)

    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

    # network.py demo
    #
    # net = network.Network([784, 30, 10])

    # t0 = time.time()
    # net.SGD(
    #     training_data=training_data,
    #     epochs=30,
    #     mini_batch_size=10,
    #     learning_rate=1.8,
    #     test_data=test_data,
    #     full_batch=True
    # )
    # print("Training completed in {} seconds".format(int(time.time() - t0)))

    # network2.py demo
    #
    net = network2.Network(
        sizes=[784, 30, 10],
        cost=network2.CrossEntropyCost
    )
    t0 = time.time()
    evaluation_cost, evaluation_accuracy, training_cost, training_accuracy = net.SGD(
        training_data=training_data,
        epochs=30,
        mini_batch_size=10,
        eta=0.3,
        # lmbda=5.0,
        evaluation_data=test_data,
        full_batch=True,
        monitor_evaluation_accuracy=True
    )
    print("Training completed in {} seconds".format(int(time.time() - t0)))
    net.save('net-cross-entropy.json')
    utils.plot_results(evaluation_accuracy)
