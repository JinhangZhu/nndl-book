import argparse
import random
import time
import json

import numpy as np

import mnist_loader
import network
import network2
import utils


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    # parser.add_argument('--origin', type=str, default='P30__P30_04', help='The name of original data downloaded from Supervisely.')
    # parser.add_argument('--out', type=str, default='P30', help='The name of the output dataset folder.')
    # parser.add_argument('--meta', type=str, default='meta.json', help='The name of the meta file of the data.')
    # parser.add_argument('--shuffle', action='store_true', help='Whether to randomly split image set.')
    # parser.add_argument('--train-size', type=float, default=0.9, help='Percentage of train set.')
    # parser.add_argument('--val-size', type=float, default=0.1, help='Percentage of validation set.')
    opt = parser.parse_args()
    print(opt)

    # Read the training data and set random seed
    seed = opt.seed
    random.seed(seed)
    np.random.seed(seed)

    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    training_data, validation_data, test_data = list(training_data), list(validation_data), list(test_data)

    # network.py demo
    #
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

    # # network2.py demo
    # #
    # #
    # net = network2.Network(
    #     sizes=[784, 30, 10],
    #     cost=network2.CrossEntropyCost
    # )
    # t0 = time.time()
    # evaluation_cost, evaluation_accuracy, training_cost, training_accuracy = net.SGD(
    #     training_data=training_data,
    #     epochs=30,
    #     mini_batch_size=10,
    #     eta=0.3,
    #     # lmbda=5.0,
    #     evaluation_data=test_data,
    #     full_batch=True,
    #     monitor_evaluation_accuracy=True,
    #     monitor_training_accuracy=True
    # )
    # print("Training completed in {} seconds".format(int(time.time() - t0)))
    # results = {}
    # results['Accuracy on the training data'] = utils.make_percentage(training_accuracy, len(training_data))
    # results['Accuracy on the test data'] = utils.make_percentage(evaluation_accuracy, len(test_data))
    # utils.plot_results(results, 0, 30, show_max=True, merge=True)

    # network2.py demo check generalization
    #
    #
    # net = network2.Network(
    #     sizes=[784, 30, 10],
    #     cost=network2.CrossEntropyCost
    # )
    # t0 = time.time()
    # evaluation_cost, evaluation_accuracy, training_cost, training_accuracy = net.SGD(
    #     training_data=training_data[:1000],
    #     epochs=400,
    #     mini_batch_size=10,
    #     eta=0.5,
    #     # lmbda=5.0,
    #     evaluation_data=test_data,
    #     full_batch=True,
    #     monitor_evaluation_accuracy=True,
    #     monitor_training_cost=True
    # )
    # print("Training completed in {} seconds".format(int(time.time() - t0)))
    # results = {}
    # results['Accuracy on the test data'] = evaluation_accuracy
    # results['Cost on the training data'] = training_cost
    # utils.plot_results(results, 200, 400)

    # # network2.py demo regularization
    # #
    # #
    # net = network2.Network(
    #     sizes=[784, 30, 10],
    #     cost=network2.CrossEntropyCost
    # )
    # net.large_weight_initializer()
    # t0 = time.time()
    # evaluation_cost, evaluation_accuracy, training_cost, training_accuracy = net.SGD(
    #     training_data=training_data[:1000],
    #     epochs=400,
    #     mini_batch_size=10,
    #     eta=0.5,
    #     lmbda=0.1,
    #     evaluation_data=test_data,
    #     full_batch=True,
    #     monitor_evaluation_accuracy=True,
    #     monitor_training_cost=True
    # )
    # print("Training completed in {} seconds".format(int(time.time() - t0)))
    # results = {}
    # results['Accuracy on the test data'] = utils.make_percentage(evaluation_accuracy, len(test_data))
    # results['Cost on the training data'] = training_cost
    # utils.plot_results(results, 200, 400)

    # # network2.py demo expanding training set
    # See more_data.py

    # # network2.py demo weight initialization
    # See weight_initialization.py

    # # network2.py demo early stopping
    #
    #
    net = network2.Network(
        sizes=[784, 30, 10],
        cost=network2.CrossEntropyCost
    )
    evaluation_cost, evaluation_accuracy, training_cost, training_accuracy = net.SGD(
        training_data=training_data[:1000],
        epochs=30,
        mini_batch_size=10,
        eta=0.5,
        lmbda=5.0,
        evaluation_data=validation_data,
        full_batch=True,
        monitor_evaluation_accuracy=True,
        monitor_training_cost=True,
        early_stopping_n=10
    )
    results = {}
    results['Accuracy on the test data'] = utils.make_percentage(evaluation_accuracy, len(validation_data))
    results['Cost on the training data'] = training_cost
    utils.plot_results(results, 0, 30)
