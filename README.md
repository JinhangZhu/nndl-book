# nndl-book

## Reproduction

Random seed:

```python
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
```

- MSE-based, not matrix-based, mini-batch SGD

```python
net = network.Network([784, 30, 10])
net.SGD(
    training_data=training_data,
    epochs=30,
    mini_batch_size=10,
    learning_rate=1.8,
    test_data=test_data,
    # full_batch=True
)
```



![](\images\mse-non-full-matrix.png)

- MSE-based, matrix-based, mini-batch SGD

```python
net = network.Network([784, 30, 10])
net.SGD(
    training_data=training_data,
    epochs=30,
    mini_batch_size=10,
    learning_rate=1.8,
    test_data=test_data,
    full_batch=True
)
```



![](\images\mse-full-matrix.png)

- Cross-entropy-based, matrix-based, mini-batch SGD

```python
net = network2.Network(
    sizes=[784, 30, 10],
    cost=network2.CrossEntropyCost
)
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
```



![](\images\cross-entropy-matrix.png)