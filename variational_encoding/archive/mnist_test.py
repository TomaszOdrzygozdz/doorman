import keras
import numpy as np

(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
mnist_digits = np.concatenate([x_train, x_test], axis=0)
mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255
mnist_digits_rotated = []

for digit in mnist_digits:
    mnist_digits_rotated.append(np.rot90(digit,1))
mnist_digits_rotated = np.array(mnist_digits_rotated)

mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255

print(mnist_digits['0'])