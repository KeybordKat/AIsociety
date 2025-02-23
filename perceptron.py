import numpy as np
import matplotlib.pyplot as plt


def perceptron(X, y, T):
    theta = np.zeros(X.shape[1])
    theta_zero = 0
    n_samples = X.shape[0]

    for t in range(T):
        for i in range(n_samples):
            x = X[i]
            label = y[i]

            # Update weights and bias if misclassified
            if label * (np.dot(theta, x) + theta_zero) <= 0:
                theta = theta + label * x
                theta_zero = theta_zero + label

    return theta, theta_zero


# Generate random data to train on -> 2 features, 30 samples
X = np.random.uniform(low=-10, high=10, size=(30, 2))
# Create labels based on a linear decision boundary
y = np.sign(np.dot(X, np.array([2, 4])) + 3)

T = 40
theta, theta_zero = perceptron(X, y, T)  # Run Perceptron algorithm

# Plot the graph
colors = np.choose(y > 0, np.transpose(np.array(['r', 'b']))).flatten()
fig, ax = plt.subplots()
ax.scatter(X[:, 0], X[:, 1], c=colors, marker='o')

# Plot the hyperplane (decision boundary)
y_intercept = -theta_zero / theta[1]
slope = -theta[0] / theta[1]

xmin, xmax = -12, 12
point1 = slope * xmin + y_intercept
point2 = slope * xmax + y_intercept

# Plot the decision boundary line
ax.plot([xmin, xmax], [point1, point2], color="green", label="Decision Boundary")

ax.set_xlim([xmin, xmax])
ax.set_ylim([-12, 12])
ax.set_title("Perceptron Classifier")
ax.legend()

plt.show()
