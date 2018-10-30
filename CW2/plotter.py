import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

_1a_left = -0.3
_1a_right = 1.3

def get_data():
    N = 25
    X = np.reshape(np.linspace(0, 0.9, N), (N, 1))
    Y = np.cos(10*X**2) + 0.1 * np.sin(100*X)
    return X,Y

def linear_regression(X, Y):
    plot_points(X, Y, _1a_left, _1a_right)
    for K in [0, 1, 2, 3, 11]:
        # train
        phi = init_phi(X, K, 25)
        tmp = np.dot(phi.T, phi)
        inv_tmp = np.linalg.inv(tmp)
        theta = np.dot(np.dot(inv_tmp, phi.T), Y)

        # predict
        x_plot = np.linspace(-0.3, 1.3, 200)
        phi_plot = init_phi(x_plot, K, 200)
        y_plot = np.dot(theta.T, phi_plot.T).reshape(200)
        plot_curve(x_plot, y_plot, _1a_left, _1a_right, "order = {}".format(K))
    plt.title("Polynomial Linear Regression")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

def init_phi(X, K, N):
    phi = np.zeros((K+1, N))
    phi[0] = 1
    for i in range(K):
        phi[i+1] = (X**(i+1)).reshape(N)
    return phi.T

def plot_curve(X, Y, x_left, x_right, label):
    plt.plot(X, Y, label=label)
    plt.xlim((x_left, x_right))
    plt.ylim((-2,4))

def plot_points(X, Y, x_left, x_right):
    plt.scatter(X, Y, c='black', marker='.')
    plt.xlim((x_left, x_right))


if __name__ == '__main__':
    X, Y = get_data()
    linear_regression(X, Y)
