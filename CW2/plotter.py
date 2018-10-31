import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

_1a_left = -1
_1a_right = 1.2

def get_data():
    N = 25
    X = np.reshape(np.linspace(0, 0.9, N), (N, 1))
    Y = np.cos(10*X**2) + 0.1 * np.sin(100*X)
    return X,Y

def linear_regression(X, Y):
    # plot_points(X, Y, _1a_left, _1a_right)
    # Polynomail Linear Regression
    # for K in [0, 1, 2, 3, 11]:
    #     # train
    #     phi = init_phi(X, K, 25)
    #     tmp = np.dot(phi.T, phi)
    #     inv_tmp = np.linalg.inv(tmp)
    #     theta = np.dot(np.dot(inv_tmp, phi.T), Y)
    #
    #     # predict
    #     x_plot = np.linspace(-0.3, 1.3, 200)
    #     phi_plot = init_phi(x_plot, K, 200)
    #     y_plot = np.dot(theta.T, phi_plot.T).reshape(200)
    #     plot_curve(x_plot, y_plot, _1a_left, _1a_right, "order = {}".format(K))
    # plt.title("Polynomial Linear Regression")
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.legend()
    # plt.show()

    # Trignometric
    for K in [1, 11]:
        # train
        tri = init_tri(X, K, 25)
        tmp = np.dot(tri.T, tri)
        inv_tmp = np.linalg.inv(tmp)
        theta = np.dot(np.dot(inv_tmp, tri.T), Y)

        # predict
        x_plot = np.linspace(-0.3, 1.3, 200)
        tri_plot = init_tri(x_plot, K, 200)
        y_plot = np.dot(theta.T, tri_plot.T).reshape(200)
        plot_curve(x_plot, y_plot, _1a_left, _1a_right, "order = {}".format(K))
    plt.title("Linear Regression with trigonometric basis functions")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

    # overfitting
    # N = 25
    # errors = np.zeros(11)
    # for K in range(11):
    #     error = 0
    #     for i in range(N):
    #         X_train = np.delete(X, i, 0)
    #         y_train = np.delete(Y, i, 0)
    #         X_test = X[i]
    #         y_test = Y[i]
    #
    #         tri = init_tri(X_train, K, N-1)
    #         tmp = np.dot(tri.T, tri)
    #         inv_tmp = np.linalg.inv(tmp)
    #         theta = np.dot(np.dot(inv_tmp, tri.T), y_train)
    #         #test
    #         tri_test = init_tri(X_test, K, 1)
    #         y_predict = np.dot(theta.T, tri_test.T).reshape(1)
    #         error += (y_predict - y_test)**2
    #     errors[K] = np.sqrt(error/N)
    # sigma = errors**2
    # plot_error(sigma, errors, np.arange(11)+1)
    # plt.title('overfitting with LOO')
    # plt.legend()
    # plt.xlabel('Order')
    # plt.show()


def plot_error(sigma, error, X):
    plt.plot(X, error, label='average squared error')
    plt.plot(X, sigma, label='maximum likelihood sigma^2')
    plt.ylim((0,20))


def init_tri(X, K, N):
    tri = np.zeros((2*K+1, N))
    tri[0] = 1
    for i in range(2*K):
        if i+1 % 2 == 0:
            tri[i+1] = np.cos(2*np.pi*(i+1)*X).reshape(N)
        else:
            tri[i+1] = np.sin(2*np.pi*(i+1)*X).reshape(N)
    return tri.T

def init_phi(X, K, N):
    phi = np.zeros((K+1, N))
    phi[0] = 1
    for i in range(K):
        phi[i+1] = (X**(i+1)).reshape(N)
    return phi.T

def plot_curve(X, Y, x_left, x_right, label):
    plt.plot(X, Y, label=label)
    plt.xlim((x_left, x_right))
    plt.ylim((-5,5))

def plot_points(X, Y, x_left, x_right):
    plt.scatter(X, Y, c='black', marker='.')
    plt.xlim((x_left, x_right))


if __name__ == '__main__':
    X, Y = get_data()
    linear_regression(X, Y)
