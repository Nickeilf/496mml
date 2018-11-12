import sys
sys.path.append('496differentiation_zl5118')
import answers

import numpy as np
import matplotlib.pyplot as plt



def f2(X,Y):
    a = np.array([1, 0])
    b = np.array([0, -1])
    B = np.array([[3, -1],
                  [-1, 3]])

    first = np.sin((X-1)**2 + Y**2)
    second = 3*(X**2) - 2*X*(Y+1) + 3*(Y+1)**2
    return first+second

def f3(X,Y):
    a = np.array([1, 0])
    b = np.array([0, -1])
    B = np.array([[3, -1],
                  [-1, 3]])
    first = 1.0 - np.exp(-(X-1)**2 - Y**2)
    second = np.exp(-(3*(X**2) - 2*X*(Y+1) + 3*(Y+1)**2))
    third = -1/10*np.log((X**2 + Y**2)/100.0 + 1.0/10000.0)
    return first-second-third

def gradient_descend_f2():
    delta = 0.1
    iterations = 50
    gamma = 0.28

    x = np.arange(-2, 2, delta)
    y = np.arange(-2, 2, delta)
    X, Y = np.meshgrid(x, y)
    Z=f2(X,Y)
    fig, ax = plt.subplots()
    CS = ax.contour(X, Y, Z, 20)
    ax.clabel(CS, inline=1, fontsize=10)
    ax.set_title('Gradient Descent for f2 with iteration={}, gamma={}'.format(iterations,gamma))
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')

    X_prev = np.array([1, -1])
    ax.plot(X_prev[0],X_prev[1],color='black',marker='x',markersize=8)


    for iteration in range(iterations):
        grad = answers.grad_f2(X_prev)
        X_new = X_prev - gamma*grad
        ax.plot(X_prev[0],X_prev[1],color='black',marker='x',markersize=8,alpha=0.5+iteration/50)
        ax.plot([X_prev[0],X_new[0]],[X_prev[1],X_new[1]],'b-')
        X_prev = X_new

    plt.show()

def gradient_descend_f3():
    delta = 0.1
    iterations = 50
    gamma = 0.1

    x = np.arange(-1.5, 1.5, delta)
    y = np.arange(-1.5, 1.5, delta)
    X, Y = np.meshgrid(x, y)
    Z=f3(X,Y)
    fig, ax = plt.subplots()
    CS = ax.contour(X, Y, Z, 30)
    ax.clabel(CS, inline=1, fontsize=10)
    ax.set_title('Gradient Descent for f3 with iteration={}, gamma={}'.format(iterations,gamma))
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')


    X_prev = np.array([1, -1])
    ax.plot(X_prev[0],X_prev[1],color='black',marker='x',markersize=8)


    for iteration in range(iterations):
        grad = grad_f3(X_prev)
        X_new = X_prev - gamma*grad
        ax.plot(X_prev[0],X_prev[1],color='black',marker='x',markersize=8,alpha=0.5+iteration/50)
        ax.plot([X_prev[0],X_new[0]],[X_prev[1],X_new[1]],'b-')
        X_prev = X_new

    plt.show()


def grad_f3(x):
    """
    This question is optional. The test will still run (so you can see if you
    are correct by looking at the testResults.txt file), but the marks are for
    grad_f1 and grad_f2 only.

    Do not delete this function.

    :param x: input array with shape (2, )
    :return: the gradient of f3, with shape (2, )
    """
    from torch.autograd import Variable
    import torch
    temp=torch.DoubleTensor(x)
    input = Variable(temp.reshape(2,1), requires_grad=True)
    a = torch.DoubleTensor([[1.] ,[0.]])
    b = torch.DoubleTensor([[0.] ,[-1.]])
    B = torch.DoubleTensor([[3., -1.],
                      [-1., 3.]])
    eye = torch.eye(2).double()

    x_a = input - a
    x_b = input - b

    first = 1 - torch.exp(-(x_a.t().mm(x_a)))
    second = torch.exp(-(x_b.t().mm(B).mm(x_b)))
    third = -1/10 * torch.log(torch.det(eye/100 + input*input.t()))

    f3 = first - second - third
    f3.backward()
    return input.grad.reshape([2])



if __name__ == '__main__':
    print(f3(2,1))
    # f3_2(np.array([[2],[1]]))
    gradient_descend_f3()
