import matplotlib.pyplot as plt
import numpy as np

def xmap(x):
    return 3.95 * x * (1 - x)

def psi(x0,t):
    i = 0
    x = x0
    while i < t:
        x = xmap(x)
        i += 1
    return x

def H(x):
    return x + np.random.normal(0, 0.01)


def obs_error(y, x, t):
    xt = psi(x,t)
    return y - H(xt)  

def varobjective4d(x0, y):
    xb = 1./3.
    b0 = 0.1 ** 2

    R = 0.01 ** 2

    J = (1/b0) * (x0 - xb) ** 2 

    for i in range(len(y)):
        e = obs_error(y[i], x0, i)

        J += (1./R) * (e ** 2)

    return J

if __name__ == '__main__':
    
    true_obs = [0.25]
    for i in range(5):
        true_obs.append(xmap(true_obs[-1]))

    obs = [H(x) for x in true_obs]

    Jvals = []

    for i in range(1,len(obs)):
        Jvals.append(varobjective4d(obs[0], obs[:i]))


    plt.plot(Jvals)
    plt.title("4DVAR objective function vs number of data points")
    plt.xlabel("Number of observations")
    plt.ylabel("4DVAR objective")
    plt.show()




