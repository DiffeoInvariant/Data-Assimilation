import jax.numpy as np
from jax.experimental.ode import odeint
from jax import grad
from jax.flatten_util import ravel_pytree


def 4dvar_ode_solve(xprime, tsteps, x0, atol, rtol):
    ys = odeint(xprime, x0, tsteps, atol, rtol)
    
def run_4dvar(xprime, x0, yobs, H, R, Binv, window_lens, ode_int_atol=1e-12, ode_int_rtol=1e-12, **kwargs):




if __name__ == '__main__':

    tmp = np.eye(40)

    H1 = tmp[0:8:2,:]
    H2 = tmp[10:18:2,:]
    H3 = tmp[20:28:2,:]
    H4 = tmp[30:38;2,:]

    H = np.concatenate((H1,H2,H3,H4),axis=1)

    sigma0 = 1/np.sqrt(10)

    R = sigma0 ** 2 * np.eye(H.shape[0])

    y = np.zeros(20)
