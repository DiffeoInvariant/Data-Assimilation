import numpy as np
from scipy.stats import anderson
import matplotlib.pyplot as plt


def advance_and_observe(forecast, x0, obs_var):
    obs_error = np.random.normal(0.0, np.sqrt(obs_var))
    return forecast(x0) + obs_error


def forward_model(step_model, x0, n):
    obs = [x0]
    for i in range(n):
        obs.append(step_model(obs[-1]))

    return np.array(obs)

def draw_ensemble(mean, var, n):
    return np.random.normal(mean, np.sqrt(var), n)

def create_ensemble(obs, ensemble_mean, ensemble_var, n):
    return obs + draw_ensemble(ensemble_mean, ensemble_var, n)

def EnKF_Step(forecast, ensemble, obs, obs_var, n, inflation=1.0, H=None, adthreshold=0.05):
    
    for i in range(n):
        #forecast ensemble
        ensemble[i] = forecast(ensemble[i])
    #check if prior is Gaussian with Anderson-Darling
    stat, crit_vals, sig_lvls = anderson(ensemble, 'norm')
    #5% is the third critical value
    if stat >= crit_vals[2]:
        print("Ensemble fails Anderson-Darling test (is rejected as Normal) with test statistic %f and critical value %f" % (stat, crit_vals[2]))
        failed=True
    else:
        failed=False
    if not inflation == 1.0:
        ensemble *= np.sqrt(inflation)
    #covariance
    amean = np.mean(ensemble) #analysis mean

    C = np.sum((ensemble - amean) ** 2) / (n-1)

    
    #update ensemble
    if H is None:

        K = C / (C + obs_var)

        ensemble += K * (obs + draw_ensemble(0.0, obs_var, n) - ensemble)

        C -= K * C

        amean = np.mean(ensemble) #analysis mean
        aspread = np.sqrt(C)


    return amean, aspread, ensemble, failed


def EnKF_Run(forecast, init_ensemble, obs, obs_var, timesteps, n,inflation=1.0, H=None, adthreshold=0.05):
    ameans = []
    apreads = []
    nfailed = 0
    for t in range(timesteps):
        am, asp, init_ensemble, failed = EnKF_Step(forecast, init_ensemble, obs[t], obs_var, n, inflation=1.0, H=None, adthreshold=0.05)
        if failed:
            nfailed += 1
        ameans.append(am)
        apreads.append(asp)

    return ameans, apreads, nfailed

if __name__ == '__main__':

    def advance(x):
        xp = 3.95 * x * (1-x)
        if xp < 0.0:
            xp = 0.0
        elif xp > 1.0:
            xp = 1.0

        return xp


    #generate obs
    n = 200
    x0 = 0.25
    x = forward_model(advance, x0, n-1)
    
    obs_var = 0.01
    ensemble_size = 50
    xerr = draw_ensemble(0.0, obs_var, n)

    obs = x + xerr
    
    iensemble = draw_ensemble(1.0/3.0, 0.01, ensemble_size)
    
    emeans, evars, nfailed = EnKF_Run(advance, iensemble, obs, obs_var, n, ensemble_size)

    print("non-gaussian on fraction of timesteps" , (float(nfailed)/200.0), '\n')

    err = np.abs(obs - emeans)

    bigerr=0

    for i in range(n):
        if err[i] > np.sqrt(obs_var):
            bigerr += 1

    print("Error greater than observation error on fraction of timesteps", float(bigerr)/200.0, '\n')


    plt.plot(np.array(list(range(n))), err)
    plt.title('EnKF Error vs Timestep')


    #random matrix 
    A = np.random.rand(100,100)

    Q, R = np.linalg.qr(A)

    D = np.identity(100)
    D[0,0] = np.sqrt(100)

    Broot = Q @ D @ Q.T

    x = np.empty((100,110))

    for i in range(110):
        x[:,i] = Broot @ np.random.standard_normal(100)

    C = np.cov(x)

    lbda, v = np.linalg.eig(C)

    lbda = lbda[np.argsort(lbda)[-100:]] #10 largest


    plt.figure()
    plt.hist(lbda, bins=50)

    
    plt.show()
        
        


    

