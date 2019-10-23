import numpy as np

def gamma_pseudoinverse(gmat, n):
    g = np.zeros_like(gmat)
    for i in range(n):
        g[i,i] = 1/ np.sqrt(1 + g[i,i])
        
    return g

if __name__ =='__main__':
    x = 2 ** 0.5 * np.array([[1.0,1.0,0.0],[-1.0,0.0, 1.0],[0.0,-1.0,-1.0]])

    H = np.eye(3)
    R = np.eye(3)

    xdelta = x - x.mean(axis=0, keepdims=True)

    B = 0.5 * xdelta.T @ xdelta

    #H is identity
    K = B @ np.linalg.inv(B + R)

    #again, H is identity
    C = (H - K) @ B

    print("Posterior covariance: ", C)

    A = xdelta.T / (2 ** 0.5)

    V = H @ A

    F, sgm, W = np.linalg.svd(A.T @ A)
    sigma = np.diag(sgm)
    sigmaInv = np.linalg.pinv(sigma)

    sigmaPseudoId = sigmaInv @ sigma @ sigma @ sigmaInv

    #V^T V is Hermitian, and R is identity so V^T R V = V^T V
    Gamma, Q = np.linalg.eigh(V.T @ V)

    Gamma[0] = 0.0

    #2b

    GammaInv = np.diag(1.0/np.sqrt(Gamma + 1))

    C = A @ Q @ GammaInv @ sigmaPseudoId @ GammaInv @ Q.T @ A.T

    print("EAKF C: ", C)

    #2c, same as above, just flip the columns in Gamma and Q

    A = xdelta.T / (2 ** 0.5)

    V = H @ A

    F, sgm, W = np.linalg.svd(A.T @ A)
    sigma = np.diag(sgm)
    sigmaInv = np.linalg.pinv(sigma)

    sigmaPseudoId = sigmaInv @ sigma @ sigma @ sigmaInv
    Gamma, Q = np.linalg.eigh(V.T @ V)

    Qp = np.fliplr(Q)

    Gamma[0] = Gamma[2]
    Gamma[2] = 0.0

    GammaInv = np.diag(1.0/np.sqrt(Gamma + 1))

    print(sigmaPseudoId)

    C = A @ Qp @ GammaInv @ sigmaPseudoId @ GammaInv @ Qp.T @ A.T

    print("EAKF C: ", C)
    
