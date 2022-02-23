
import numpy as np
import jax
from jax import jit, vmap
import jax.numpy as jnp
import scipy
from sklearn.metrics.pairwise import pairwise_distances
from tqdm import tqdm
from functools import partial

def udfs(X, gamma=1e6, lamb=1e-6, eps=1e-4, k=5, n_cluster=5, verbose=True, maxiter=1000,mdir=None):
    """
    min_W Tr(W^T M W) + gamma ||W||_{2,1}, s.t. W^T W = I
    Input
    -----
    X: numpy array, shape (n_samples, n_features)
        input data
    gamma: float
        parameter in the objective function of UDFS (default is 1)
    n_cluster: int
        Number of clusters
    k: int
        number of nearest neighbor
    verbose: {boolean}
        True if want to display the objective function value, false if not
    Output
    ------
    W: numpy array, shape(n_features, n_cluster)
        feature weight matrix
    Reference
    Yang, Yi et al. "l2,1-Norm Regularized Discriminative Feature Selection for Unsupervised Learning." AAAI 2012.
    Li, Zechao et al., "Unsupervised Feature Selection Using Nonnegative Spectral Analysis." AAAI 2012.
    """

    @partial(jit, static_argnums=(3,))
    def calculate_obj(X, W, M, gamma):
        return jnp.trace(jnp.dot(jnp.dot(W.T, M), W)) + gamma*(jnp.sqrt(jnp.multiply(W, W).sum(1))).sum()

    @jit
    def generate_D(U):
        """Generate a diagonal matrix D from U: D_ii = 0.5 / ||U[i,:]|| """
        Dii = jnp.clip(jnp.sqrt(jnp.multiply(U, U).sum(1)), a_min=lamb)
        D = jnp.diag(jnp.reciprocal(2*Dii))
        return D

    X = preprocess(X, stand=True)
    X = jnp.array(X)

    # construct M
    n_sample, n_feature = X.shape
    if mdir is not None:
        M = np.loadtxt(mdir)
    else:
        M = construct_M(X, k, gamma, lamb)

    D = jnp.eye(n_feature)
    max_iter = 1000
    obj = []
    solution_path = []
    for iter_step in tqdm(range(maxiter), desc='updating W'):
        # update W as the eigenvectors of P corresponding to the first n clusters
        # smallest eigenvalues
        P = M + gamma*D
        eigenvalues, eigenvectors = scipy.linalg.eigh(a=P)
        eigenvalues = eigenvalues[0:n_cluster]
        W = eigenvectors[:, 0:n_cluster]

        # update D as D_ii = 1 / (2 * ||W(i,:)||)
        D = generate_D(W)
        obj.append(calculate_obj(X, W, M, gamma).item())
        solution_path.append(W)

        if verbose:
            tqdm.write('iter: {0} obj: {1} max ev: {2} min ev: {3}'.format(iter_step+1, 
                obj[iter_step], eigenvalues.max(), eigenvalues.min()))

        if iter_step >= 1 and jnp.abs(obj[iter_step] - obj[iter_step-1]) < 1e-4:
            break
    W = solution_path[jnp.argmin(jnp.array(obj))]
    return W, M

def construct_M(X, k, gamma, lamb):
    """Construct M """
    n_sample, n_feature = X.shape
    Xt = X.T
    D = pairwise_distances(X, n_jobs=-2)
    # sort the distance matrix D in ascending order
    idx = jnp.argsort(D, axis=1)
    # choose the k-nearest neighbors for each instance
    idx_new = idx[:, 0:k+1]
    I = jnp.eye(k+1)
    H = I - 1/(k+1) * jnp.ones((k+1, k+1))
    Mi = jnp.zeros((n_sample, n_sample))

    sample_iter = jnp.arange(n_sample)
    q = jnp.arange(k+1)
    nbatch = 1000


    @jit
    def _mi(i):
        Xi = Xt[:, idx_new[i, :]]
        Xi_tilde =jnp.dot(Xi, H)
        Bi = jnp.linalg.inv(jnp.dot(Xi_tilde.T, Xi_tilde) + lamb*I)

        Si = jnp.zeros((n_sample, k+1))
        Si = Si.at[idx_new[i], q].set(1)

        return jnp.dot(jnp.dot(Si, jnp.dot(jnp.dot(H, Bi), H)), Si.T)

    for i, ite in enumerate(tqdm(jnp.array_split(sample_iter, nbatch),desc='construct m')): 
        Mis = vmap(_mi, in_axes=(0))(ite)
        Mi = Mi + jnp.sum(Mis, axis=0)

    M = jnp.dot(jnp.dot(X.T, Mi), X)
    return M

def preprocess(data, norm=False, stand=False):
    assert norm or stand
    if norm:
        return (data - data.min(axis=0)[0]) / (data.max(axis=0)[0] - data.min(axis=0)[0])
    elif stand:
        return (data - data.mean(axis=0)) / data.std(axis=0)
