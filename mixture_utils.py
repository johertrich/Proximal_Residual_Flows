# Code extracted from https://github.com/PaulLyonel/conditionalSNF
import numpy as np

# draw num_samples samples from the distributions given by the mixture_params
# returns those samples
def draw_mixture_dist(mixture_params, num_samples):
    n = len(mixture_params)
    sigmas=np.stack([np.sqrt(sigma) for w,mu,sigma in mixture_params])
    probs=np.array([w for w, mu, sigma in mixture_params])
    zs = np.random.choice(n, (num_samples,), p=probs/probs.sum())
    mus = np.stack([mu for w, mu, sigma in mixture_params])[zs]
    sigmas_samples = sigmas[zs]
    multinomial_samples = np.random.normal(size=(num_samples, mus.shape[1]))
    if len(sigmas_samples.shape)==1:
        sigmas_samples=sigmas_samples[:,None]
    out_samples = mus + multinomial_samples*sigmas_samples
    return out_samples

# gets mean and covariance of the Gaussian posterior with linear forward model
# mean, sigma are the parameters of the prior distribution
def get_single_gaussian_posterior(mean, sigma, forward_mat, b_sq, y):
    ATA = forward_mat**2/b_sq
    cov_gauss = 1/(ATA+1/sigma)

    mean_gauss = cov_gauss*forward_mat*y/b_sq+cov_gauss*mean/sigma
    return mean_gauss, cov_gauss

# returns the mixture parameters of the posterior given the mixture parameters of the
# prior, the forward model and the likelihood (for a specific y)
def get_mixture_posterior(x_gauss_mixture_params, forward_mat, b_sq, y):
    out_mixtures = []
    nenner = 0
    ws=np.zeros(len(x_gauss_mixture_params))
    mus_new=[]
    sigmas_new=[]
    log_zaehler=np.zeros(len(x_gauss_mixture_params))
    for k,(w, mu, sigma) in enumerate(x_gauss_mixture_params):
        mu_new, sigma_new = get_single_gaussian_posterior(mu, sigma, forward_mat, b_sq, y)
        mus_new.append(mu_new)
        sigmas_new.append(sigma_new)
        ws[k]=w
        log_zaehler[k]=np.log(w)+(0.5*np.sum(mu_new**2/sigma_new)-0.5*np.sum(mu**2)/sigma)
    const=np.max(log_zaehler)
    log_nenner=np.log(np.sum(np.exp(log_zaehler-const)))+const
    for k in range(len(x_gauss_mixture_params)):
        out_mixtures.append((np.exp(log_zaehler[k]-log_nenner),mus_new[k],sigmas_new[k]))
    return out_mixtures

# creates forward map
# scale controls how illposed the problem is

def create_forward_model(scale,dimension):
    s = np.ones(dimension)
    for i in range(dimension):
        s[i] = scale/(i+1)
    return s

# evaluates forward_map
def forward_pass(x, forward_map):
    return x*forward_map
