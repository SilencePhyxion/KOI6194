import pymc3 as pm
import pymc3_ext as pmx
import aesara_theano_fallback.tensor as tt
from celerite2.theano import terms, GaussianProcess
import multiprocessing, pickle

def run_sampling(model, map_soln, 
                 tune=1500, 
                 draws=1000, 
                 cores=1, 
                 chains=2, 
                 target_accept=0.95, 
                 return_inferencedata=True, 
                 random_seed=[203771098, 203775000],
                 init="adapt_full"):
    """
    Runs MCMC sampling in the given model using PyMC3.

    Parameters:
        model: pm.Model
            The PyMC3 model to sample from.
        map_soln: dict
            Initial parameter values (e.g., from pm.find_MAP()).
        tune: int
            Number of tuning steps.
        draws: int
            Number of samples.
        cores: int
            Number of CPU cores (parallel chains; >1 only in scripts, not notebooks).
        chains: int
            Number of chains to sample.
        target_accept: float
            Target acceptance probability for step sampler.
        return_inferencedata: bool
            Whether to return arviz InferenceData.
        random_seed: int, list, or None
            Random seed(s) for reproducibility.
        init: str
            Initialization method for NUTS sampler.

    Returns:
        trace: pm.backends.base.MultiTrace or arviz.InferenceData
    """
    with model:
        trace = pm.sample(
            tune=tune,
            draws=draws,
            initvals=map_soln,
            cores=cores,
            chains=chains,
            target_accept=target_accept,
            return_inferencedata=return_inferencedata,
            random_seed=random_seed,
            mp_ctx=multiprocessing.get_context("fork"),
            init=init,
        )
    return trace
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("map_soln.pkl", "rb") as f:
    map_soln = pickle.load(f)

trace = run_sampling(model, map_soln, cores=2, chains=2)

with open("trace.pkl", "wb") as f:
    pickle.dump(trace, f)

print("Sampling finished and trace saved!")