import numpy as np
import matplotlib.pyplot as plt

from functools import partial
from scipy.stats import norm

import sys
sys.path.append("../..")
from ip_mcmc import (EnsembleManager,
                     MCMCSampler,
                     pCNAccepter,
                     StandardRWAccepter,
                     ConstrainAccepter,
                     CountedAccepter,
                     ConstSteppCNProposer,
                     EvolutionPotential,
                     GaussianDistribution,
                     ConstStepStandardRWProposer,
                     VarStepStandardRWProposer)


sys.path.append("../Sod_shock_tube")
from utilities import (FVMObservationOperator,
                       PerturbedSodIC,
                       HLLCMCMC,
                       MeasurerMultiD,
                       autocorrelation,
                       DATA_DIR)

from hllc import EulersEquation


def export_to_tikz_array(name, x, y):
    """Export in tikz usable format
    Overwrite existing files"""
    x, y = (np.asarray(x), np.asarray(y))
    assert len(x.shape) == 1, ""
    assert len(y.shape) == 1, ""
    assert x.size == y.size, ""

    with open(DATA_DIR + name + ".tikz.txt", "w") as f:
        for x_, y_ in zip(x, y):
            print(x_, y_, file=f)


def hist_export_to_tikz_array(name, vals):
    bins, binsize = np.linspace(-0.6, 0.6, num=25, retstep=True)
    hist, _ = np.histogram(vals, bins=bins, density=True)

    # give center of bins
    x = np.array([left_edge + binsize/2 for left_edge in bins[:-1]])
    export_to_tikz_array(name, x, hist)


def mean_MAP_plot(name, samples):
    # extract mean
    mean = np.mean(samples, axis=1)
    print(f"mean={mean}")

    # extract MAP
    MAP = np.zeros_like(mean)
    for i in range(MAP.shape[0]):
        histvals, bins = np.histogram(samples[i, :], bins=1000)
        MAP[i] = (bins[np.argmax(histvals)] + bins[np.argmax(histvals) + 1]) / 2
    print(f"MAP={MAP}")

    # run FVM with those point as input
    integrator = create_integrator()
    mean_endstate = integrator(PerturbedRiemannIC(mean))
    MAP_endstate = integrator(PerturbedRiemannIC(MAP))

    # export to tikz
    x_vals = Settings.Simulation.get_xvals()
    export_to_tikz_array(name + "_mean", x_vals, mean_endstate)
    export_to_tikz_array(name + "_MAP", x_vals, MAP_endstate)


class PWLinear:
    """linearly decrease delta until burn_in is finished, then keep it constant"""
    def __init__(self, start_delta, end_delta, len_burn_in):
        self.d_s = start_delta
        self.d_e = end_delta
        self.l = len_burn_in

        self.slope = (start_delta - end_delta) / len_burn_in

    def __call__(self, i):
        if i > self.l:
            return self.d_e
        return self.d_s - self.slope * i

    def __repr__(self):
        """For filename"""
        return f"pwl_{self.d_s}_{self.d_e}_{self.l}"


class Settings:
    """'Static' class to collect the settings for a MCMC simulation"""
    # 'Attributes' that derive from other attributes need to be impleneted
    # using a getter-method, so that they get updated when the thing
    # they depend on changes.
    class Simulation:
        class IC:
            names = ["rho_l", "v_l", "p_l", "rho_r", "v_r", "p_r"]
            rho_l = 0
            v_l = 0
            p_l = 0
            rho_r = 0
            v_r = 0
            p_r = 0
            ground_truth = np.array([rho_l, v_l, p_l,
                                     rho_r, v_r, p_r])

        domain = (0, 1)
        N_gridpoints = 128
        T_end = 0.2
        model = EulersEquation

        @staticmethod
        def get_xvals():
            return HLLCMCMC(EulersEquation,
                            Settings.Simulation.domain,
                            Settings.Simulation.N_gridpoints,
                            0).FVM.x[1:-1]

    class Measurement:
        # These are actually 15 measurements (5 locations, 3 state variables)
        points = [0.1, 0.25, 0.5, 0.75, 0.9]
        weights = [1, 1, 1, 1, 1]
        interval = 0.1

    class Noise:
        mean = np.array([0] * 15)
        std_dev = 0.05
        covariance = std_dev**2 * np.identity(15)

        @staticmethod
        def get_distribution():
            return GaussianDistribution(Settings.Noise.mean,
                                        Settings.Noise.covariance)

    class Prior:
        mean = np.array([-0.1,    # rho_l
                         0.1,    # v_l
                         -0.1,    # p_l
                         0.1,    # rho_r
                         0.1,   # v_r
                         0.1])   # p_r
        std_dev = 0.15
        covariance = std_dev**2 * np.identity(len(mean))

        @staticmethod
        def get_distribution():
            return GaussianDistribution(Settings.Prior.mean,
                                        Settings.Prior.covariance)

    class Sampling:
        step = PWLinear(0.0005, 0.0005, 250)
        u_0 = np.zeros(6)
        N = 1500
        burn_in = 500
        sample_interval = 10
        rng = np.random.default_rng(2)

    @staticmethod
    def filename():
        return f"sod_testing_n={Settings.Sampling.N}_h={Settings.Simulation.N_gridpoints}"


def create_integrator():
    return HLLCMCMC(Settings.Simulation.model,
                    Settings.Simulation.domain,
                    Settings.Simulation.N_gridpoints,
                    Settings.Simulation.T_end)


def create_measurer():
    return MeasurerMultiD(Settings.Measurement.points,
                          Settings.Measurement.interval,
                          Settings.Simulation.get_xvals(),
                          Settings.Measurement.weights)


def create_mcmc_sampler():
    # Proposer
    prior = Settings.Prior.get_distribution()
    # proposer = ConstSteppCNProposer(Settings.Sampling.step, prior)
    # proposer = ConstStepStandardRWProposer(Settings.Sampling.step, prior)
    proposer = VarStepStandardRWProposer(Settings.Sampling.step, prior)

    # Accepter
    integrator = create_integrator()
    measurer = create_measurer()
    IC_true = PerturbedSodIC(Settings.Simulation.IC.ground_truth)
    observation_operator = FVMObservationOperator(PerturbedSodIC,
                                                  Settings.Prior.mean,
                                                  integrator,
                                                  measurer)

    # compute the ground truth on a very fine grid
    old_N_gridpoints = Settings.Simulation.N_gridpoints
    Settings.Simulation.N_gridpoints = 1024
    # Settings.Simulation.N_gridpoints = 128
    ground_truth = create_measurer()(create_integrator()(IC_true))
    Settings.Simulation.N_gridpoints = old_N_gridpoints

    noise = Settings.Noise.get_distribution()
    potential = EvolutionPotential(observation_operator,
                                   ground_truth,
                                   noise)
    # accepter = CountedAccepter(pCNAccepter(potential))
    accepter = ConstrainAccepter(CountedAccepter(StandardRWAccepter(potential, prior)),
                                 PerturbedSodIC.is_valid)

    return MCMCSampler(proposer, accepter)


def run_MCMC():
    rng = [Settings.Sampling.rng]

    sampler = create_mcmc_sampler()
    chain_start = partial(sampler.run,
                          Settings.Sampling.u_0,
                          Settings.Sampling.N,
                          Settings.Sampling.burn_in,
                          Settings.Sampling.sample_interval)

    ensemble_manager = EnsembleManager(DATA_DIR,
                                       f"{Settings.filename()}")

    ensemble = ensemble_manager.compute(chain_start, rng, 1)

    for i in range(Settings.Sampling.N):
        ensemble[0, :, i] += Settings.Prior.mean

    # plt.plot(autocorrelation(ensemble[0, :, :], 50).transpose())
    # plt.show()

    show_ensemble(ensemble)


def show_ensemble(ensemble):
    """Plz write docstring"""

    title = f"Chain length: {ensemble.shape[2]}, ensemble members: {ensemble.shape[0]}"
    for k in range(ensemble.shape[0] - 1):
        for i in range(6):
            plt.plot(ensemble[k, i, :])

    for i in range(6):
        plt.plot(ensemble[-1, i, :], label=Settings.Simulation.IC.names[i])

    plt.legend()
    plt.title(title)
    plt.show()

    # densities
    intervals = [(-2, 2)] * 6
    priors = [norm(loc=mu, scale=Settings.Prior.std_dev)
              for mu in Settings.Prior.mean]
    fig, plts = plt.subplots(2, 3, figsize=(20, 10))

    plot_info = list(zip(intervals,
                         [0] * 6,
                         Settings.Simulation.IC.names,
                         priors,
                         plts.flatten()))

    for k in range(ensemble.shape[0] - 1):
        for i, (interval, _, __, ___, ax) in enumerate(plot_info):
            x_range = np.linspace(*interval, num=500)
            ax.hist(ensemble[k, i, :], density=True, alpha=0.5)

    for i, (interval, true_val, name, prior, ax) in enumerate(plot_info):
        x_range = np.linspace(*interval, num=500)
        ax.plot(x_range, [prior.pdf(x) for x in x_range])
        ax.hist(ensemble[-1, i, :], density=True, color='b', alpha=0.5)
        ax.axvline(true_val, c='r')
        ax.set_title(f"Posterior for {name}")
        ax.set(xlabel=name, ylabel="Probability")

    plt.show()


if __name__ == '__main__':
    run_MCMC()