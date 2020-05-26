import numpy as np
from math import sqrt, pi, exp, pow
import matplotlib.pyplot as plt
import scipy.stats as stats

capacities = [20, 60, 1000]

distributions = [
    {
        'name': 'normal',
        'sample': lambda size: np.random.normal(size=size),
        'stat': stats.norm(loc=0, scale=1),
        'pf': lambda x: stats.norm(loc=0, scale=1).pdf(x),
        'limits': [-4, 4]
    },
    {
        'name': 'cauchy',
        'sample': lambda size: np.random.standard_cauchy(size=size),
        'stat': stats.cauchy(loc=0, scale=1),
        'pf':  lambda x: stats.cauchy(loc=0, scale=1).pdf(x),
        'limits': [-4, 4]
    },
    {
        'name': 'laplace',
        'sample': lambda size: np.random.laplace(0, 1 / sqrt(2), size=size),
        'stat': stats.laplace(loc=0, scale=1 / sqrt(2)),
        'pf':  lambda x: stats.laplace(loc=0, scale=1 / sqrt(2)).pdf(x),
        'limits': [-4, 4]
    },
    {
        'name': 'poisson',
        'sample': lambda size: np.random.poisson(10, size=size),
        'stat': stats.poisson(10),
        'pf': lambda x: stats.poisson(10).pmf(np.ceil(x)),
        'limits': [6, 14]
    },
    {
        'name': 'uniform',
        'sample': lambda size: np.random.uniform(-sqrt(3), sqrt(3), size=size),
        'stat': stats.uniform(-sqrt(3), 2 * sqrt(3)),
        'pf': lambda x: stats.uniform(-sqrt(3), 2 * sqrt(3)).pdf(x),
        'limits': [-4, 4]
    },
]

def empirical_distributions(x, sample):
    return len([_ for _ in sample if _ < x]) / len(sample)


def kernel(u):
    return 1 / sqrt(2 * pi) * exp(-u * u / 2)


def empirical_density(x, sample):
    n = len(sample)
    sum = 0
    hn = 1.06 * sqrt(np.var(sample)) * pow(n, -1 / 5)
    hHalf = hn / 2
    doubleH = 2 * hn
    for xi in sample:
        sum += kernel((x-xi) / hn)
    return sum / n / hn


for dist in distributions:
    i = 1
    for cap in capacities:
        plt.subplot(1, 3, i)
        plt.title(dist['name'])
        plt.xlabel("capacity - %s" % cap)
        plt.grid()
        plt.xlim(dist['limits'][0], dist['limits'][1])

        sample = sorted(dist['sample'](size=cap))
        x = [_ for _ in sample if dist['limits'][0] <= _ <= dist['limits'][1]]
        y = [empirical_distributions(_, x) for _ in x]
        y_cdf = dist['stat'].cdf(x)

        plt.plot(x, y, color="tomato")
        plt.plot(x, y_cdf, color="darkgreen")
        i += 1
    plt.show()


for dist in distributions:
    i = 1
    for cap in capacities:
        plt.subplot(1, 3, i)
        plt.title(dist['name'])
        plt.xlabel("capacity - %s" % cap)
        plt.grid()
        plt.xlim(dist['limits'][0], dist['limits'][1])

        sample = sorted(dist['sample'](size=cap))
        x = [_ for _ in sample if dist['limits'][0] <= _ <= dist['limits'][1]]
        y = [empirical_density(_, x) for _ in x]
        y_pf = dist['pf'](x)

        plt.plot(x, y, color="tomato")
        plt.plot(x, y_pf, color="darkgreen")
        i += 1
    plt.show()