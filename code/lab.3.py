import math
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt

params = [
    {
        'name': "normal",
        'distr': stats.norm(0, 1)
    },
    {
        'name': "cauchy",
        'distr': stats.cauchy(loc=0, scale=1)
    },
    {
        'name': "laplace",
        'distr': stats.laplace(loc=0, scale=1 / math.sqrt(2))
    },
    {
        'name': "uniform",
        'distr': stats.uniform(-math.sqrt(3), 2 * math.sqrt(3))
    },
    {
        'name': "poisson",
        'distr': stats.poisson(10)
    }
]

sizes = [20, 100]
calculations_number = 1000

for param in params:
    values = []
    for size in sizes:
        values.append(param['distr'].rvs(size=size))

    fig, ax = plt.subplots()
    plt.grid(axis='x')
    ax.boxplot(values, vert=False, medianprops=dict(color='r'), labels=["20", "100"])
    plt.savefig(param['name'] + ".png")

for param in params:
    for size in sizes:
        outliers = np.zeros(calculations_number)
        for i in range(0, calculations_number):
            values = param['distr'].rvs(size=size)
            lq = np.quantile(values, 0.25)
            uq = np.quantile(values, 0.75)
            iqr = uq - lq
            l = lq - 1.5 * iqr
            r = uq + 1.5 * iqr

            out1 = values[values < l]
            out2 = values[values > r]
            outliers[i] = len(out1) + len(out2)
        outliers /= size
        prob = sum(outliers) / calculations_number
        disp = (1 / calculations_number) * (sum(outliers * outliers)) - prob * prob
        print(param['name'], "with n =", size)
        print("P =", round(prob, 6))
        print("D =", round(disp, 6))

    distr = param['distr']
    X1 = distr.ppf(0.25) - 1.5 * (distr.ppf(0.75) - distr.ppf(0.25))
    X2 = distr.ppf(0.75) + 1.5 * (distr.ppf(0.75) - distr.ppf(0.25))
    P = distr.cdf(X1) - (distr.cdf(X1 + 0.0000001) - distr.cdf(X1 - 0.0000001)) + (1 - distr.cdf(X2))
    print("theoretical", param['name'], ":", P)
    print("")