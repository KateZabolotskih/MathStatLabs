from matplotlib import pyplot as plt
import numpy as np
from scipy import stats
import math

corratios = [
    {
        'name': "pearson",
        'fun': lambda x, y: stats.pearsonr(x, y)[0]
    },
    {
        'name': "spearman",
        'fun': lambda x, y: stats.spearmanr(x, y)[0]
    },
    {
        'name': "quadrant",
        'fun': lambda x, y: np.mean(np.sign(x - np.median(x)) * np.sign(y - np.median(y)))
    }
]


def mix_rvs(size):
    data = np.zeros((size, 2, 2))
    data[:, 0] = stats.multivariate_normal(mean=[0, 0],
                                           cov=[[1, 0.9], [0.9, 1]]).rvs(size=size)
    data[:, 1] = stats.multivariate_normal(mean=[0, 0],
                                           cov=[[100, -90], [-90, 100]]).rvs(size=size)
    index = np.random.choice(np.arange(2), size=size, p=[0.9, 0.1])
    return data[np.arange(size), index]


sizes = [20, 60, 100]
n_times = 1000

params = [
    {
        'name': "rho = 0.0",
        'meanX': 0, 'meanY': 0,
        'sigmaX': 1, 'sigmaY': 1,
        'rho': 0,
        'rvs': stats.multivariate_normal(mean=[0, 0],
                                         cov=[[1, 0], [0, 1]]).rvs
    },
    {
        'name': "rho = 0.5",
        'meanX': 0, 'meanY': 0,
        'sigmaX': 1, 'sigmaY': 1,
        'rho': 0.5,
        'rvs': stats.multivariate_normal(mean=[0, 0],
                                         cov=[[1, 0.5], [0.5, 1]]).rvs
    },
    {
        'name': "rho = 0.9",
        'meanX': 0, 'meanY': 0,
        'sigmaX': 1, 'sigmaY': 1,
        'rho': 0.9,
        'rvs': stats.multivariate_normal(mean=[0, 0],
                                         cov=[[1, 0.9], [0.9, 1]]).rvs
    },
    {
        'name': "mix",
        'rvs': mix_rvs
    }
]


def digits(vrnc):
    return max(0, round(-math.log10(abs(vrnc))))


TAB = "    "
TAB2 = TAB + TAB
TAB3 = TAB2 + TAB

out = ""
for param in params:
    out += "For " + param['name'] + "\n"
    for size in sizes:
        out += (TAB + "size = " + str(size) + ":\n")
        for corr in corratios:
            out += (TAB2 + corr['name'] + ":\n")
            values = np.zeros(n_times)
            for j in range(0, n_times):
                samples = param['rvs'](size=size)
                values[j] = corr['fun'](samples[:, 0], samples[:, 1])
            var = np.var(values)
            var2 = np.var(values * values)
            mean = np.mean(values)
            mean2 = np.mean(values * values)
            out += (TAB3 + "mean = " + str(np.round(mean, digits(var))) + "\n")
            out += (TAB3 + "mean^2 = " + str(np.round(mean2, digits(var2))) + "\n")
            out += (TAB3 + "variance = " + str(np.round(var, 6)) + "\n")
f = open("out.txt", 'w')
print(out, file=f)
f.close()

params = [params[0], params[1], params[2]]

def ellipse_fun(mean_x, mean_y, sigma_x, sigma_y, rho):
    return lambda x, y: (1 / (sigma_x * sigma_x)) * ((x - mean_x) ** 2) - \
                        2 * rho / (sigma_x * sigma_y) * (x - mean_x) * (y - mean_y) + \
                        1 / (sigma_y * sigma_y) * ((y - mean_y) ** 2)


def rad2(elfun, samples):
    return max(elfun(samples[:, 0], samples[:, 1]))


for param in params:
    ellipse = ellipse_fun(param['meanX'], param['meanY'], param['sigmaX'], param['sigmaY'], param['rho'])
    for size in sizes:
        samples = param['rvs'](size=size)
        plt.scatter(samples[:, 0], samples[:, 1], c='tomato')

        rad_sq = rad2(ellipse, samples)
        plt.title("R = " + str(round(rad_sq, 6)))
        x = np.linspace(-5, 5, 100)
        y = np.linspace(-5, 5, 100)
        x, y = np.meshgrid(x, y)
        z = ellipse(x, y)
        plt.contour(x, y, z, [rad_sq], colors=['maroon'])

        plt.savefig(param['name'] + ", " + "n = " + str(size) + ".png")
        plt.close()
