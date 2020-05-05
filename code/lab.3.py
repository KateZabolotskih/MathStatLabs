import numpy as np
from math import sqrt, pi, exp, pow, factorial, fabs, ceil
import matplotlib.pyplot as plt

capacities = [20, 100]

class Sample:
    @staticmethod
    def normal(size):
        return np.random.normal(size=size)

    @staticmethod
    def cauchy(size):
        return np.random.standard_cauchy(size=size)

    @staticmethod
    def laplace(size):
        return np.random.laplace(0, 1 / sqrt(2), size=size)

    @staticmethod
    def poisson(size):
        return np.random.poisson(10, size=size)

    @staticmethod
    def uniform(size):
        return np.random.uniform(-sqrt(3), sqrt(3), size=size)

def ShowBoxPlot(sample1, sample2, name):
    _, ax = plt.subplots()
    ax.boxplot([sample1, sample2], vert=False)
    ax.set_title(name)
    ax.set_yticklabels(['20', '100'])
    plt.show()

ShowBoxPlot(Sample.normal(capacities[0]), Sample.normal(capacities[1]), "normal")
ShowBoxPlot(Sample.cauchy(capacities[0]), Sample.cauchy(capacities[1]), " cauchy")
ShowBoxPlot(Sample.laplace(capacities[0]), Sample.laplace(capacities[1]), "laplace")
ShowBoxPlot(Sample.poisson(capacities[0]), Sample.poisson(capacities[1]), "poisson")
ShowBoxPlot(Sample.uniform(capacities[0]), Sample.uniform(capacities[1]), "uniform")

for cap in capacities:
    samples = [Sample.normal(cap),
               Sample.cauchy(cap),
               Sample.laplace(cap),
               Sample.poisson(cap),
               Sample.uniform(cap)]
    distributions = ["normal", "cauchy", "laplace", "poisson", "uniform"]
    sum = 0
    i = 0
    for sam in samples:
        for _ in range(1000):
            array = sorted(sam)
            l = len(array)
            Q1 = array[int(1 / 4 * l)]
            Q3 = array[int(3 / 4 * l)]
            X1 = Q1 - 3 / 2 * (Q3 - Q1)
            X2 = Q3 + 3 / 2 * (Q3 - Q1)
            discharge = list(filter(lambda x: x < X1 or x > X2, array))
            discharges = len(list(filter(lambda x: x < X1 or x > X2, array)))
            sum += discharges
        print("%s-%s average discharges proportion %s" % (distributions[i], cap, sum / 1000 / cap))
        i += 1


# experiment PARAMETERS
sizes = [20, 100]
n_tests = len(sizes)
n_calc = 1000
whis = 1.5
n_digits = 6
eps = 0.0000001

for distr in sf.distrs:
    for j in range(0, n_tests):
        outliers = np.zeros(n_calc)
        for i in range(0, n_calc):
            values = distr['stat'].rvs(size=sizes[j])
            # calculating borders as LQ - whis * IQR, UQ + whis * IQR
            lq = np.quantile(values, 0.25)
            uq = np.quantile(values, 0.75)
            iqr = uq - lq
            l = lq - whis * iqr
            r = uq + whis * iqr
            # test data for outliers
            for m in range(0, sizes[j]):
                if values[m] < l or values[m] > r:
                    outliers[i] += 1
        outliers /= sizes[j]
        prob = sum(outliers) / n_calc  # outliers frequency
        disp = (1 / n_calc) * (sum(outliers * outliers)) - prob * prob
        print(distr['name'] + ", n=" + str(sizes[j]) + " :")
        print("P = " + str(round(prob, n_digits)))
        print("D = " + str(round(disp, n_digits)))

print("")

# theoretical outlier probabilities
print("")
print("Theoretical:")
for distr in sf.distrs:
    stat = distr['stat']
    X1 = stat.ppf(1/4) - (3/2) * (stat.ppf(3/4) - stat.ppf(1/4))
    X2 = stat.ppf(3/4) + (3/2) * (stat.ppf(3/4) - stat.ppf(1/4))
    P = stat.cdf(X1) - (stat.cdf(X1+eps) - stat.cdf(X1-eps)) + (1 - stat.cdf(X2))
    print(distr['name'] + ": " + str(P))
