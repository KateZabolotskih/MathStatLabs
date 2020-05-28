from scipy import stats
import numpy as np
import math
def chi2(loc, scale, data, alpha):
    # calculating k using Rice Rule
    N = len(data)
    #k = math.ceil(2 * pow(N, 1 / 3))
    k = math.ceil(1.72 * math.pow(N, 1 / 3))
    distr = stats.norm(loc=loc, scale=scale)

    # calculating h using Scott's normal reference rule
    h = 3.49 * math.sqrt(np.var(data)) / pow(N, 1 / 3)

    # intervals builder
    center = distr.ppf(0.5)
    a = [-np.inf]
    for i in range(0, k - 1):
        a.append(center + h * (i - ((k - 1) / 2)))
    a.append(np.inf)

    # theoretical probabilities and frequencies
    p = np.zeros(k)
    n = np.zeros(k)
    for i in range(0, k):
        p[i] = distr.cdf(a[i + 1]) - distr.cdf(a[i])
        n[i] = len(data[(a[i] < data) & (data <= a[i + 1])])

    chi2_1 = (1 / N) * \
             sum(np.square(n - N * p) / p)

    chi2_2 = stats.chi2(df=k - 1).ppf(1 - alpha)

    if chi2_1 < chi2_2:
        return {
            'is normal?': True,
            'chi2s': np.round([chi2_1, chi2_2], 4),
            'a_i': np.round(a, 2),
            'p_i': np.round(p, 4),
            'n_i': n,
            'n_i - n * p_i': np.round(n - N * p, 4)
        }
    else:
        return {
            'is normal?': False,
            'chi2s': np.round([chi2_1, chi2_2], 4)
        }


if __name__ == "__main__":
    # here you can set alpha value
    alpha = 0.05

    # here you can add tests
    test = [
        {
            'name': "N(0, 1) - 100",
            'data': stats.norm(0, 1).rvs(100)
        },
        {
            'name': "L(0, 1) - 20",
            'data': stats.laplace(0, 1/math.sqrt(2)).rvs(20)
        }
    ]

    TAB = "    "
    for t in test:
        loc, scale = stats.norm.fit(t['data'])
        print("loc =", loc, "\nscale =", scale, "\n")
        res = chi2(loc, scale, t['data'], alpha)
        print(t['name'] + ":")
        for val in res:
            print(TAB + val + ":", res[val])
        print("\n")
