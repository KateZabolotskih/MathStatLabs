from scipy import stats
import math


def inter_classic(samples, alpha):
    m, s = stats.norm.fit(samples)
    n = len(samples)

    t = stats.t(n - 1)
    chi2 = stats.chi2(n - 1)

    m_l = m - s * t.ppf(1 - alpha / 2) / math.sqrt(n - 1)
    m_r = m + s * t.ppf(1 - alpha / 2) / math.sqrt(n - 1)
    s_l = s * math.sqrt(n / chi2.ppf(1 - alpha / 2))
    s_r = s * math.sqrt(n / chi2.ppf(alpha / 2))

    return [m_l, m_r], [s_l, s_r]


def inter_asymptotic(samples, alpha):
    m, s = stats.norm.fit(samples)
    n = len(samples)
    e = stats.kurtosis(samples)

    u = stats.norm(0, 1)
    U = u.ppf(1 - alpha / 2) * math.sqrt((e + 2) / n)

    m_l = m - s * u.ppf(1 - alpha / 2) / math.sqrt(n)
    m_r = m + s * u.ppf(1 - alpha / 2) / math.sqrt(n)
    s_l = s * math.pow(1 + U, -1 / 2)
    s_r = s * math.pow(1 - U, -1 / 2)

    return [m_l, m_r], [s_l, s_r]

inter_estimations = [
    {
        'name': "classic",
        'fun': inter_classic
    },
    {
        'name': "asymptotic",
        'fun': inter_asymptotic
    }
]
sizes = [20, 100]
alpha = 0.05
TAB = "    "
TAB2 = TAB + TAB

if __name__ == "__main__":
    for size in sizes:
        samples = stats.norm(0, 1).rvs(size=size)
        print("Size " + str(size) + ":")
        for est in inter_estimations:
            m, s = est['fun'](samples, alpha)
            print(TAB + est['name'])
            print(TAB2 + "m in", m)
            print(TAB2 + "s in", s)
        print("")