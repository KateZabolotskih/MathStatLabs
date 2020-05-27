import numpy as np
from scipy import stats
from scipy import optimize
from matplotlib import pyplot as plt
# least modulus
def least_mod(x, y):
    def fun(z):
        return np.sum(np.abs(y - (z[0] + z[1] * x)))
    res = optimize.minimize(fun, np.array([0, 0]), method='Nelder-Mead')
    a = res['x'][0]
    b = res['x'][1]
    return a, b

if __name__ == "__main__":
    x = np.linspace(-1.8, 2, 20)
    y0 = 2 + 2 * x
    y1 = y0 + stats.norm(0, 1).rvs(20)
    y2 = np.copy(y1)
    y2[0] += 10
    y2[19] -= 10

    data = [
        {
            'name': "simple",
            'val': y1
        },
        {
            'name': "with outliers",
            'val': y2
        }
    ]

    for d in data:
        print(d['name'])
        y = d['val']
        b0, b1 = stats.linregress(x, d['val'])[0:2]
        print(" least squares:\n", "  b0 =", round(b0, 2),
              "b1 =", round(b1, 2),
              "Q =", np.round(np.sum(np.square(y - (b0 + b1 * x))), 4),
              "M =", np.round(np.sum(np.abs(y - (b0 + b1 * x))), 4))
        plt.plot(x, b0 + b1 * x, 'orange', label="МНК")

        b0, b1 = least_mod(x, d['val'])
        print(" least modulus:\n", "  b0 =", round(b0, 2),
              "b1 =", round(b1, 2),
              "Q =", np.round(np.sum(np.square(y - (b0 + b1 * x))), 4),
              "M =", np.round(np.sum(np.abs(y - (b0 + b1 * x))), 4))
        plt.plot(x, b0 + b1 * x, 'tomato', label="МНМ")
        print(" ")

        plt.plot(x, y0, 'red', label="эталон")
        plt.scatter(x, d['val'], facecolors='none', edgecolors='black', zorder=50, label="выборка")
        plt.legend()
        plt.savefig(d['name'] + ".png")
        plt.close()