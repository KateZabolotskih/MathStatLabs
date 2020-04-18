import numpy as np
import matplotlib.pyplot as plot
import matplotlib.patches as mpatches
import seaborn as sbrn
from math import sqrt, pi, exp, pow, factorial, fabs

capacities = [10, 50, 1000]

def uniform_density(x):
    if x < sqrt(3) and x > -sqrt(3):
        return 1 / (2 * sqrt(3))
    else:
        return 0

def normal_density(x):
    return 1 / sqrt(2 * pi) * exp((-x * x) / 2)

def cauchy_density(x):
    return 1 / (pi * (x * x + 1))

def laplas_density(x):
    return 1 / (2 * sqrt(2)) * exp(-abs(x) / sqrt(2))

def poisson_density(x):
    return (pow(10, round(x)) / factorial(round(x))) * exp(- 10)



def show_normal_histogram():
    counter = 1
    for cap in capacities:
        plot.subplot(1, 3, counter)
        values = np.random.normal(size=cap)
        x = sorted(set(values))
        y = [normal_density(_) for _ in x]
        plot.plot(x, y, color='tomato')
        sbrn.distplot(x)
        plot.ylabel("density")
        plot.xlabel("capacity - %s" % cap)
        plot.grid()
        destiny = mpatches.Patch(color='tomato', label='плотность')
        #kde = mpatches.Patch(color='blue', label='ядерная оценка')
        plot.legend(handles=[destiny])
        counter += 1
    plot.show()

def show_cauchy_histogram():
    counter = 1
    for cap in capacities:
        plot.subplot(1, 3, counter)
        values = np.random.standard_cauchy(size=cap)
        sbrn.distplot(values)
        x = sorted(set(values))
        y = [cauchy_density(_) for _ in x]
        plot.plot(x, y, color='tomato')
        plot.ylabel("density")
        plot.xlabel("capacity - %s" % cap)
        plot.grid()
        destiny = mpatches.Patch(color='tomato', label='плотность')
        #kde = mpatches.Patch(color='blue', label='ядерная оценка')
        plot.legend(handles=[destiny])
        counter += 1
    plot.show()

def show_laplas_histogram():
    counter = 1
    for cap in capacities:
        plot.subplot(1, 3, counter)
        values = np.random.laplace(0, 1/sqrt(2), size=cap)
        sbrn.distplot(values)
        x = sorted(set(values))
        y = [laplas_density(_) for _ in x]
        plot.plot(x, y, color='tomato')
        plot.ylabel("density")
        plot.xlabel("capacity - %s" % cap)
        plot.grid()
        destiny = mpatches.Patch(color='tomato', label='плотность')
        #kde = mpatches.Patch(color='blue', label='ядерная оценка')
        plot.legend(handles=[destiny])
        counter += 1
    plot.show()

def show_poisson_histogram():
    counter = 1
    for cap in capacities:
        plot.subplot(1, 3, counter)
        values = np.random.poisson(10, size=cap)
        sbrn.distplot(values)
        x = sorted(set(values))
        y = [poisson_density(_) for _ in x]
        plot.plot(x, y, color='tomato')
        plot.ylabel("density")
        plot.xlabel("capacity - %s" % cap)
        plot.grid()
        destiny = mpatches.Patch(color='tomato', label='плотность')
        #kde = mpatches.Patch(color='blue', label='ядерная оценка')
        plot.legend(handles=[destiny])
        counter += 1
    plot.show()

def show_uniform_histogram():
    counter = 1
    for cap in capacities:
        plot.subplot(1, 3, counter)
        values = np.random.uniform(-sqrt(3), sqrt(3), size=cap)
        sbrn.distplot(values)
        x = sorted(set(values))
        y = [uniform_density(_) for _ in x]
        plot.plot(x, y, color='tomato')
        plot.ylabel("density")
        plot.xlabel("capacity - %s" % cap)
        plot.grid()
        destiny = mpatches.Patch(color='tomato', label='плотность')
        #kde = mpatches.Patch(color='blue', label='ядерная оценка')
        plot.legend(handles=[destiny])
        counter += 1
    plot.show()

show_normal_histogram()
show_cauchy_histogram()
show_laplas_histogram()
show_poisson_histogram()
show_uniform_histogram()