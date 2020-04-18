import numpy as np
from math import sqrt, ceil

capacities = [10, 100, 1000]

properties = [
    {
        "name": "M",
        "func": lambda x: np.mean(x)
    },
    {
        "name": "med",
        "func": lambda x: np.median(x)
    },
    {
        "name": "D",
        "func": lambda x: np.var(x)
    },
    {
        "name": "Zr",
        "func": lambda x: (min(x)+max(x)) / 2
    },
    {
        "name": "Zq",
        "func": lambda x: (sorted(x)[ceil(len(x) * 1/4)] + sorted(x)[ceil(len(x) * 3 / 4)]) / 2
    },
    {
        "name": "Ztr",
        "func": lambda x: np.mean([x[i] for i in range(len(x)) if i > (len(x) / 4) and i < len(x) * 3 / 4])
    }
]

samples = [
    {
        "distribution": "Normal",
        "func": lambda size: np.random.normal(size=size)
    },
    {
        "distribution": "Cauchy",
        "func": lambda size: np.random.standard_cauchy(size=size)
    },
    {
        "distribution": "Laplace",
        "func": lambda size: np.random.laplace(0, 1/sqrt(2), size=size)
    },
    {
        "distribution": "Poisson",
        "func": lambda size: np.random.poisson(10, size=size)
    },
    {
        "distribution": "Uniform",
        "func": lambda size: np.random.uniform(-sqrt(3), sqrt(3), size=size)
    }
]



for smpl in samples:
    print("*************************")
    print(smpl["distribution"])
    for cap in capacities:
        print("**********  " + str(cap) + "  ***********")
        for prop in properties:
            print(prop["name"])
            _ = []
            for i in range(1000):
                x = smpl["func"](cap)
                _.append(prop["func"](x))
            print("E: %s" % np.mean(_))
            print("D: %s" % np.var(_))