import numpy as np
from ann_criterion import optimality_criterion


def PSO(func, dim, iteration):
    X = np.array([0.5819, 0.3683, 0.4413])
    Y = 0.3675
    w = 0.9
    cp = 2.5
    cg = 0.5
    pb_temp = [[] for _ in range(dim)]
    pb_obj_temp = []
    n_particles = 600
    X = np.random.rand(dim, n_particles) * 5
    V = np.random.randn(dim, n_particles) * 0.1
    pb = X
    for i in range(n_particles):
        pb_obj_temp.append(func(np.array(X[:, i])))

    pb_obj = np.array(pb_obj_temp)
    gb = pb[:, pb_obj.argmin()]
    gb_obj = pb_obj.min()
    for i in range(iteration):
        rp, rg = np.random.rand(2)
        V = w * V + cp * rp * (pb - X) + cg * rg * (gb.reshape(-1, 1) - X)
        X = X + V
        obj_temp = []
        for i in range(n_particles):
            obj_temp.append(func(np.array(X[:, i])))
        obj = np.array(obj_temp)
        pb[:, (pb_obj >= obj)] = X[:, (pb_obj >= obj)]
        pb_obj = np.array([pb_obj, obj]).min(axis=0)
        gb = pb[:, pb_obj.argmin()]
        gb_obj = pb_obj.min()
    return gb, gb_obj


if __name__ == '__main__':
    gb, gb_obj = PSO(lambda x: optimality_criterion(x), 60, 50)
    print(gb, gb_obj)
