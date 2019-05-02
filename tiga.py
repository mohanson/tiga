import argparse
import copy
import os
import os.path

import numpy as np
import skimage.draw
import skimage.io
import skimage.transform


class Base:
    def __init__(self, r, c, color, alpha):
        self.r = r
        self.c = c
        self.color = color
        self.alpha = alpha


class Gene:
    def __init__(self):
        self.base = []

    def copy(self):
        return copy.deepcopy(self)


class GA:
    def __init__(self,
                 control_im_path: str,
                 pop_size: int,
                 dna_size: int,
                 max_iter: int,
                 pc: float,
                 pm: float,
                 im_size: tuple,
                 ):
        self.pop_size = pop_size
        self.dna_size = dna_size
        self.max_iter = max_iter
        self.pc = pc
        self.pm = pm

        im = skimage.io.imread(control_im_path)
        if im.shape[2] == 4:
            im = skimage.color.rgba2rgb(im)
            im = (255 * im).astype(np.uint8)
        if im_size:
            self.control_im = skimage.transform.resize(
                im, im_size, mode='reflect', preserve_range=True).astype(np.uint8)
        else:
            self.control_im = im

    def decode(self, per):
        im = np.ones(self.control_im.shape, dtype=np.uint8) * 255
        for e in per.base:
            rr, cc = skimage.draw.polygon(e.r, e.c)
            skimage.draw.set_color(im, (rr, cc), e.color, e.alpha)
        return im

    def perfit(self, per):
        im = self.decode(per)
        assert im.shape == self.control_im.shape
        d = np.linalg.norm(np.where(self.control_im > im, self.control_im - im, im - self.control_im))
        return (self.control_im.size * 195075) ** 0.5 - d

    def getfit(self, pop):
        fit = np.zeros(self.pop_size)
        for i, per in enumerate(pop):
            fit[i] = self.perfit(per)
        return fit

    def genpop(self):
        pop = []
        for _ in range(self.pop_size):
            per = Gene()
            for _ in range(self.dna_size):
                r = np.random.randint(0, self.control_im.shape[0], 3, dtype=np.uint16)
                c = np.random.randint(0, self.control_im.shape[1], 3, dtype=np.uint16)
                color = np.random.randint(0, 256, 3)
                alpha = np.random.random() * 0.45
                per.base.append(Base(r, c, color, alpha))
            pop.append(per)
        return pop

    def select(self, pop, fit):
        fit = fit - np.min(fit)
        fit = fit + np.max(fit) / 2 + 0.01
        idx = np.random.choice(np.arange(self.pop_size), size=self.pop_size, replace=True, p=fit / fit.sum())
        son = []
        for i in idx:
            son.append(pop[i].copy())
        return son

    def optret(self, f):
        def mt(*args, **kwargs):
            opt = None
            opf = None
            for pop, fit in f(*args, **kwargs):
                max_idx = np.argmax(fit)
                min_idx = np.argmax(fit)
                if opf is None or fit[max_idx] >= opf:
                    opt = pop[max_idx]
                    opf = fit[max_idx]
                else:
                    pop[min_idx] = opt
                    fit[min_idx] = opf
                yield pop, fit
        return mt

    def crosso(self, pop):
        for i in range(0, self.pop_size, 2):
            if np.random.random() < self.pc:
                a = pop[i]
                b = pop[i + 1]
                p = np.random.randint(1, self.dna_size)
                a.base[p:], b.base[p:] = b.base[p:], a.base[p:]
                pop[i] = a
                pop[i + 1] = b
        return pop

    def mutate(self, pop):
        for per in pop:
            for base in per.base:
                if np.random.random() < self.pm:
                    base.r = np.random.randint(0, self.control_im.shape[0], 3, dtype=np.uint16)
                    base.c = np.random.randint(0, self.control_im.shape[1], 3, dtype=np.uint16)
                    base.color = np.random.randint(0, 256, 3)
                    base.alpha = np.random.random() * 0.45
        return pop

    def evolve(self):
        pop = self.genpop()
        pop_fit = self.getfit(pop)
        for _ in range(self.max_iter):
            chd = self.select(pop, pop_fit)
            chd = self.crosso(chd)
            chd = self.mutate(chd)
            chd_fit = self.getfit(chd)
            yield chd, chd_fit
            pop = chd
            pop_fit = chd_fit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('control_im_path')
    parser.add_argument('save_dir')
    parser.add_argument('--pop_size', type=int, default=80, help='population size')
    parser.add_argument('--dna_size', type=int, default=100, help='dna size')
    parser.add_argument('--max_iter', type=int, default=3000, help='population iterations')
    parser.add_argument('--pc', type=float, default=0.6, help='genetic crossover probability')
    parser.add_argument('--pm', type=float, default=0.008, help='genetic mutation probability')
    parser.add_argument('--im_size', type=str, help='size, [rows]x[cols]')
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if args.im_size:
        im_size = [int(e) for e in args.im_size.split('x')]
    else:
        im_size = None

    ga = GA(
        args.control_im_path,
        args.pop_size,
        args.dna_size,
        args.max_iter,
        args.pc,
        args.pm,
        im_size,
    )
    for i, (pop, fit) in enumerate(ga.optret(ga.evolve)()):
        j = np.argmax(fit)
        per = pop[j]
        per_fit = ga.perfit(per)
        print(f'{i:0>5} {per_fit}')
        skimage.io.imsave(os.path.join(args.save_dir, f'{i:0>5}.jpg'), ga.decode(per))


if __name__ == '__main__':
    main()
