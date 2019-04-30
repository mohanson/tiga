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
    def __init__(self, control_im_path):
        self.pop_size = 80
        self.dna_size = 100
        self.max_iter = 3000
        self.pc = 0.6
        self.pm = 0.008

        im = skimage.io.imread(control_im_path)
        if im.shape[2] == 4:
            im = skimage.color.rgba2rgb(im)
            im = (255 * im).astype(np.uint8)
        self.control_im = skimage.transform.resize(
            im, (128, 128), mode='reflect', preserve_range=True).astype(np.uint8)

    def decode(self, per):
        im = np.ones(self.control_im.shape, dtype=np.uint8) * 255
        for e in per.base:
            rr, cc = skimage.draw.polygon(e.r, e.c)
            skimage.draw.set_color(im, (rr, cc), e.color, e.alpha)
        return im

    def perfit(self, per):
        im = self.decode(per)
        assert im.shape == self.control_im.shape
        # 三维矩阵的欧式距离
        d = np.linalg.norm(np.where(self.control_im > im, self.control_im - im, im - self.control_im))
        # 使用一个较大的数减去欧式距离
        # 此处该数为 (self.control_im.size * ((3 * 255 ** 2) ** 0.5) ** 2) ** 0.5
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
                r = np.random.randint(0, self.control_im.shape[0], 3, dtype=np.uint8)
                c = np.random.randint(0, self.control_im.shape[1], 3, dtype=np.uint8)
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
                    base.r = np.random.randint(0, self.control_im.shape[0], 3, dtype=np.uint8)
                    base.c = np.random.randint(0, self.control_im.shape[1], 3, dtype=np.uint8)
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
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    ga = GA(args.control_im_path)
    for i, (pop, fit) in enumerate(ga.optret(ga.evolve)()):
        j = np.argmax(fit)
        per = pop[j]
        per_fit = ga.perfit(per)
        print(f'{i:0>5} {per_fit}')
        skimage.io.imsave(os.path.join(args.save_dir, f'{i:0>5}.jpg'), ga.decode(per))


if __name__ == '__main__':
    main()
