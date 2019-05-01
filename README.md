# Tiga

Use 100 triangles to generate any images you want.

![img](./res/firefox_out.png)

# Usage

```sh
# Install the requirements
$ pip3 install numpy scikit-image

# Run with arguments [Control image path] [Output dir]
$ python3 tiga.py ./res/firefox.png /tmp/img
```

Wait a few hours, you will get 3000 images, each closer to the firefox icon.

```text
usage: tiga.py [-h] [--pop_size POP_SIZE] [--dna_size DNA_SIZE]
               [--max_iter MAX_ITER] [--pc PC] [--pm PM] [--im_size IM_SIZE]
               control_im_path save_dir

positional arguments:
  control_im_path
  save_dir

optional arguments:
  -h, --help           show this help message and exit
  --pop_size POP_SIZE  population size
  --dna_size DNA_SIZE  dna size
  --max_iter MAX_ITER  population iterations
  --pc PC              genetic crossover probability
  --pm PM              genetic mutation probability
  --im_size IM_SIZE    size, default 128x128
```

# How does it works?

The principle is on my personal blog: [http://accu.cc/content/daze/ga/evolve/](http://accu.cc/content/daze/ga/evolve/). The article is in Chinese, and I am very sorry that there are no other languages available.

But simply put, I used Genetic-Algorithms.

# Licences

WTFPL
