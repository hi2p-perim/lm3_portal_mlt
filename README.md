# lm3_portal_mlt

Code repository for "[Portal-Based Path Perturbation for Metropolis Light Transport](http://lightmetrica.org/h-otsu/project/portal_mlt/)". We implemented the approach on [Lightmetrica Version 3](https://github.com/lightmetrica/lightmetrica-v3). Please refer to the [documentation](https://lightmetrica.github.io/lightmetrica-v3-doc) for the project setup.

## Build (e.g, Windows)

Clone and setup environment:

    $ cd {repository root}
    $ git clone --recursive git@github.com:lightmetrica/lightmetrica-v3.git
    $ cd lightmetrica-v3
    $ conda env create -f environment.yml

Build:

    $ cd {repository root}
    $ mkdir build
    $ cd build
    $ conda activate lm3_dev
    $ cmake -G "Visual Studio 15 2017 Win64" ..
    $ 