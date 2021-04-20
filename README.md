# Local Binary Patterns: Overview and Application on Helicobacter Pylory

Contains all the useful scripts for performing experiments with LBP and its variants.

## Operating Enviroment

- Operational System: Debian 10 Buster and Ubuntu 20.04
- Python 3.7

## To install, run the following commands in the user directory on Linux.

```bash
# Download Anaconda from the official site (https://www.anaconda.com/products/individual#Downloads).
  $ wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
  
# Install Anaconda
  $ bash Anaconda3-2020.11-Linux-x86_64.sh

# Follow the steps showed and continue (sometimes, it is necessary restart the terminal).

# Create a vitural enviroment for experimentation:

  $ conda create -n venv_lbp python=3.7

# Activate the enviroment created:

  $ conda activate venv_lbp

# Change to github diretory:
  $ cd tcc/

# Install Librarys requireds:

  $ pip install -r requirements.txt

# Build the LBP library:

  $ python setup.py build_ext --inplace
```

## Example of use

```bash
# If you need of some help, run:
  $ python exec_run.py -h
  
  usage: exec_run.py [-h] --dataset DATASET --variant VARIANT [--method METHOD]
                     [--points POINTS] [--radius RADIUS] 
                     [--size_train SIZE_TRAIN] [--output OUTPUT] [--load LOAD]

  optional arguments:
    -h, --help            show this help message and exit
    --dataset DATASET, -d DATASET
                          Dataset that contains the images
    --variant VARIANT, -v VARIANT
                          Descritor LBP variant accepted: base_lbp,
                          improved_lbp, extended_lbp, completed_lbp, hamming_lbp
    --method METHOD, -m METHOD
                          Method Accepted: `nri_uniform` and `uniform`
    --points POINTS, -p POINTS
                          Number of points at neighboorhood
    --radius RADIUS, -r RADIUS
                          Radius of points at neighboorhood
    --size_train SIZE_TRAIN, -s SIZE_TRAIN
                          Length of train dataset
    --size_val SIZE_VAL, -s SIZE_VAL
                          Length of validation dataset
    --size_test SIZE_TEST, -s SIZE_TEST
                          Length of test dataset
    --output OUTPUT, -o OUTPUT
                          Path to output results
    --load LOAD, -l LOAD  Save descriptors computed

# For example, when running:
  $ python exec_run.py -d path_dataset -v completed_lbp -m uniform -p 8 -r 1 -s 0.8 -o results -l False
  
The code will perform the calculation of the Completed LBP descriptor on the ´path_dataset´ dataset using 
the uniform and invariant method of rotation, parameters P and R equal to 8 and 1, respectively, separating a
total of 80% for training and the rest for validation and testing, equaly.

The ´-o´ parameter indicates a folder where the results will be stored. If this folder does not exist,
the program will automatically create it.

The ´-l´ parameter is a boolean indicating whether you want to load the calculated descriptors in a `.txt` 
file so that it is not necessary to recalculate them. If you are going to run the experiment for the first
time, this argument should be ignored.

```
