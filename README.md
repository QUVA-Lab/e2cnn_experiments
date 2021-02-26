# Experiments for General E(2)-Equivariant Steerable CNNs
--------------------------------------------------------------------------------
**[Paper](https://arxiv.org/abs/1911.08251)** | **[Library](https://github.com/QUVA-Lab/e2cnn)** 


## Getting Started - Environment


First, you can set up a Conda environment containing some packages required 

```
conda create --name e2exp python=3.6
source activate e2exp

conda install -y pytorch=1.3 torchvision cudatoolkit=10.0 -c pytorch
conda install -y -c conda-forge matplotlib
conda install -y scipy=1.5 pandas scikit-learn=0.23
conda install -y -c anaconda sqlite
```

Now, we add the [e2cnn](https://github.com/QUVA-Lab/e2cnn) library.
Since the environment has Python 3.6, we clone the [legacy_py3.6](https://github.com/QUVA-Lab/e2cnn/tree/legacy_py3.6)
branch.

NOTE: make sure you are in the `./experiments/` folder before running the following commands.

```
mkdir tmp_e2cnn
cd tmp_e2cnn
git clone --single-branch --branch legacy_py3.6 https://github.com/QUVA-Lab/e2cnn
mv e2cnn/e2cnn ../e2cnn
cd ..
rm -rf tmp_e2cnn
```

If you use Python 3.7 or higher, you can install the library just using
```
pip install e2cnn
```

These commands are already included in the file [setting_up_env.sh](./experiments/setting_up_env.sh), so you can also just run
```
cd experiments
./setting_up_env.sh
```

## Getting Started - Datasets

To automatically download the MNIST variants datasets, you can run the following commands 
(assuming you are in the `./experiments/` folder):

```
cd datasets
./download_mnist.sh

source activate e2exp

cd mnist_rot
python convert.py

cd ../mnist_fliprot
python convert.py

cd ../mnist12k
python convert.py

```


## Getting Started - Experiments

All the experiments can be run automatically through the following few scripts
(assuming you are in the `./experiments/` folder).


To run all the model benchmarking experiments on transformed MNIST datasets:
```
./mnist_bench.sh
```

To run the MNIST experiments with group restriction:
```
./mnist_restrict.sh
```

To run the competitive MNIST experiments:
```
./mnist_final.sh
```

To run the CIFAR10 and the CIFAR100 experiments:
```
./cifar_experiments.sh
```

To run the experiments on the full STL10 dataset:
```
./stl10_experiments.sh
```

To run the data ablation study on STL10
```
./stl10_ablation.sh
```

You can find more details about the single experiments in each bash file.


Experiments' logs and results are stored in a new `./results` folder.
A summary of all experiments can be printed with the `print_results.py` script.


## Cite

The development of the library and the experiments was part of the work done for our paper
[General E(2)-Equivariant Steerable CNNs](https://arxiv.org/abs/1911.08251).
Please cite this work if you use our code:

```
@inproceedings{e3cnn,
    title={{General E(2)-Equivariant Steerable CNNs}},
    author={Weiler, Maurice and Cesa, Gabriele},
    booktitle={Conference on Neural Information Processing Systems (NeurIPS)},
    year={2019},
}
```

Feel free to [contact us](mailto:cesa.gabriele@gmail.com,m.weiler@uva.nl).

## License

This code and the *e2cnn* library are distributed under BSD Clear license. See LICENSE file.
