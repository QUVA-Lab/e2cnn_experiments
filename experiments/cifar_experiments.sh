#!/bin/bash


for i in $(seq 1 3)
do  

    # small D8D4D1 models
    ./cifar_single.sh --fixparams --model e2wrn28_7 --restrict 3 --dataset cifar10
    ./cifar_single.sh --fixparams --model e2wrn28_7 --restrict 3 --dataset cifar100

    # big D8D4D1 models
    ./cifar_single.sh --fixparams --model e2wrn28_10 --restrict 3 --dataset cifar10
    ./cifar_single.sh --fixparams --model e2wrn28_10 --restrict 3 --dataset cifar100

    # D8D4D4 models
    ./cifar_single.sh --fixparams --model e2wrn28_10 --restrict 1 --dataset cifar10
    ./cifar_single.sh --fixparams --model e2wrn28_10 --restrict 1 --dataset cifar100

    # C8C4C1 models
    ./cifar_single.sh --fixparams --model e2wrn28_10R --restrict 3 --dataset cifar10
    ./cifar_single.sh --fixparams --model e2wrn28_10R --restrict 3 --dataset cifar100

    # D1D1D1 models
    ./cifar_single.sh --fixparams --model e2wrn28_10 --restrict 0 --dataset cifar10
    ./cifar_single.sh --fixparams --model e2wrn28_10 --restrict 0 --dataset cifar100


    # AutoAugment Experiments

    # small D8D4D1 models + AA
    ./cifar_single.sh --fixparams --model e2wrn28_7 --restrict 3 --augment --dataset cifar10
    ./cifar_single.sh --fixparams --model e2wrn28_7 --restrict 3 --augment --dataset cifar100

    # big D8D4D1 models + AA
    ./cifar_single.sh --fixparams --model e2wrn28_10 --restrict 3 --augment --dataset cifar10
    ./cifar_single.sh --fixparams --model e2wrn28_10 --restrict 3 --augment --dataset cifar100
    
done
