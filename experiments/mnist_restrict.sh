#!/bin/bash

INTERPOLATION=3
# frequency cut-off policy; use the default one
F="None"

# layers where to perform restriction
res=("1" "2" "3" "4" "5")

for i in $(seq 1 5)
do
    for r in "${res[@]}"
    do
        # D_16 -> C_16 model on MNIST rot
        ./mnist_bench_single.sh --S 1 --type regular  --dataset "mnist_rot" --N 16 --flip --sgid 16 --restrict $r --F "$F" --sigma "None" --fixparams
        # D_16 -> C_1={e} model on MNIST 12k
        ./mnist_bench_single.sh --S 1 --type regular  --dataset "mnist12k"  --N 16 --flip --sgid 1  --restrict $r --F "$F" --sigma "None" --fixparams
        # C_16 -> C_1={e} model on MNIST 12k
        ./mnist_bench_single.sh --S 1 --type regular  --dataset "mnist12k"  --N 16        --sgid 1  --restrict $r --F "$F" --sigma "None" --fixparams
    done
done
