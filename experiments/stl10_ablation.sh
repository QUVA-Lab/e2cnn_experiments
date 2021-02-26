#!/bin/bash

sizes=("250" "500" "1000" "2000" "4000")

for i in $(seq 1 3)
do

    for s in "${sizes[@]}"
    do

        dataset="STL10cif|$s"

        # Equivariant Big Model D8D4D1
        ./stl10_exp.sh --augment --deltaorth --model e2wrn16_8 --restrict 3 --fixparams --split --dataset "$dataset" --validate

        # Equivariant Small Model D8D4D1
        ./stl10_exp.sh --augment --deltaorth --model e2wrn16_8 --restrict 3 --dataset "$dataset" --validate

        # Conventional Model (with DeltaOrthogonal)
        ./stl10_exp.sh --augment --deltaorth --model wrn16_8  --dataset "$dataset" --validate

    done
done
