#!/bin/bash

for i in $(seq 1 5)
do  
    
    # Flip Equivariant Small Model D1D1D1
    ./stl10_single.sh --augment --deltaorth --model e2wrn16_8_stl --restrict 0

    # Flip Equivariant Large Model D1D1D1
    ./stl10_single.sh --augment --deltaorth --model e2wrn16_8_stl --restrict 0 --fixparams
    
    # Equivariant Small Model D8D4D1
    ./stl10_single.sh --augment --deltaorth --model e2wrn16_8_stl --restrict 3

    # Equivariant Large Model D8D4D1
    ./stl10_single.sh --augment --deltaorth --model e2wrn16_8_stl --restrict 3 --fixparams --split

    # Conventional Model (with DeltaOrthogonal)
    ./stl10_single.sh --augment --deltaorth --model wrn16_8_stl
    
done
