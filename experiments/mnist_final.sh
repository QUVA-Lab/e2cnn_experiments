#!/bin/bash


# C16 regular model
./mnist_final_single.sh --N 16                     --model E2SFCNN      --F "None" --fixparams --S 6

# D16 -> C16 regular model
./mnist_final_single.sh --N 16 --flip --restrict 5 --model E2SFCNN      --F "None" --fixparams --S 6

# C16 quotient model
./mnist_final_single.sh --N 16                     --model E2SFCNN_QUOT --F "None" --fixparams --S 6

