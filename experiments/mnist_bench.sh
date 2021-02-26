#!/bin/bash


datasets=("mnist_rot" "mnist12k" "mnist_fliprot" )
Ns=("2" "3" "4" "5" "6" "7" "8" "9" "12" "16" "20")

# frequency cut-off policy; use the default one
F="None"

for i in $(seq 1 5)
do

    # C_N and C_D regular and quotient models and conventional CNN
    for dataset in "${datasets[@]}"
    do
        # baseline standard CNN
        ./mnist_bench_single.sh --S 1 --model CNN --type "None"  --dataset "$dataset" --N 1        --F "$F" --sigma "None" --fixparams --regularize

        # steerable CNN with C_1 equivariance (equivalent to a standard CNN)
        ./mnist_bench_single.sh --S 1             --type regular --dataset "$dataset" --N 1        --F "$F" --sigma "None" --fixparams --regularize

        # steerable CNN with D_1 equivariance (flip equivariance)
        ./mnist_bench_single.sh --S 1             --type regular --dataset "$dataset" --N 1 --flip --F "$F" --sigma "None" --fixparams --regularize

        for N in "${Ns[@]}"
        do
            # C_N
            ./mnist_bench_single.sh --S 1 --type regular --dataset "$dataset" --N $N        --F "$F" --sigma "None" --fixparams --regularize
            # D_N
            ./mnist_bench_single.sh --S 1 --type regular --dataset "$dataset" --N $N --flip --F "$F" --sigma "None" --fixparams --regularize

            # C_N with quotient representations
            ./mnist_bench_single.sh --S 1 --type quotient --dataset "$dataset" --N $N       --F "$F" --sigma "None" --fixparams --regularize
        done
    done

    # Other D_N and C_N models, only instantiated for N=16
    for dataset in "${datasets[@]}"
    do

        ./mnist_bench_single.sh --S 1 --type "scalarfield" --dataset "$dataset" --N 16        --F "$F" --sigma "None" --fixparams --regularize
        ./mnist_bench_single.sh --S 1 --type "scalarfield" --dataset "$dataset" --N 16 --flip --F "$F" --sigma "None" --fixparams --regularize
        
        dicrete_types=( "vectorfield" "regvector")
        # only C_N models
        for type in "${dicrete_types[@]}"
        do
            ./mnist_bench_single.sh --S 1 --type "$type" --dataset "$dataset" --N 16       --F "$F" --sigma "None" --fixparams --regularize
        done
    done
    
    # SO(2) and O(2) models
    # frequency is encoded with a minus sign, to distinguish it from the order of C_N when passed as an argumetn with --N
    freq=("-1" "-3" "-5" "-7")
    for dataset in "${datasets[@]}"
    do
        # O(2) invariant network using only isotropic filters
        ./mnist_bench_single.sh --S 1 --type "trivial" --dataset "$dataset" --N -1 --flip  --F "$F" --sigma "None" --fixparams --regularize

        # experiment with different irrep types, up to frequency "-$f"
        for f in "${freq[@]}"
        do
            ./mnist_bench_single.sh --S 1 --type "hnet_conv" --dataset "$dataset" --N $f        --F "$F" --sigma "None" --fixparams --regularize
            ./mnist_bench_single.sh --S 1 --type "hnet_conv" --dataset "$dataset" --N $f --flip --F "$F" --sigma "None" --fixparams --regularize

            ./mnist_bench_single.sh --S 1 --type "realhnet"  --dataset "$dataset" --N $f        --F "$F" --sigma "None" --fixparams --regularize
            ./mnist_bench_single.sh --S 1 --type "realhnet2" --dataset "$dataset" --N $f        --F "$F" --sigma "None" --fixparams --regularize

            ./mnist_bench_single.sh --S 1 --type "inducedhnet_conv" --dataset "$dataset" --N $f --flip --F "$F" --sigma "None" --fixparams --regularize

        done

        # other SO(2) models
        ./mnist_bench_single.sh --S 1 --type "squash"      --dataset "$dataset" --N -3       --F "$F" --sigma "None" --fixparams --regularize
        ./mnist_bench_single.sh --S 1 --type "hnet_norm"   --dataset "$dataset" --N -3       --F "$F" --sigma "None" --fixparams --regularize
        ./mnist_bench_single.sh --S 1 --type "sharednorm"  --dataset "$dataset" --N -3       --F "$F" --sigma "None" --fixparams --regularize
        ./mnist_bench_single.sh --S 1 --type "sharednorm2" --dataset "$dataset" --N -3       --F "$F" --sigma "None" --fixparams --regularize

        ./mnist_bench_single.sh --S 1 --type "gated_conv"        --dataset "$dataset" --N -3       --F "$F" --sigma "None" --fixparams --regularize
        ./mnist_bench_single.sh --S 1 --type "gated_conv_shared" --dataset "$dataset" --N -3       --F "$F" --sigma "None" --fixparams --regularize
        ./mnist_bench_single.sh --S 1 --type "gated_norm"        --dataset "$dataset" --N -3       --F "$F" --sigma "None" --fixparams --regularize
        ./mnist_bench_single.sh --S 1 --type "gated_norm_shared" --dataset "$dataset" --N -3       --F "$F" --sigma "None" --fixparams --regularize

        # other O(2) models
        ./mnist_bench_single.sh --S 1 --type "gated_conv"        --dataset "$dataset" --N -3 --flip --F "$F" --sigma "None" --fixparams --regularize
        ./mnist_bench_single.sh --S 1 --type "gated_norm"        --dataset "$dataset" --N -3 --flip --F "$F" --sigma "None" --fixparams --regularize
        ./mnist_bench_single.sh --S 1 --type "inducedgated_conv" --dataset "$dataset" --N -3 --flip --F "$F" --sigma "None" --fixparams --regularize
        ./mnist_bench_single.sh --S 1 --type "inducedgated_norm" --dataset "$dataset" --N -3 --flip --F "$F" --sigma "None" --fixparams --regularize

    done

done

