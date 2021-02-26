#!/bin/bash

SEEDS=1
N="16"
MODEL="EXP"
TYPE="regular"
RESTRICT="0"
DATASET="mnist_rot"
J="-1"
F="None"
sigma="None"
SGID=""

FIXPARAMS=0
DELTAORTH=0
FLIP=0
INTERPOLATION=2
REGULARIZE=0

while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    --N)
    N="$2"
    shift
    shift
    ;;
    --J)
    J="$2"
    shift
    shift
    ;;
    --F)
    F="$2"
    shift
    shift
    ;;
    --sgid)
    SGID="$2"
    shift
    shift
    ;;
    --sigma)
    sigma="$2"
    shift
    shift
    ;;
    --restrict)
    RESTRICT="$2"
    shift
    shift
    ;;
    --flip)
    FLIP=1
    shift
    ;;
    --regularize)
    REGULARIZE=1
    shift
    ;;
    --fixparams)
    FIXPARAMS=1
    shift
    ;;
    --deltaorth)
    DELTAORTH=1
    shift
    ;;
    --interpolation)
    INTERPOLATION="$2"
    shift
    shift
    ;;
    --model)
    MODEL="$2"
    shift
    shift
    ;;
    --type)
    TYPE="$2"
    shift
    shift
    ;;
    --dataset)
    DATASET="$2"
    shift
    shift
    ;;
    --S)
    SEEDS="$2"
    shift
    shift
    ;;
    *)    # unknown option
    shift # past argument
    ;;
esac
done


PARAMS="--dataset=$DATASET --model=$MODEL --type=$TYPE --N=$N --restrict=$RESTRICT --F=$F --sigma=$sigma --interpolation=$INTERPOLATION --epochs=30 --lr=0.001 --batch_size=64 --augment --time_limit=300 --verbose=2  --adapt_lr=exponential --lr_decay_start=10 --reshuffle" #--no_earlystop

if [ "$SGID" != "" ]; then
    PARAMS="$PARAMS --sgsize=$SGID"
fi

if [ "$FLIP" -eq "1" ]; then
    PARAMS="$PARAMS --flip"
fi

if [ "$REGULARIZE" -eq "1" ]; then
    PARAMS="$PARAMS --weight_decay=0.0 --optimizer=sfcnn --lamb_fully_L2=0.0000001 --lamb_conv_L2=0.0000001 --lamb_bn_L2=0 --lamb_softmax_L2=0"
else
    PARAMS="$PARAMS --weight_decay=0.0 --optimizer=Adam"
    PARAMS="$PARAMS --lamb_fully_L2=0.0 --lamb_conv_L2=0.0 --lamb_bn_L2=0 --lamb_softmax_L2=0"
fi

if [ "$FIXPARAMS" -eq "1" ]; then
    PARAMS="$PARAMS --fixparams"
fi

if [ "$DELTAORTH" -eq "1" ]; then
    PARAMS="$PARAMS --deltaorth"
fi


if [ "$J" -ne "-1" ]; then
    PARAMS="$PARAMS --J=$J"
fi

echo $PARAMS

#python3.7 -O multiple_exps.py --S=$SEEDS $PARAMS
python multiple_exps.py --S=$SEEDS $PARAMS



