#!/bin/bash

SEEDS=1
N="16"
MODEL="E2SFCNN"
RESTRICT="0"
DATASET="mnist_rot"
J="-1"
SGID=""
F="None"
sigma="None"

FLIP=0

FIXPARAMS=0
DELTAORTH=0

INTERPOLATION=0


while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    --N)
    N="$2"
    shift
    shift
    ;;
    --model)
    MODEL="$2"
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
    --fixparams)
    FIXPARAMS=1
    shift
    ;;
    --sgid)
    SGID="$2"
    shift
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


PARAMS="--dataset=$DATASET --model=$MODEL --N=$N --restrict=$RESTRICT --F=$F --sigma=$sigma --epochs=40 --lr=0.015 --batch_size=64 --augment --time_limit=300 --verbose=2 --optimizer=sfcnn --l1 --adapt_lr=exponential --lr_decay_start=15 --lr_decay_factor=0.8 --lr_decay_epoch=1 --no_earlystop"
PARAMS="$PARAMS --interpolation=$INTERPOLATION"

if [ "$SGID" != "" ]; then
    PARAMS="$PARAMS --sgsize=$SGID"
fi

if [ "$FLIP" -eq "1" ]; then
    PARAMS="$PARAMS --flip"
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

python multiple_exps.py --S=$SEEDS $PARAMS
#python count_parameters.py $PARAMS



