#!/bin/bash


SEEDS=1
N="8"

#MODEL="e2wrn16_8"
MODEL="wrn16_8"

RESTRICT="3"
DATASET="STL10cif"
WD="0.0005"
F="1."
SIGMA="None"
FIXPARAMS=0
AUGMENT=0
DELTAORTH=0
SPLIT=0
VALIDATE=0

while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    --N)
    N="$2"
    shift
    ;;
    --restrict)
    RESTRICT="$2"
    shift
    ;;
    --weight_decay)
    WD="$2"
    shift
    ;;
    --F)
    F="$2"
    shift
    ;;
    --sigma)
    SIGMA="$2"
    shift
    ;;
    --model)
    MODEL="$2"
    shift
    ;;
    --dataset)
    DATASET="$2"
    shift
    ;;
    --fixparams)
    FIXPARAMS=1
    shift
    ;;
    --augment)
    AUGMENT=1
    shift
    ;;
    --deltaorth)
    DELTAORTH=1
    shift
    ;;
    --split)
    SPLIT=1
    shift
    ;;
    --validate)
    VALIDATE=1
    shift
    ;;
    --S)
    SEEDS="$2"
    shift
    ;;
    *)    # unknown option
    shift # past argument
    ;;
esac
done


PARAMS="--dataset=$DATASET --model=$MODEL --N=$N  --restrict=$RESTRICT --F=$F --sigma=$SIGMA"
TRAIN_PARAMS="--adapt_lr=exponential --epochs=1000 --lr=0.1 --optimizer=SGD --momentum=0.9 --weight_decay=$WD --eval_frequency=-5"
TRAIN_PARAMS="$TRAIN_PARAMS --lr_decay_schedule 300 400 600 800 --lr_decay_factor=0.2"

if [ "$VALIDATE" -eq "0" ]; then
TRAIN_PARAMS="$TRAIN_PARAMS --no_earlystop --eval_test"
fi

if [ "$SPLIT" -eq "1" ]; then
TRAIN_PARAMS="$TRAIN_PARAMS --eval_batch_size=128 --batch_size=64 --accumulate=2"
else
TRAIN_PARAMS="$TRAIN_PARAMS --eval_batch_size=128 --batch_size=128 --accumulate=1"
fi

if [ "$DELTAORTH" -eq "1" ]; then
TRAIN_PARAMS="$TRAIN_PARAMS --deltaorth"
fi

if [ "$FIXPARAMS" -eq "1" ]; then
    PARAMS="$PARAMS --fixparams"
fi

if [ "$AUGMENT" -eq "1" ]; then
    TRAIN_PARAMS="$TRAIN_PARAMS --augment"
fi

TRAIN_PARAMS="$TRAIN_PARAMS --store_plot --plot_frequency=-5 --backup_model --verbose=4"


echo $PARAMS
echo $TRAIN_PARAMS

#python -O multiple_exps.py --S=$SEEDS $PARAMS $TRAIN_PARAMS
python multiple_exps.py --S=$SEEDS $PARAMS $TRAIN_PARAMS
#python count_parameters.py $PARAMS $TRAIN_PARAMS



