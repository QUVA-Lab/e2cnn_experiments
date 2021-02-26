#!/bin/bash

#SBATCH -p gpu
#SBATCH -N 1


################  ignore this ############  SBATCH --time=30:00:00

#module load CUDA/9.0.176
#module load CUDA/10.0.130
#source activate e2cnn


SEEDS=1
N="8"

#MODEL="e2wrn28_10"
#MODEL="e2wrn28_10R"
#MODEL="e2wrn28_7"
MODEL="e2wrn28_7R"

RESTRICT="3"
DATASET="cifar10"
WD="0.0005"
FIXPARAMS=0
AUGMENT=0

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
    --S)
    SEEDS="$2"
    shift
    ;;
    *)    # unknown option
    shift # past argument
    ;;
esac
done


PARAMS="--dataset=$DATASET --model=$MODEL --N=$N  --restrict=$RESTRICT --F=1. --sigma=0.45"
TRAIN_PARAMS="--adapt_lr=exponential --epochs=200 --lr=0.1 --batch_size=128 --optimizer=SGD --momentum=0.9 --weight_decay=$WD  --eval_frequency=-1 --backup_model --no_earlystop --eval_test"
TRAIN_PARAMS="$TRAIN_PARAMS --lr_decay_epoch=60 --lr_decay_factor=0.2 --lr_decay_start=0"
#TRAIN_PARAMS="$TRAIN_PARAMS --lr_decay_schedule 60 120 180 --lr_decay_factor=0.2"

PARAMS="$PARAMS --deltaorth"

if [ "$FIXPARAMS" -eq "1" ]; then
    PARAMS="$PARAMS --fixparams"
fi

if [ "$AUGMENT" -eq "1" ]; then
    TRAIN_PARAMS="$TRAIN_PARAMS --augment"
fi

echo $DATASET

# python count_parameters.py $PARAMS $TRAIN_PARAMS
python multiple_exps.py --S=$SEEDS $PARAMS $TRAIN_PARAMS
#python -O multiple_exps.py --S=$SEEDS $PARAMS $TRAIN_PARAMS



