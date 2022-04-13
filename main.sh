#!/bin/bash

PROJPATH="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source .env.sh
cd $PROJPATH

################################################################################
######################## Checking if the features exist ########################
################################################################################
MISSINGFEAT=""
for DS in "cifarfs" "tieredimagenet"; do
  for DT in "novel" "val"; do
    for BB in "resnet10" "resnet18" "resnet34" "resnet50" "resnet101"; do
      if [[ ! -f "./features/${DS}_${DT}_${BB}.pth" ]]; then
        MISSINGFEAT="./features/${DS}_${DT}_${BB}.pth"
      fi
    done
  done
done

if [[ ${MISSINGFEAT} != "" ]]; then
  echo "Features file ${MISSINGFEAT} is missing. Running ./features/download.sh ..."
  ./features/download.sh
fi

################################################################################
######################### Running the configs in a loop ########################
################################################################################
CFGPREFIXLIST=("01_firth_1layer/firth_1layer" \
               "02_l2_1layer/l2_1layer" \
               "03_imbal_1layer/imbal_1layer" \
               "04_firth_3layer/firth_3layer" \
               "05_l2_3layer/l2_3layer" \
               "06_imbal_3layer/imbal_3layer" \
               "07_firth_5_10way/firth_5_10way" \
               "08_firth_tieredcifar/firth_tieredcifar" \
               "09_l2_tieredcifar/l2_tieredcifar" \
               "10_firth_densemobile/firth_densemobile" \
               "11_entropy/entropy")

mkdir -p joblogs
for CFGPREFIX in "${CFGPREFIXLIST[@]}"; do
  OUTLOG="./joblogs/${CFGPREFIX##*/}.out"
  echo "Running Configuration $CFGPREFIX"
  echo "  ==> The json config will be read from ./configs/${CFGPREFIX}.json"
  echo "  ==> The results csv file will be saved at ./results/${CFGPREFIX}.csv"
  echo "  ==> The training logs will be saved at ${OUTLOG}"
  echo "  + python main.py --device cuda:0 --configid ${CFGPREFIX} > $OUTLOG 2>&1"
  python main.py --device cuda:0 --configid ${CFGPREFIX} > $OUTLOG 2>&1
  echo "----------------------------------------"
done

