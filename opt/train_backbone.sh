#!/bin/bash

SCRIPTDIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJPATH="$(dirname $SCRIPTDIR)"
cd $PROJPATH
source .env.sh
cd $PROJPATH

################################################################################
###################### Sequential Job Array Specification ######################
################################################################################
DATASETNAMELIST=("miniimagenet" "cifarfs" "tieredimagenet")
RESNETNOLIST=("10" "18" "34" "50" "101")
DEVICE="cuda:0"

# In case you are suffering from an extremely slow storage
# while having a large amount of RAM, you may find it useful
# to set CPDATA2RAM=1. If you set CPDATA2RAM=1, the script will
# copy the whole dataset into the RAM, and then use the RAM
# copy for the pytorch dataset/data-loader. If your storage is
# reasonably fast and is not the bottleneck, then you do not
# need to turn the CPDATA2RAM feature on.
CPDATA2RAM=0

################################################################################
############################## Utility Function ################################
################################################################################
function parcp() {
  # parallel copying of a directory (replacing cp due to slowness)
  # usage: parcp source_dir dest_dir
  SOURCEDIR="$1"   #Ex: "./datasets/tieredimagenet"
  TARGETDIR="$2"   #Ex: "/dev/shm/tieredimagenet"

  rm -rf $TARGETDIR
  mkdir -p $TARGETDIR
  CP_PARALLEL=10
  nroffiles=$(ls "$SOURCEDIR" | wc -w)
  setsize=$(( nroffiles/CP_PARALLEL + 1 ))
  shopt -s lastpipe
  find "$SOURCEDIR" -maxdepth 1 -mindepth 1 | xargs -n "$setsize" | while read workset; do
    cp -rL $workset $TARGETDIR &
  done
  shopt -u lastpipe
  wait
}

################################################################################
######################## Checking if the datasets exist ########################
################################################################################
EXITGRACEFULLY="0"
for DATASETNAME in "${DATASETNAMELIST[@]}"; do
  echo "* Checking the existence of the ${DATASETNAME} dataset"
  if [[ ${DATASETNAME} == "miniimagenet" ]]; then
    SAMPLEFILE="${PROJPATH}/datasets/miniimagenet/mini-imagenet-cache-train.pkl"
    if [[ ! -f ${SAMPLEFILE} ]]; then
      echo "  ! Dataset file ${SAMPLEFILE} is missing."
      echo "   --> Please run ${PROJPATH}/datasets/download.sh to download it."
      EXITGRACEFULLY="1"
    fi
  elif [[ ${DATASETNAME} == "cifarfs" ]]; then
    SAMPLEFILE="${PROJPATH}/datasets/cifarfs/apple/red_delicious_s_000412.png"
    if [[ ! -f ${SAMPLEFILE} ]]; then
      echo "  ! Dataset file ${SAMPLEFILE} is missing."
      echo "   --> Please run ${PROJPATH}/datasets/download.sh to download it."
      EXITGRACEFULLY="1"
    fi
  elif [[ ${DATASETNAME} == "tieredimagenet" ]]; then
    SAMPLEFILE="${PROJPATH}/datasets/tieredimagenet/n01440764/n01440764_10026.JPEG"
    if [[ ! -f ${SAMPLEFILE} ]]; then
      echo "  ! Dataset file ${SAMPLEFILE} is missing."
      echo "   --> Please soft-link or place an imagenet extracted directory"
      echo "       at ${PROJPATH}/datasets/tieredimagenet."
      EXITGRACEFULLY="1"
    fi
  fi
done

if [[ ${EXITGRACEFULLY} == "1" ]]; then
  echo "* Exiting gracefully due to the absence of dataset files."
  exit 0
fi

################################################################################
######################## Running the trainings in a loop #######################
################################################################################
mkdir -p joblogs
for DATASETNAME in "${DATASETNAMELIST[@]}"; do
  DATAROOTOPT=""
  if [[ $CPDATA2RAM == "1" ]]; then
    echo Copying the whole ${DATASETNAME} dataset into the RAM.
    DEVSHMTRG="/dev/shm/${DATASETNAME}"
    REALDATASETPATH=$(realpath ./datasets/"${DATASETNAME}")
    echo "  + cp -r ${REALDATASETPATH} ${DEVSHMTRG}"
    parcp ${REALDATASETPATH} ${DEVSHMTRG} && DATAROOTOPT="--dataroot ${DEVSHMTRG} "
  fi

  for RESNETNO in "${RESNETNOLIST[@]}"; do
    OUTLOG="./joblogs/resnet${RESNETNO}_${DATASETNAME}.out"
    OUTLOG=$(readlink -m ${OUTLOG})
    echo "* Training ResNet${RESNETNO} on the ${DATASETNAME} dataset and the ${DEVICE} device"
    echo "  ==> The checkpoints will be saved at "
    echo "      ${PROJPATH}/backbones/${DATASETNAME}_resnet${RESNETNO}_v2_best.pth.tar and "
    echo "      ${PROJPATH}/backbones/${DATASETNAME}_resnet${RESNETNO}_v2_last.pth.tar"
    echo "  ==> The training log will be saved at ${OUTLOG}"
    echo -n "  + python opt/train_backbone.py --device ${DEVICE} --resnet_no ${RESNETNO} "
    echo "--dataset ${DATASETNAME} ${DATAROOTOPT}>> ${OUTLOG} 2>&1"
    python opt/train_backbone.py --device "${DEVICE}" --resnet_no "${RESNETNO}" \
      --dataset "${DATASETNAME}" ${DATAROOTOPT}>> ${OUTLOG} 2>&1
    echo "----------------------------------------"
    break
  done
  break

  if [[ $CPDATA2RAM == "1" ]]; then
    echo "Freeing up the RAM by removing ${DEVSHMTRG}"
    rm -rf $DEVSHMTRG
  fi
done
