#!/bin/bash

# Usage: ./download.sh [base]
# note:  This file downloads the features from the google drive.
#        It only downloads the novel and val set features by default.
#        If you need the base features, run the script with the "base"
#        option. The raw links are also included, so feel free to
#        download and untar the files yourself.

SCRIPTDIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd $SCRIPTDIR

# importing the gdluntar helper function from the utils directory
source ../utils/bashfuncs.sh

echo "Working on miniimagent"
cd miniimagenet
FILEID="1FYqRTetn2Gh0G4REbHbUKV372Qc-OY9X"
FILENAME="miniimagenet.tar"
GDRIVEURL="https://drive.google.com/file/d/1FYqRTetn2Gh0G4REbHbUKV372Qc-OY9X/view?usp=sharing"
PTHMD5FILE="miniimagenet.md5"
RMAFTERDL="1"
ISSMALL="0"
gdluntar ${FILEID} ${FILENAME} ${GDRIVEURL} ${PTHMD5FILE} ${RMAFTERDL} ${ISSMALL}
cd ..
echo "========================================================================="

echo "Working on cifarfs"
cd cifarfs
FILEID="13OuOM3nflvISHt_WUjvWGbVUkI7sATCP"
FILENAME="cifarfs.tar"
GDRIVEURL="https://drive.google.com/file/d/13OuOM3nflvISHt_WUjvWGbVUkI7sATCP/view?usp=sharing"
PTHMD5FILE="cifarfs.md5"
RMAFTERDL="1"
ISSMALL="0"
gdluntar ${FILEID} ${FILENAME} ${GDRIVEURL} ${PTHMD5FILE} ${RMAFTERDL} ${ISSMALL}
cd ..
echo "========================================================================="

echo "Working on order json files"
FILEID="1LLPjeKzEzLYnytMzAPSuLbKHnc7-EBqS"
FILENAME="orders.tar"
GDRIVEURL="https://drive.google.com/file/d/1LLPjeKzEzLYnytMzAPSuLbKHnc7-EBqS/view?usp=sharing"
PTHMD5FILE="orders.md5"
RMAFTERDL="1"
ISSMALL="0"
gdluntar ${FILEID} ${FILENAME} ${GDRIVEURL} ${PTHMD5FILE} ${RMAFTERDL} ${ISSMALL}
echo "========================================================================="
