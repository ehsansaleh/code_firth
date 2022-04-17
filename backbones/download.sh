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

FILEID="1LPxN-S2BquVc8mojkhZYIkZAWgj9uCd0"
FILENAME="backbones.tar"
GDRIVEURL="https://drive.google.com/file/d/1LPxN-S2BquVc8mojkhZYIkZAWgj9uCd0/view?usp=sharing"
PTHMD5FILE="backbones.md5"
RMAFTERDL="1"
ISSMALL="0"
gdluntar ${FILEID} ${FILENAME} ${GDRIVEURL} ${PTHMD5FILE} ${RMAFTERDL} ${ISSMALL}
