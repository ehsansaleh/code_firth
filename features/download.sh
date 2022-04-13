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

FILEID="1cf57AsY3IHxlDGEvB4RemDrbtU92aSM0"
FILENAME="miniimagenet_novel.tar"
GDRIVEURL="https://drive.google.com/file/d/1cf57AsY3IHxlDGEvB4RemDrbtU92aSM0/view?usp=sharing"
PTHMD5FILE="miniimagenet_novel.md5"
gdluntar ${FILEID} ${FILENAME} ${GDRIVEURL} ${PTHMD5FILE}

FILEID="1wCoEyGoU4mFu0h8RaEB1FpWpMWjUbd1u"
FILENAME="miniimagenet_val.tar"
GDRIVEURL="https://drive.google.com/file/d/1wCoEyGoU4mFu0h8RaEB1FpWpMWjUbd1u/view?usp=sharing"
PTHMD5FILE="miniimagenet_val.md5"
gdluntar ${FILEID} ${FILENAME} ${GDRIVEURL} ${PTHMD5FILE}


FILEID="162hsEHUtvpS0-kX8MJXGFXYATWS4NXnS"
FILENAME="cifarfs_novel.tar"
GDRIVEURL="https://drive.google.com/file/d/162hsEHUtvpS0-kX8MJXGFXYATWS4NXnS/view?usp=sharing"
PTHMD5FILE="cifarfs_novel.md5"
gdluntar ${FILEID} ${FILENAME} ${GDRIVEURL} ${PTHMD5FILE}

FILEID="1UyqkgV-1ATDutTc8DS6FzfGU9RLGRx3d"
FILENAME="cifarfs_val.tar"
GDRIVEURL="https://drive.google.com/file/d/1UyqkgV-1ATDutTc8DS6FzfGU9RLGRx3d/view?usp=sharing"
PTHMD5FILE="cifarfs_val.md5"
gdluntar ${FILEID} ${FILENAME} ${GDRIVEURL} ${PTHMD5FILE}

FILEID="1ULphDbXW-zXZcK0-Hzcu8Q_uKynRnzcZ"
FILENAME="tieredimagenet_novel.tar"
GDRIVEURL="https://drive.google.com/file/d/1ULphDbXW-zXZcK0-Hzcu8Q_uKynRnzcZ/view?usp=sharing"
PTHMD5FILE="tieredimagenet_novel.md5"
gdluntar ${FILEID} ${FILENAME} ${GDRIVEURL} ${PTHMD5FILE}

FILEID="1L3KQvV7IDdaAFWLl6vuVzQC2Ym_yvdeT"
FILENAME="tieredimagenet_val.tar"
GDRIVEURL="https://drive.google.com/file/d/1L3KQvV7IDdaAFWLl6vuVzQC2Ym_yvdeT/view?usp=sharing"
PTHMD5FILE="tieredimagenet_val.md5"
gdluntar ${FILEID} ${FILENAME} ${GDRIVEURL} ${PTHMD5FILE}

# The following are base features, which are optional.
if [[ $1 == "base" ]]; then
  FILEID="1BGSICHLI3FI3zrGptHpJQKF1QBdu0fIp"
  FILENAME="miniimagenet_base.tar"
  GDRIVEURL="https://drive.google.com/file/d/1BGSICHLI3FI3zrGptHpJQKF1QBdu0fIp/view?usp=sharing"
  PTHMD5FILE="miniimagenet_base.md5"
  gdluntar ${FILEID} ${FILENAME} ${GDRIVEURL} ${PTHMD5FILE}

  FILEID="1yeELkq_oLf3_AnwW9Kv8qv_U51IKBCl3"
  FILENAME="cifarfs_base.tar"
  GDRIVEURL="https://drive.google.com/file/d/1yeELkq_oLf3_AnwW9Kv8qv_U51IKBCl3/view?usp=sharing"
  PTHMD5FILE="cifarfs_base.md5"
  gdluntar ${FILEID} ${FILENAME} ${GDRIVEURL} ${PTHMD5FILE}

  FILEID="1MnctphdBzoZzrzhsCIBf40bfOmaXIUf0"
  FILENAME="tieredimagenet_base.tar"
  GDRIVEURL="https://drive.google.com/file/d/1MnctphdBzoZzrzhsCIBf40bfOmaXIUf0/view?usp=sharing"
  PTHMD5FILE="tieredimagenet_base.md5"
  gdluntar ${FILEID} ${FILENAME} ${GDRIVEURL} ${PTHMD5FILE}
fi

