#!/bin/bash
# Copyright 2017  Johns Hopkins University (Author: Hossein Hadian)
# Apache 2.0


# To be run from ..
# Flat start chain model training.
### Currently it is not from flat-start. Later this will be implemented
### to work from flat start

# Begin configuration section.
lm_opts="--num-extra-lm-states=2000"
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 2 ]; then
  echo "Usage: $0 [options] <ali-dir> <out-dir>"
  exit 1;
fi

alidir=$1
dir=$2

mkdir -p $dir
gunzip -c $alidir/ali.*.gz | \
  ali-to-phones $alidir/final.mdl ark:- ark:- | \
  chain-est-phone-lm $lm_opts ark:- $dir/phone_lm.fst
