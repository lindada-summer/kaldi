#!/bin/bash
# Copyright 2017  Johns Hopkins University (Author: Hossein Hadian)
# Apache 2.0


iter=76
echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 1 ]; then
  echo "Usage: steps/e2e_tie_nnet.sh [options] <chaindir>"
  exit 1;
fi

dir=$1


old_numpdfs=$(tree-info $dir/tree|grep num-pdfs|awk '{print $2}')
numpdfs=$(tail -1 $dir/pdf-map.txt|awk '{print $2}')
#sed "s|$old_numpdfs|$numpdfs|g" $dir/configs/final.config >$dir/configs/tied.config
tail -5 $dir/configs/final.config | \
  sed "s|$old_numpdfs|$numpdfs|g" | \
  sed "s|param-stddev=0.0|param-stddev=0.0|" | \
  sed "s|bias-stddev=0.0|bias-stddev=0.0|" >$dir/configs/tied.config
nnet3-am-copy --nnet-config=$dir/configs/tied.config $dir/$iter.mdl $dir/$iter.mdl
cp $dir/$iter.mdl $dir/$[$iter-1].mdl
rm $dir/cache*
