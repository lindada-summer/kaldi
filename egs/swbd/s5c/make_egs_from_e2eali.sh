#!/bin/bash

set -e -o pipefail

stage=0
nj=30
left_tolerance=1
right_tolerance=1
trainset=train_nodup_sp
# End configuration section.
echo "$0 $@"  # Print the command line for logging

. cmd.sh
. ./path.sh
. ./utils/parse_options.sh

chaindir=exp/chain/tdnn_7k_sp
lats_dir=exp/chain/e2emmi_dim800_chain_dataEx1-shared-tr1sl1_lats3
dir=exp/chain/e2emmi_dim800_chain_dataEx1-shared-tr1sl1_egs

left_context=`cat $chaindir/egs/info/left_context`
right_context=`cat $chaindir/egs/info/right_context`
left_context_initial=`cat $chaindir/egs/info/left_context_initial`
right_context_final=`cat $chaindir/egs/info/right_context_final`
frames_per_eg=`cat $chaindir/egs/info/frames_per_eg`
frame_subsampling_factor=`cat $chaindir/frame_subsampling_factor`
cmvn_opts=`cat $chaindir/cmvn_opts`

if [ $stage -le 1 ]; then
  echo "$0: generating egs..."
  steps/nnet3/chain/get_egs.sh --cmd queue.pl --alignment-subsampling-factor 1 \
             --left-tolerance $left_tolerance --right-tolerance $right_tolerance \
             --left-context $left_context --right-context $right_context \
             --left-context-initial $left_context_initial --right-context-final $right_context_final \
             --frames-per-eg $frames_per_eg --frames-per-iter 1500000 \
             --frame-subsampling-factor $frame_subsampling_factor \
             --cmvn-opts "$cmvn_opts" \
             --online-ivector-dir exp/nnet3/ivectors_${trainset} \
             data/${trainset}_hires $chaindir \
             $lats_dir $dir
fi
