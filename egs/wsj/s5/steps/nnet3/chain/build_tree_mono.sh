#!/bin/bash
# Copyright 2012-2015  Johns Hopkins University (Author: Daniel Povey).
#  Apache 2.0.


# This script builds a tree for use in the 'chain' systems (although the script
# itself is pretty generic and doesn't use any 'chain' binaries).  This is just
# like the first stages of a standard system, like 'train_sat.sh', except it
# does 'convert-ali' to convert alignments to a monophone topology just created
# from the 'lang' directory (in case the topology is different from where you
# got the system's alignments from), and it stops after the tree-building and
# model-initialization stage, without re-estimating the Gaussians or training
# the transitions.


# Begin configuration section.
stage=-5
exit_stage=-100 # you can use this to require it to exit at the
                # beginning of a specific stage.  Not all values are
                # supported.
cmd=run.pl
context_opts=  # e.g. set this to "--context-width 5 --central-position 2" for quinphone.
cluster_thresh=-1  # for build-tree control final bottom-up clustering of leaves
frame_subsampling_factor=1
leftmost_questions_truncate=-1  # note: this used to default to 10, but we never
                                # use this option now with value != -1, and
                                # we're changing the default
tree_stats_opts=
cluster_phones_opts=
repeat_frames=false
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if [ $# != 5 ]; then
  echo "Usage: steps/train_sat.sh <#leaves> <data> <lang> <ali-dir> <exp-dir>"
  echo " e.g.: steps/train_sat.sh 2500 15000 data/train_si84 data/lang exp/tri2b_ali_si84 exp/tri3b"
  echo "Main options (for others, see top of script file)"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --config <config-file>                           # config containing options"
  echo "  --stage <stage>                                  # stage to do partial re-run from."
  echo "  --repeat-frames <true|false>                     # Only affects alignment conversion at"
  echo "                                                   # the end. If true, generate an "
  echo "                                                   # alignment using the frame-subsampled "
  echo "                                                   # topology that is repeated "
  echo "                                                   # --frame-subsampling-factor times "
  echo "                                                   # and interleaved, to be the same "
  echo "                                                   # length as the original alignment "
  echo "                                                   # (useful for cross-entropy training "
  echo "                                                   # of reduced frame rate systems)."
  exit 1;
fi

numleaves=$1
data=$2
lang=$3
alidir=$4
dir=$5

for f in $data/feats.scp $lang/phones.txt $alidir/final.mdl $alidir/ali.1.gz; do
  [ ! -f $f ] && echo "train_sat.sh: no such file $f" && exit 1;
done

oov=`cat $lang/oov.int`
nj=`cat $alidir/num_jobs` || exit 1;
silphonelist=`cat $lang/phones/silence.csl`
ciphonelist=`cat $lang/phones/context_indep.csl` || exit 1;
sdata=$data/split$nj;
splice_opts=`cat $alidir/splice_opts 2>/dev/null` # frame-splicing options.
cmvn_opts=`cat $alidir/cmvn_opts 2>/dev/null`
delta_opts=`cat $alidir/delta_opts 2>/dev/null`

mkdir -p $dir/log
cp $alidir/splice_opts $dir 2>/dev/null # frame-splicing options.
cp $alidir/cmvn_opts $dir 2>/dev/null # cmn/cmvn option.
cp $alidir/delta_opts $dir 2>/dev/null # delta option.

utils/lang/check_phones_compatible.sh $lang/phones.txt $alidir/phones.txt || exit 1;
cp $lang/phones.txt $dir || exit 1;

echo $nj >$dir/num_jobs
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;

# Set up features.

if [ -f $alidir/final.mat ]; then feat_type=lda; else feat_type=delta; fi
echo "$0: feature type is $feat_type"

## Set up speaker-independent features.
case $feat_type in
  delta) feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | add-deltas $delta_opts ark:- ark:- |";;
  lda) feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $alidir/final.mat ark:- ark:- |"
    cp $alidir/final.mat $dir
    cp $alidir/full.mat $dir 2>/dev/null
    ;;
  *) echo "$0: invalid feature type $feat_type" && exit 1;
esac

# Add fMLLR transforms if available
if [ -f $alidir/trans.1 ]; then
  echo "$0: Using transforms from $alidir"
  feats="$feats transform-feats --utt2spk=ark:$sdata/JOB/utt2spk ark,s,cs:$alidir/trans.JOB ark:- ark:- |"
fi

# Do subsampling of feats, if needed
if [ $frame_subsampling_factor -gt 1 ]; then
  feats="$feats subsample-feats --n=$frame_subsampling_factor ark:- ark:- |"
fi

if [ $stage -le -5 ]; then
  echo "$0: Initializing monophone model (for alignment conversion, in case topology changed)"

  [ ! -f $lang/phones/sets.int ] && exit 1;
  shared_phones_opt="--shared-phones=$lang/phones/sets.int"
  # get feature dimension
  example_feats="`echo $feats | sed s/JOB/1/g`";
  if ! feat_dim=$(feat-to-dim "$example_feats" - 2>/dev/null) || [ -z $feat_dim ]; then
    feat-to-dim "$example_feats" - # to see the error message.
    echo "error getting feature dimension"
    exit 1;
  fi
  $cmd JOB=1 $dir/log/init_mono.log \
    gmm-init-mono $shared_phones_opt "--train-feats=$feats subset-feats --n=10 ark:- ark:-|" $lang/topo $feat_dim \
      $dir/1.mdl $dir/tree || exit 1;
fi

if [ $stage -le -4 ]; then
  # Get tree stats.
  echo "$0: Converting alignments"
  $cmd JOB=1:$nj $dir/log/acc_tree.JOB.log \
     convert-ali --frame-subsampling-factor=$frame_subsampling_factor \
         $alidir/final.mdl $dir/1.mdl $dir/tree "ark:gunzip -c $alidir/ali.JOB.gz|" "ark:|gzip -c >$dir/ali.JOB.gz"
fi

cp $dir/1.mdl $dir/final.mdl

echo $0: Done building mono tree
