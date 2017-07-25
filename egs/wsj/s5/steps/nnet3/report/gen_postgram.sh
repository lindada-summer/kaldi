#!/bin/bash
# Hossein Hadian


acoustic_scale=1.0
online_ivector_dir=
feat_type=raw
granularity=sets
iter=final

. ./cmd.sh
. ./utils/parse_options.sh

if [ $# != 4 ]; then
   echo "Get the posteriorgram for an utterance using an nnet3 model (CE, LFR, disc, chain, end2end)."
   echo "usage: $0 <utt-id> <data-dir> <lang-dir> <nnet3-model-di>"
   echo "e.g.:  $0 utt-0001 data/test data/lang exp/chain/7k"
   echo "some of the options: "
   echo "--iter iter"
   exit 1;
fi

. ./path.sh


uid=$1
data=$2
lang=$3
dir=$4

for f in $dir/$iter.mdl $data/feats.scp; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

if [ $feat_type != "raw" ]; then
  echo "Only raw is supported for now."
  exit 1;
fi


cmvn_opts=`cat $dir/cmvn_opts` || exit 1
feats="ark,s,cs:grep $uid $data/feats.scp | apply-cmvn $cmvn_opts --utt2spk=ark:$data/utt2spk scp:$data/cmvn.scp scp:- ark:- |"

if [ ! -z "$online_ivector_dir" ]; then
  ivector_period=$(cat $online_ivector_dir/ivector_period) || exit 1;
  ivector_opts="--online-ivectors=scp:$online_ivector_dir/ivector_online.scp --online-ivector-period=$ivector_period"
fi

frame_subsampling_opt=
if [ -f $dir/frame_subsampling_factor ]; then
  # e.g. for 'chain' systems
  frame_subsampling_opt="--frame-subsampling-factor=$(cat $dir/frame_subsampling_factor)" || exit 1
fi

if [ -z "$acoustic_scale" ]; then
  if grep -q chain $dir/log/train.0.1.log; then
    acoustic_scale=1.0
  else
    acoustic_scale=0.1
  fi
fi

if [ "$granularity" == "sets" ]; then
  phones=$(cat $lang/phones/sets.txt | awk '{ if ($1~".*_B") {printf substr($1,0,length($1)-2) ","}  else  {printf $1 ","}}')
elif [ "$granularity" == "phones" ]; then
  phones=$(cat $lang/phones.txt | awk '{ if ($1!~"<eps>.*" && $1!~"#.*") {printf $1",";}}')
fi

transcript=$(grep $uid $data/text | cut -d' ' -f2-)
outdir=postgrams/$(basename $dir)/$iter
mkdir -p $outdir
nnet3-compute-postgram --use-gpu=no --acoustic-scale=$acoustic_scale --granularity=$granularity \
                       --phone-sets-file=$lang/phones/sets.int $frame_subsampling_opt \
                       $ivector_opts $dir/$iter.mdl "$feats" ark,t:- | tee mat.txt | \
                       steps/nnet3/report/matrix-to-postgram.py --phones "$phones" \
                       --transcript "$transcript" --outdir $outdir \
                       --title-prefix "$(basename $dir)/$iter.mdl"

