#!/bin/sh

trainset=train_nodup
# if not there, utils/data/get_utt2dur.sh data/train_nodup_seg

#cat data/train/text | \
#  utils/text_to_phones.py data/lang data/local/dict_charv1/lexicon.txt | \
#  utils/sym2int.pl -f 2- data/lang/phones.txt | \
#  chain-est-phone-lm --num-extra-lm-states=2000 ark:- posdep_phone_lm.fst

# in run.sh:
utils/data/segment_data.sh data/$trainset data/${trainset}_seg
python utils/perturb_speed_to_allowed_lengths.py 12 data/${trainset}_seg data/${trainset}_seg_spEx_hires
cat data/${trainset}_seg_spEx_hires/utt2dur |  awk '{print $1 " " substr($1,5)}' >data/${trainset}_seg_spEx_hires/utt2uniq
utils/data/fix_data_dir.sh data/${trainset}_seg_spEx_hires
steps/make_mfcc.sh --nj 70 --mfcc-config conf/mfcc_hires.conf --cmd queue.pl data/${trainset}_seg_spEx_hires exp/make_hires/train_seg_hires mfcchires_seg
steps/compute_cmvn_stats.sh data/${trainset}_seg_spEx_hires exp/make_hires/train_seg_spEx_hires mfcchires_seg

#local/chain/run_e2e_1f.sh
