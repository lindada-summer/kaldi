#!/bin/sh
set -e

trainset=train_si284
python utils/perturb_speed_to_allowed_lengths.py 12 data/${trainset} data/${trainset}_spEx_hires
cat data/${trainset}_spEx_hires/utt2dur |  awk '{print $1 " " substr($1,5)}' >data/${trainset}_spEx_hires/utt2uniq
utils/fix_data_dir.sh data/${trainset}_spEx_hires
steps/make_mfcc.sh --nj 70 --mfcc-config conf/mfcc_hires.conf --cmd queue.pl data/${trainset}_spEx_hires exp/make_hires/train_spEx_hires mfcchires_spEx
steps/compute_cmvn_stats.sh data/${trainset}_spEx_hires exp/make_hires/train_spEx_hires mfcchires_spEx
