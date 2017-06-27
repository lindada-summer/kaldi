#!/bin/sh

trainset=train_nodup
# in run.sh:
utils/data/segment_data.sh data/$trainset data/${trainset}_seg
python utils/perturb_speed_to_allowed_lengths.py 12 data/${trainset}_seg data/${trainset}_seg_sp_hires
cat data/train_nodup_seg_spEx/utt2dur |  awk '{print $1 " " substr($1,5)}' >data/train_nodup_seg_spEx/utt2uniq
fix_data_dir.sh data/${trainset}_seg_sp_hires
steps/make_mfcc.sh --nj 70 --mfcc-config conf/mfcc_hires.conf --cmd queue.pl data/${trainset}_seg_sp_hires exp/make_hires/train_seg_hires mfcchires_seg
steps/compute_cmvn_stats.sh ${trainset}_seg_sp_hires exp/make_hires/train_seg_hires mfcchires_seg

local/chain/run_e2e_1f.sh
