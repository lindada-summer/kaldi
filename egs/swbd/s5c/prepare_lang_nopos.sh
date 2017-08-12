#!/bin/sh

utils/prepare_lang.sh --position-dependent-phones false data/local/dict "<unk>" data/local/lang_nopos data/lang_nopos
LM=data/local/lm/sw1.o3g.kn.gz
srilm_opts="-subset -prune-lowprobs -unk -tolower -order 3"
utils/format_lm_sri.sh --srilm-opts "$srilm_opts" \
                       data/lang_nopos $LM data/local/dict/lexicon.txt data/lang_nopos_sw1_tg
LM=data/local/lm/sw1_fsh.o4g.kn.gz
utils/build_const_arpa_lm.sh $LM data/lang_nopos data/lang_nopos_sw1_fsh_fg
