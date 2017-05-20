

# in run.sh:
utils/data/segment_data.sh data/train_nodup data/train_nodup_seg
python utils/perturb_speed_to_allowed_lengths.py 12 data/train_nodup_seg data/train_nodup_seg_sp
fix_data_dir.sh data/train_nodup_seg_sp/
steps/make_mfcc.sh --nj 70 --mfcc-config conf/mfcc_hires.conf --cmd queue.pl data/train_nodup_seg_sp exp/make_hires/train_seg mfcchires_seg
steps/compute_cmvn_stats.sh data/train_nodup_seg_sp exp/make_hires/train_seg mfcchires_seg


gmm-init-mono $shared_phones_opt "--train-feats=$feats subset-feats --n=10 ark:- ark:-|" $lang/topo $feat_dim \
    $dir/0.mdl $dir/tree || exit 1;
fi

numgauss=`gmm-info --print-args=false $dir/0.mdl | grep gaussians | awk '{print $NF}'`
incgauss=$[($totgauss-$numgauss)/$max_iter_inc] # per-iter increment for #Gauss

if [ $stage -le -2 ]; then
  echo "$0: Compiling training graphs"
  $cmd JOB=1:$nj $dir/log/compile_graphs.JOB.log \
    compile-train-graphs --read-disambig-syms=$lang/phones/disambig.int $dir/tree $dir/0.mdl  $lang/L.fst \
    "ark:sym2int.pl --map-oov $oov_sym -f 2- $lang/words.txt < $sdata/JOB/text|" \
    "ark:|gzip -c >$dir/fsts.JOB.gz" || exit 1;
fi

