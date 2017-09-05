// nnet3/nnet-chain-training.h

// Copyright    2015  Johns Hopkins University (author: Daniel Povey)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#ifndef KALDI_NNET3_NNET_CHAIN_TRAINING_H_
#define KALDI_NNET3_NNET_CHAIN_TRAINING_H_

#include "nnet3/nnet-example.h"
#include "nnet3/nnet-computation.h"
#include "nnet3/nnet-compute.h"
#include "nnet3/nnet-optimize.h"
#include "nnet3/nnet-chain-example.h"
#include "nnet3/nnet-training.h"
#include "chain/chain-training.h"
#include "chain/chain-den-graph.h"

namespace kaldi {
namespace nnet3 {

struct NnetChainTrainingOptions {
  NnetTrainerOptions nnet_config;
  chain::ChainTrainingOptions chain_config;
  bool apply_deriv_weights;
  int32 num_phone_sets, num_pdfs_to_tie;
  std::string write_pdf_map_filename;
  int32 num_pdfs_per_phone;
  NnetChainTrainingOptions(): apply_deriv_weights(true),
                              num_phone_sets(0),
                              percent_pdfs_to_tie(0),
                              num_pdfs_per_phone(1) { }

  void Register(OptionsItf *opts) {
    nnet_config.Register(opts);
    chain_config.Register(opts);
    opts->Register("apply-deriv-weights", &apply_deriv_weights,
                   "If true, apply the per-frame derivative weights stored with "
                   "the example");
    opts->Register("num-phone-sets", &num_phone_sets, "Number of phone sets -- "
                   "used for tying");
    opts->Register("percent-pdfs-to-tie", &percent_pdfs_to_tie,
                   "Percentage of pdfs to tie "
                   "at the end of this iteration");
    opts->Register("write-pdf-map-filename", &write_pdf_map_filename,
                   "filename to write the pdf tying map.");
    opts->Register("num-pdfs-per-phone", &num_pdfs_per_phone,
                   "Number of pdfs in "
                   "each phone according to the topo");
  }
};


/**
   This class is for single-threaded training of neural nets using the 'chain'
   model.
*/
class NnetChainTrainer {
 public:
  NnetChainTrainer(const NnetChainTrainingOptions &config,
                   const fst::StdVectorFst &den_fst,
                   Nnet *nnet);

  // train on one minibatch.
  void Train(const NnetChainExample &eg);

  // Prints out the final stats, and return true if there was a nonzero count.
  bool PrintTotalStats() const;

  // Prints out the max-change stats (if nonzero): the percentage of time that
  // per-component max-change and global max-change were enforced.
  void PrintMaxChangeStats() const;


  // #pdf_tying
  void DoPdfTying() const;

  ~NnetChainTrainer();
 private:
  void ProcessOutputs(const NnetChainExample &eg,
                      NnetComputer *computer);

  // Applies per-component max-change and global max-change to all updatable
  // components in *delta_nnet_, and use *delta_nnet_ to update parameters
  // in *nnet_.
  void UpdateParamsWithMaxChange();

  const NnetChainTrainingOptions opts_;

  chain::DenominatorGraph den_graph_;
  Nnet *nnet_;
  Nnet *delta_nnet_;  // Only used if momentum != 0.0 or max-param-change !=
                      // 0.0.  nnet representing accumulated parameter-change
                      // (we'd call this gradient_nnet_, but due to
                      // natural-gradient update, it's better to consider it as
                      // a delta-parameter nnet.
  CachingOptimizingCompiler compiler_;

  // This code supports multiple output layers, even though in the
  // normal case there will be just one output layer named "output".
  // So we store the objective functions per output layer.
  int32 num_minibatches_processed_;

  // stats for max-change.
  std::vector<int32> num_max_change_per_component_applied_;
  int32 num_max_change_global_applied_;

  unordered_map<std::string, ObjectiveFunctionInfo, StringHasher> objf_info_;

  // This is only used if we are collecting stats for #pdf_tying which is when
  // opts.num_pdfs_to_tie is set to some positive integer
  // This will store for each pdf pair, the total distance between their
  // posterior probabilities for all the frames processed in 1 iteration of
  // chain training
  //  unordered_map<std::pair<int32, int32>, BaseFloat, PairHasher<int32> >
  std::map<std::pair<int32, int32>, BaseFloat>
  pdf_pair_distance_;
  int32 num_frames_processed_;
};


} // namespace nnet3
} // namespace kaldi

#endif // KALDI_NNET3_NNET_CHAIN_TRAINING_H_
