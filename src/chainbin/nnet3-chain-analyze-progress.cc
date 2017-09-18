// nnet3bin/nnet3-chain-analyze-progress.cc

// Copyright 2015 Johns Hopkins University (author:  Hossein Hadian)


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

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "hmm/transition-model.h"
#include "nnet3/nnet-utils.h"
#include "nnet3/nnet-diagnostics.h"
#include "nnet3/am-nnet-simple.h"
#include "chain/chain-kernels-ansi.h"
#include "chain/chain-numerator.h"
#include "chain/chain-training.h"
#include "chain/chain-denominator.h"
#include "chain/chain-num-graph.h"
#include "chain/chain-full-numerator.h"
#include "fstext/fstext-lib.h"
#include "nnet3/nnet-chain-example.h"

using namespace kaldi;
using namespace kaldi::nnet3;
using namespace fst;
using namespace kaldi::chain;
typedef kaldi::int32 int32;
typedef kaldi::int64 int64;

void Compute(const NnetComputeProbOptions &nnet_config,
             const chain::ChainTrainingOptions &chain_config,
             const Nnet &nnet,
             const NnetChainExample &chain_eg,
             CuMatrix<BaseFloat>* nnet_output) {
  CachingOptimizingCompiler compiler(nnet, nnet_config.optimize_config);
  Nnet *deriv_nnet;
  if (nnet_config.compute_deriv) {
    deriv_nnet = new Nnet(nnet);
    // bool is_gradient = true;
    // SetZero(is_gradient, deriv_nnet);
  }

  bool need_model_derivative = nnet_config.compute_deriv,
       store_component_stats = false;
  ComputationRequest request;
  bool use_xent_regularization = (chain_config.xent_regularize != 0.0),
       use_xent_derivative = false;
  GetChainComputationRequest(nnet, chain_eg, need_model_derivative,
                             store_component_stats, use_xent_regularization,
                             use_xent_derivative, &request);
  const NnetComputation *computation = compiler.Compile(request);
  NnetComputer computer(nnet_config.compute_config, *computation,
                        nnet, deriv_nnet);
  // give the inputs to the computer object.
  computer.AcceptInputs(nnet, chain_eg.inputs);
  computer.Run();

  CuMatrix<BaseFloat> tmp(computer.GetOutput("output"));
  nnet_output->Resize(tmp.NumRows(), tmp.NumCols());
  nnet_output->CopyFromMat(tmp);

  //if (nnet_config.compute_deriv)
  //  computer.Run();
}

void ReadSharedPhonesList(std::string rxfilename, std::vector<std::vector<int32> > *list_out) {
  list_out->clear();
  Input input(rxfilename);
  std::istream &is = input.Stream();
  std::string line;
  while (std::getline(is, line)) {
    list_out->push_back(std::vector<int32>());
    if (!SplitStringToIntegers(line, " \t\r", true, &(list_out->back())))
      KALDI_ERR << "Bad line in shared phones list: " << line << " (reading "
                << PrintableRxfilename(rxfilename) << ")";
    std::sort(list_out->rbegin()->begin(), list_out->rbegin()->end());
    if (!IsSortedAndUniq(*(list_out->rbegin())))
      KALDI_ERR << "Bad line in shared phones list (repeated phone): " << line
                << " (reading " << PrintableRxfilename(rxfilename) << ")";
  }
}

void AliToPhoneDurPairs(const TransitionModel& trans_model,
                        const std::map<int, int>& tid_map,
                        const std::vector<int32> ali,
                        std::vector<std::pair<int32, int32> >* pairs) {
  int32 dur = 1;
  for (int32 i = 1; i < ali.size(); i++) {
    if (tid_map.count(ali[i]) == 0)
      KALDI_ERR << "Not found tid:" << ali[i] << ". Transition model mismatch?\n";
    if (tid_map.at(ali[i]) != tid_map.at(ali[i - 1])) {
      auto pair = std::make_pair(tid_map.at(ali[i - 1]), dur);
      pairs->push_back(pair);
      dur = 0;
    }
    dur++;
  }
  pairs->push_back(std::make_pair(tid_map.at(ali[ali.size() - 1]), dur));
}

int main(int argc, char *argv[]) {
  try {

    const char *usage =
        "Show ali differences between two iterations in chain training.\n"
        "\n"
        "Usage:  nnet3-chain-analyze-progress [options] <old-nnet-am-in> <new-nnet-in>"
        " <training-examples-in>\n";

    ParseOptions po(usage);

    NnetComputeProbOptions nnet_opts;
    std::string phone_sets_file = "";
    chain::ChainTrainingOptions chain_opts;
    std::string use_gpu = "no";

    po.Register("use-gpu", &use_gpu,
                "yes|no|optional|wait, only has effect if compiled with CUDA");
    po.Register("phone-sets-file", &phone_sets_file,
                "Phone sets file. Required if granularity=sets");
    nnet_opts.Register(&po);
    chain_opts.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }
    chain_opts.viterbi = true;
    chain_opts.offset_first_transitions = true;
#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif

    std::string nnet1_rxfilename = po.GetArg(1),
                nnet2_rxfilename = po.GetArg(2),
                examples_rspecifier = po.GetArg(3);

    TransitionModel trans_model;
    AmNnetSimple am_nnet;
    {
      bool binary;
      Input ki(nnet1_rxfilename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_nnet.Read(ki.Stream(), binary);
    }
    Nnet nnet1 = am_nnet.GetNnet();
    Nnet nnet2;
    ReadKaldiObject(nnet2_rxfilename, &nnet2);

    SetBatchnormTestMode(true, &nnet1);
    SetDropoutTestMode(true, &nnet1);
    SetBatchnormTestMode(true, &nnet2);
    SetDropoutTestMode(true, &nnet2);

    std::map<int, int> phone_map;
    std::map<int, int> tid_map;  // maps PdfId's to phone-ids (phone-ids can be
                                 // either actual phones (phones.txt) or members
                                 // of a reduced set of phones (e.g. sets.txt))
    std::string granularity = "sets";  // tid_map will uniquely map from pdfs to
    // phones and we'll be fine because there is no tying in e2e models [yet]

    int32 granularity_dim = trans_model.NumPdfs();
    if (granularity == "sets") {
      KALDI_ASSERT(phone_sets_file != "");
      std::vector<std::vector<int32> > shared_phones;
      ReadSharedPhonesList(phone_sets_file, &shared_phones);
      granularity_dim = shared_phones.size();
      for (int i = 0; i < shared_phones.size(); i++)
        for (int j = 0; j < shared_phones[i].size(); j++)
          phone_map[shared_phones[i][j]] = i;
    } else if (granularity == "phones") {  // This is really not efficient
      const std::vector<int32> phones = trans_model.GetPhones();
      granularity_dim = phones.size();
      for (int i = 0; i < phones.size(); i++)
        phone_map[i] = i;
    }
    if (granularity == "sets" || granularity == "phones") {
      int32 num_tids = trans_model.NumTransitionIds();
      KALDI_LOG << "Num tid's:" << num_tids;
      for (int tid = 1; tid <= num_tids; tid++) {
        int32 phone = trans_model.TransitionIdToPhone(tid);
        tid_map[tid] = phone_map[phone];
      }
    }

    int32 tot_distance = 0.0;
    int32 tot_phones = 0, tot_phones_f = 0, tot_seqs = 0;
    SequentialNnetChainExampleReader example_reader(examples_rspecifier);
    for (; !example_reader.Done(); example_reader.Next()) {
      // examples.push_back(example_reader.Value());
      const NnetChainExample &eg = example_reader.Value();
      const Supervision &supervision = eg.outputs[0].supervision;
      CuMatrix<BaseFloat> nnet_output1g;
      Compute(nnet_opts, chain_opts, nnet1, eg, &nnet_output1g);
      Matrix<BaseFloat> nnet_output1(nnet_output1g, kTrans);
      CuMatrix<BaseFloat> nnet_output2g;
      Compute(nnet_opts, chain_opts, nnet2, eg, &nnet_output2g);
      Matrix<BaseFloat> nnet_output2(nnet_output2g, kTrans);

      if (!chain_opts.pdf_map_filename.empty())
        KALDI_ERR << "not supported";
      BaseFloat logp = 0;

      vector<vector<int32> > ali1;
      Align(supervision, nnet_output1, &ali1, &logp);

      vector<vector<int32> > ali2;
      Align(supervision, nnet_output2, &ali2, &logp);

      KALDI_ASSERT(ali1.size() == ali2.size());
      KALDI_ASSERT(ali1.size() == supervision.num_sequences);
      KALDI_ASSERT(trans_model.NumPdfs() == supervision.label_dim);
      int32 num_seqs = ali1.size();
      int32 T = supervision.frames_per_sequence;
      for (int32 seq = 0; seq < num_seqs; seq++) {
        tot_seqs++;
        std::vector<std::pair<int32, int32> > pairs1;
        AliToPhoneDurPairs(trans_model, tid_map, ali1[seq], &pairs1);
        std::vector<std::pair<int32, int32> > pairs2;
        AliToPhoneDurPairs(trans_model, tid_map, ali2[seq], &pairs2);
        tot_phones_f += pairs1.size();
        if (pairs1.size() != pairs2.size()) {
          KALDI_LOG << "Alignment length mismatch: " << pairs1.size() << " != " << pairs2.size();
          continue;
        }

        for (int32 t = 0; t < pairs1.size(); t++) {
          KALDI_ASSERT(t == 0 || pairs1[t].first != pairs1[t - 1].first);
          KALDI_ASSERT(t == 0 || pairs2[t].first != pairs2[t - 1].first);
          if (pairs1[t].first != pairs2[t].first) {
            KALDI_LOG << "Alignment mismatch";
            break;
          }
          tot_distance += (pairs1[t].second - pairs2[t].second) *
              (pairs1[t].second - pairs2[t].second);
          tot_phones++;
        }

      } // seq

    } // eg
    BaseFloat normalized_distance = tot_distance * 1.0 / tot_phones;
    KALDI_LOG << "normalized distance is " << normalized_distance
              << " over " << tot_phones << " phones.";
    std::cout << "Normalized distance is " << normalized_distance
              << " over " << tot_phones << " phones.\n";
    KALDI_LOG << "tot_phones_f: " << tot_phones_f;
    KALDI_LOG << "tot_seqs: " << tot_seqs;
#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}
