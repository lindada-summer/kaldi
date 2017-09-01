// nnet3/nnet-chain-training.cc

// Copyright      2015    Johns Hopkins University (author: Daniel Povey)
//                2016    Xiaohui Zhang

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

#include "nnet3/nnet-chain-training.h"
#include "nnet3/nnet-utils.h"

namespace kaldi {
namespace nnet3 {

NnetChainTrainer::NnetChainTrainer(const NnetChainTrainingOptions &opts,
                                   const fst::StdVectorFst &den_fst,
                                   Nnet *nnet):
    opts_(opts),
    den_graph_(den_fst, nnet->OutputDim("output")),
    nnet_(nnet),
    compiler_(*nnet, opts_.nnet_config.optimize_config,
              opts_.nnet_config.compiler_config),
    num_minibatches_processed_(0) {
  if (opts.nnet_config.zero_component_stats)
    ZeroComponentStats(nnet);
  KALDI_ASSERT(opts.nnet_config.momentum >= 0.0 &&
               opts.nnet_config.max_param_change >= 0.0);
  delta_nnet_ = nnet_->Copy();
  ScaleNnet(0.0, delta_nnet_);
  const int32 num_updatable = NumUpdatableComponents(*delta_nnet_);
  num_max_change_per_component_applied_.resize(num_updatable, 0);
  num_max_change_global_applied_ = 0;

  num_frames_processed_ = 0;  // for #pdf_tying
  if (!opts_.chain_config.pdf_map_filename.empty())
    den_graph_.MapPdfs(opts_.chain_config.pdf_map);

  if (opts.nnet_config.read_cache != "") {
    bool binary;
    try {
      Input ki(opts.nnet_config.read_cache, &binary);
      compiler_.ReadCache(ki.Stream(), binary);
      KALDI_LOG << "Read computation cache from " << opts.nnet_config.read_cache;
    } catch (...) {
      KALDI_WARN << "Could not open cached computation. "
                    "Probably this is the first training iteration.";
    }
  }
}


void NnetChainTrainer::Train(const NnetChainExample &chain_eg) {
  bool need_model_derivative = true;
  const NnetTrainerOptions &nnet_config = opts_.nnet_config;
  bool use_xent_regularization = (opts_.chain_config.xent_regularize != 0.0);
  ComputationRequest request;
  GetChainComputationRequest(*nnet_, chain_eg, need_model_derivative,
                             nnet_config.store_component_stats,
                             use_xent_regularization, need_model_derivative,
                             &request);
  const NnetComputation *computation = compiler_.Compile(request);

  NnetComputer computer(nnet_config.compute_config, *computation,
                        *nnet_, delta_nnet_);
  // give the inputs to the computer object.
  computer.AcceptInputs(*nnet_, chain_eg.inputs);
  computer.Run();

  this->ProcessOutputs(chain_eg, &computer);
  computer.Run();

  UpdateParamsWithMaxChange();
}

void NnetChainTrainer::DoPdfTying() const {
  // copy all the distances into a list so we can sort it by
  // distance. normalize the distances while doing this --> for debugging mostly

  //  std::vector<std::tuple<int32, int32, BaseFloat> > >
  //      distances(pdf_pair_distance_.size());
  std::vector<std::pair<std::pair<int32, int32>, BaseFloat> >
      distances(pdf_pair_distance_.size());
  int k = 0;
  for (auto it = pdf_pair_distance_.begin();
       it != pdf_pair_distance_.end(); ++it, k++) {
    BaseFloat distance = it->second;
    distance = sqrt(distance / num_frames_processed_);
    distances[k] = std::make_pair(it->first, distance);
  }
  struct {
    bool operator() (const std::pair<std::pair<int32, int32>, BaseFloat> &left,
                     const std::pair<std::pair<int32, int32>, BaseFloat> &right) const {
      return left.second < right.second;
    }
  } lowest_to_highest_distance;
  std::sort(distances.begin(), distances.end(), lowest_to_highest_distance);
  // print some of them for debugging...
  for (int32 i = 0; i < 100; i++) {
    std::cerr << "Distances[" << i << "]: " << distances[i].second
              << " -->  (" << distances[i].first.first << ", "
              << distances[i].first.second << ")\n";
  }

  // select opts.num_pdfs_to_tie of them to tie
  int32 num_pdfs = opts_.num_phone_sets * (opts_.num_phone_sets + 1);
  KALDI_ASSERT(num_pdfs == nnet_->OutputDim("output"));
  std::vector<int32> pdf_map(opts_.chain_config.pdf_map);
  if (pdf_map.size() == 0) {
    pdf_map.resize(num_pdfs);
    for (int32 i = 0; i < num_pdfs; i++)
      pdf_map[i] = i;
  }
  for (int32 i = 0; i < opts_.num_pdfs_to_tie; i++) {
    int32 n = distances[i].first.first;
    int32 m = distances[i].first.second;
    m = pdf_map[m];
    n = pdf_map[n];
    // make sure m < n
    if (n > m) std::swap(m, n);
    pdf_map[m] = n;  // tie (always map the larger pdf id to the smaller one)

    // m was mapped to n, so map to n, anything that's mapped to m
    for (int32 i = 0; i < num_pdfs; i++)
      if (pdf_map[i] == m)
        pdf_map[i] = n;
  }
  std::unordered_set<int32> unique_pdfs;
  for (int32 i = 0; i < num_pdfs; i++)
    unique_pdfs.insert(pdf_map[i]);
  KALDI_LOG << "Number of unique pdf ids after tying:" << unique_pdfs.size();
  // do a check
  for (int32 i = 0; i < num_pdfs; i++)
    if (pdf_map[pdf_map[i]] != pdf_map[i])
      std::cerr << "Error: " << i << std::endl;

  k = 0;
  std::vector<int32> map2(num_pdfs, -1);
  for (int32 i = 0; i < num_pdfs; i++) {
    if (map2[pdf_map[i]] == -1)
      map2[pdf_map[i]] = k++;
    pdf_map[i] = map2[pdf_map[i]];
  }

  std::ofstream of(opts_.write_pdf_map_filename);
  WriteIntegerVector(of, false, pdf_map);
  of << "<NumPdfs> " << k << "\n";
}


void NnetChainTrainer::ProcessOutputs(const NnetChainExample &eg,
                                      NnetComputer *computer) {
  // normally the eg will have just one output named 'output', but
  // we don't assume this.
  std::vector<NnetChainSupervision>::const_iterator iter = eg.outputs.begin(),
      end = eg.outputs.end();
  for (; iter != end; ++iter) {
    const NnetChainSupervision &sup = *iter;
    int32 node_index = nnet_->GetNodeIndex(sup.name);
    if (node_index < 0 ||
        !nnet_->IsOutputNode(node_index))
      KALDI_ERR << "Network has no output named " << sup.name;

    const CuMatrixBase<BaseFloat> &nnet_output = computer->GetOutput(sup.name);

    if (opts_.num_pdfs_to_tie != 0) { // collect stats for #pdf_tying
      KALDI_ASSERT(opts_.num_phone_sets > 0);
      KALDI_ASSERT(!opts_.write_pdf_map_filename.empty());
      num_frames_processed_ += nnet_output.NumRows();
      unordered_set<std::pair<int32, int32>, PairHasher<int32> > pdf_pair_done;

      for (int32 phone_set_index = 0; phone_set_index < opts_.num_phone_sets;
           phone_set_index++) {
        int32 stride = opts_.num_phone_sets + 1;
        // for each phone_set we have a subtree (EventMap) which has
        // a total of "stride" leaves. We will measure the distance between any
        // two of these leaves.
        for (int32 i = phone_set_index * stride;
             i < (phone_set_index + 1) * stride;
             i++) {
          for (int32 j = i + 1; j < (phone_set_index + 1) * stride; j++) {
            if (opts_.chain_config.pdf_map.size() != 0) {
              i = opts_.chain_config.pdf_map[i];
              j = opts_.chain_config.pdf_map[j];
            }
            // make sure i < j (this could happen if there is already some tying in effect)
            if (i > j) std::swap(i, j);
            auto pdf_pair = std::make_pair(i, j);
            if (pdf_pair_done.count(pdf_pair) || i == j)
              continue;
            pdf_pair_done.insert(pdf_pair);
            // for now, we define the distance to be euclidean distance
            // measure distance between columns i and j of nnet_output
            CuVector<BaseFloat> diff(nnet_output.NumRows(), kUndefined),
                vec_j(nnet_output.NumRows(), kUndefined);
            diff.CopyColFromMat(nnet_output, i);
            vec_j.CopyColFromMat(nnet_output, j);
            diff.AddVec(-1.0, vec_j, 1.0);
            pdf_pair_distance_[pdf_pair] += diff.Norm(2.0);
          }
        }
      }

      //      int i = 0;
      // for (auto it = pdf_pair_distance_.begin();
      //   it != pdf_pair_distance_.end() && i < 10; ++it, i++) {
      //std::cerr << it->second
      //          << " -->  (" << it->first.first << ", "
      //          << it->first.second << ")\n";
      //}

    }

    CuMatrix<BaseFloat> nnet_output_deriv(nnet_output.NumRows(),
                                          nnet_output.NumCols(),
                                          kUndefined);

    bool use_xent = (opts_.chain_config.xent_regularize != 0.0);
    std::string xent_name = sup.name + "-xent";  // typically "output-xent".
    CuMatrix<BaseFloat> xent_deriv;
    if (use_xent)
      xent_deriv.Resize(nnet_output.NumRows(), nnet_output.NumCols(),
                        kUndefined);

    BaseFloat tot_objf, tot_l2_term, tot_weight;

    ComputeChainObjfAndDeriv(opts_.chain_config, den_graph_,
                             sup.supervision, nnet_output,
                             &tot_objf, &tot_l2_term, &tot_weight,
                             &nnet_output_deriv,
                             (use_xent ? &xent_deriv : NULL));

    if (GetVerboseLevel() >= 2 && tot_objf == -sup.supervision.weight * sup.supervision.num_sequences *
      sup.supervision.frames_per_sequence * 10.0) {
      // Save nnet-output and derivs on disk
      KALDI_LOG << "Saving eg for debugging...";
      std::ofstream of("eg.txt");
      eg.Write(of, false);
      KALDI_LOG << "Saved eg.";
    }


    if (use_xent) {
      // this block computes the cross-entropy objective.
      const CuMatrixBase<BaseFloat> &xent_output = computer->GetOutput(
          xent_name);
      // at this point, xent_deriv is posteriors derived from the numerator
      // computation.  note, xent_objf has a factor of '.supervision.weight'
      BaseFloat xent_objf = TraceMatMat(xent_output, xent_deriv, kTrans);
      objf_info_[xent_name].UpdateStats(xent_name, opts_.nnet_config.print_interval,
                                        num_minibatches_processed_,
                                        tot_weight, xent_objf);
    }

    if (opts_.apply_deriv_weights && sup.deriv_weights.Dim() != 0) {
      CuVector<BaseFloat> cu_deriv_weights(sup.deriv_weights);
      nnet_output_deriv.MulRowsVec(cu_deriv_weights);
      if (use_xent)
        xent_deriv.MulRowsVec(cu_deriv_weights);
    }

    computer->AcceptInput(sup.name, &nnet_output_deriv);

    objf_info_[sup.name].UpdateStats(sup.name, opts_.nnet_config.print_interval,
                                     num_minibatches_processed_++,
                                     tot_weight, tot_objf, tot_l2_term);

    if (use_xent) {
      xent_deriv.Scale(opts_.chain_config.xent_regularize);
      computer->AcceptInput(xent_name, &xent_deriv);
    }
  }
}

void NnetChainTrainer::UpdateParamsWithMaxChange() {
  KALDI_ASSERT(delta_nnet_ != NULL);
  const NnetTrainerOptions &nnet_config = opts_.nnet_config;
  // computes scaling factors for per-component max-change
  const int32 num_updatable = NumUpdatableComponents(*delta_nnet_);
  Vector<BaseFloat> scale_factors = Vector<BaseFloat>(num_updatable);
  BaseFloat param_delta_squared = 0.0;
  int32 num_max_change_per_component_applied_per_minibatch = 0;
  BaseFloat min_scale = 1.0;
  std::string component_name_with_min_scale;
  BaseFloat max_change_with_min_scale;
  int32 i = 0;
  for (int32 c = 0; c < delta_nnet_->NumComponents(); c++) {
    Component *comp = delta_nnet_->GetComponent(c);
    if (comp->Properties() & kUpdatableComponent) {
      UpdatableComponent *uc = dynamic_cast<UpdatableComponent*>(comp);
      if (uc == NULL)
        KALDI_ERR << "Updatable component does not inherit from class "
                  << "UpdatableComponent; change this code.";
      BaseFloat max_param_change_per_comp = uc->MaxChange();
      KALDI_ASSERT(max_param_change_per_comp >= 0.0);
      BaseFloat dot_prod = uc->DotProduct(*uc);
      if (max_param_change_per_comp != 0.0 &&
          std::sqrt(dot_prod) > max_param_change_per_comp) {
        scale_factors(i) = max_param_change_per_comp / std::sqrt(dot_prod);
        num_max_change_per_component_applied_[i]++;
        num_max_change_per_component_applied_per_minibatch++;
        KALDI_VLOG(2) << "Parameters in " << delta_nnet_->GetComponentName(c)
                      << " change too big: " << std::sqrt(dot_prod) << " > "
                      << "max-change=" << max_param_change_per_comp
                      << ", scaling by " << scale_factors(i);
      } else {
        scale_factors(i) = 1.0;
      }
      if  (i == 0 || scale_factors(i) < min_scale) {
        min_scale =  scale_factors(i);
        component_name_with_min_scale = delta_nnet_->GetComponentName(c);
        max_change_with_min_scale = max_param_change_per_comp;
      }
      param_delta_squared += std::pow(scale_factors(i),
                                      static_cast<BaseFloat>(2.0)) * dot_prod;
      i++;
    }
  }
  KALDI_ASSERT(i == scale_factors.Dim());
  BaseFloat param_delta = std::sqrt(param_delta_squared);
  // computes the scale for global max-change (with momentum)
  BaseFloat scale = (1.0 - nnet_config.momentum);
  if (nnet_config.max_param_change != 0.0) {
    param_delta *= scale;
    if (param_delta > nnet_config.max_param_change) {
      if (param_delta - param_delta != 0.0) {
        KALDI_WARN << "Infinite parameter change, will not apply.";
        ScaleNnet(0.0, delta_nnet_);
      } else {
        scale *= nnet_config.max_param_change / param_delta;
        num_max_change_global_applied_++;
      }
    }
  }
  if ((nnet_config.max_param_change != 0.0 &&
      param_delta > nnet_config.max_param_change &&
      param_delta - param_delta == 0.0) || min_scale < 1.0) {
    std::ostringstream ostr;
    if (min_scale < 1.0)
      ostr << "Per-component max-change active on "
           << num_max_change_per_component_applied_per_minibatch
           << " / " << num_updatable << " Updatable Components."
           << " (smallest factor=" << min_scale << " on "
           << component_name_with_min_scale
           << " with max-change=" << max_change_with_min_scale <<"). ";
    if (param_delta > nnet_config.max_param_change)
      ostr << "Global max-change factor was "
           << nnet_config.max_param_change / param_delta
           << " with max-change=" << nnet_config.max_param_change << ".";
    KALDI_LOG << ostr.str();
  }
  // applies both of the max-change scalings all at once, component by component
  // and updates parameters
  scale_factors.Scale(scale);
  AddNnetComponents(*delta_nnet_, scale_factors, scale, nnet_);
  ScaleNnet(nnet_config.momentum, delta_nnet_);
}

bool NnetChainTrainer::PrintTotalStats() const {
  unordered_map<std::string, ObjectiveFunctionInfo, StringHasher>::const_iterator
      iter = objf_info_.begin(),
      end = objf_info_.end();
  bool ans = false;
  for (; iter != end; ++iter) {
    const std::string &name = iter->first;
    const ObjectiveFunctionInfo &info = iter->second;
    ans = info.PrintTotalStats(name) || ans;
  }
  PrintMaxChangeStats();
  return ans;
}

void NnetChainTrainer::PrintMaxChangeStats() const {
  KALDI_ASSERT(delta_nnet_ != NULL);
  int32 i = 0;
  for (int32 c = 0; c < delta_nnet_->NumComponents(); c++) {
    Component *comp = delta_nnet_->GetComponent(c);
    if (comp->Properties() & kUpdatableComponent) {
      UpdatableComponent *uc = dynamic_cast<UpdatableComponent*>(comp);
      if (uc == NULL)
        KALDI_ERR << "Updatable component does not inherit from class "
                  << "UpdatableComponent; change this code.";
      if (num_max_change_per_component_applied_[i] > 0)
        KALDI_LOG << "For " << delta_nnet_->GetComponentName(c)
                  << ", per-component max-change was enforced "
                  << (100.0 * num_max_change_per_component_applied_[i]) /
                     num_minibatches_processed_ << " \% of the time.";
      i++;
    }
  }
  if (num_max_change_global_applied_ > 0)
    KALDI_LOG << "The global max-change was enforced "
              << (100.0 * num_max_change_global_applied_) /
                 num_minibatches_processed_ << " \% of the time.";
}

NnetChainTrainer::~NnetChainTrainer() {
  if (opts_.num_pdfs_to_tie != 0)
    DoPdfTying();
  if (opts_.nnet_config.write_cache != "") {
    Output ko(opts_.nnet_config.write_cache, opts_.nnet_config.binary_write_cache);
    compiler_.WriteCache(ko.Stream(), opts_.nnet_config.binary_write_cache);
    KALDI_LOG << "Wrote computation cache to " << opts_.nnet_config.write_cache;
  }
  delete delta_nnet_;
}


} // namespace nnet3
} // namespace kaldi
