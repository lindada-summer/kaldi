// chainbin/nnet3-chain-check-egs.cc

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "hmm/transition-model.h"
#include "nnet3/nnet-chain-example.h"
#include "chain/chain-num-graph.h"
#include "chain/chain-numerator.h"
#include "chain/chain-full-numerator.h"
#include "chain/chain-training.h"
#include "chainbin/profiler2.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    using namespace fst;
    using namespace kaldi::chain;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;


    const char *usage =
        "Usage:  nnet3-chain-check-egs [options] <egs-rspecifier>\n";

//    bool compress = false;
//    int32 minibatch_size = 64;
    std::string use_gpu = "no";
    int32 n = 1;
    bool subtract = true, mini_test = false;

    ParseOptions po(usage);
//    po.Register("minibatch-size", &minibatch_size, "Target size of minibatches "
//                "when merging (see also --measure-output-frames)");
    po.Register("subtract", &subtract, "subtract logprobs or not");
    po.Register("use-gpu", &use_gpu,
                "yes|no|optional|wait, only has effect if compiled with CUDA");
    po.Register("n", &n, "number of egs to check");
    po.Register("mini-test", &mini_test, "run minitest or not");
    po.Read(argc, argv);


    if (mini_test) {
      StdVectorFst num3;
      ReadFstKaldi("num3.fst", &num3);
      //StdVectorFst den;
      //ReadFstKaldi("den.fst", &den);
      Supervision sup;
      sup.frames_per_sequence = 3;
      sup.label_dim = 10;
      sup.fst = num3;
      sup.e2e_fsts.push_back(num3);
      NumeratorGraph numg(sup, true);
      //num3g.PrintInfo(true);
      CuMatrix<BaseFloat>
            deriv1(3, 10, kSetZero),
            deriv2(3, 10, kSetZero);
      //std::cout << "fst:\n";
//    sup.Write(std::cout, false); std::cout << "\n";

      //DenominatorGraph deng(den, sup.label_dim);
      CuMatrix<BaseFloat> obs_mat(sup.frames_per_sequence, sup.label_dim);
      for (int t = 0; t < obs_mat.NumRows(); t++)
        for (int j = 0; j < obs_mat.NumCols(); j++) {
          int pdfid = j + 1;
          obs_mat(t, j) = Log((float)((t+1)*(pdfid+1) % 4 + 1));
        }
      obs_mat.Write(std::cout, false);
      ChainTrainingOptions copts;
      FullNumeratorComputation numc(copts, numg, obs_mat);
      BaseFloat full_num_logprob = numc.Forward();
      numc.Backward(1.0, &deriv1);
      std::cout << "full num log prob: " << full_num_logprob << "\n";
      deriv1.Write(std::cout, false);
      NumeratorComputation nc(sup, obs_mat);
      BaseFloat num_logprob = nc.Forward();
      nc.Backward(&deriv2);
      std::cout << "num log prob: " << num_logprob << "\n";
      deriv2.Write(std::cout, false);
    }



    if (po.NumArgs() != 1) {
      po.PrintUsage();
      exit(1);
    }

#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif

    std::string examples_rspecifier = po.GetArg(1);

    SequentialNnetChainExampleReader example_reader(examples_rspecifier);

    int64 num_read = 0, num_written = 0;
    for (; !example_reader.Done(); example_reader.Next(), num_read++) {
      if (num_read >= n)
        break;
      const NnetChainExample &eg = example_reader.Value();
      std::string key = example_reader.Key();
      std::cout << "Example " << key << ":\n\n"
                << "n-inputs: " << eg.inputs.size() << "\n"
                << "n-outputs: " << eg.outputs.size() << "\n"
                << "input[0].name: " << eg.inputs[0].name << "\n"
                << "input[0].feats-size: " << eg.inputs[0].features.NumRows() << ", " << eg.inputs[0].features.NumCols() << "\n"
                << "out[0].name: " << eg.outputs[0].name << "\n"
                << "out[0].supervision.num_sequences: " << eg.outputs[0].supervision.num_sequences << "\n"
                << "out[0].supervision.frames_per_sequence: " << eg.outputs[0].supervision.frames_per_sequence << "\n"
                << "out[0].supervision.label_dim: " << eg.outputs[0].supervision.label_dim << "\n"
                << "out[0].supervision.fst.NumStates: " << eg.outputs[0].supervision.fst.NumStates() << "\n"
                << "out[0].supervision.e2e_fsts.size: " << eg.outputs[0].supervision.e2e_fsts.size() << "\n"
                ;
      //for (int i = 0; i < eg.outputs[0].supervision.fsts.size(); i++)
      //  std::cout << "fsts[i].NumStates: " << eg.outputs[0].supervision.fsts[i].NumStates() << "\n";

      Profiler pf;

      pf.tic("numGraph");
      NumeratorGraph ng(eg.outputs[0].supervision, subtract);
      ng.PrintInfo(false);
      pf.tac();

      pf.tic("matPrep");
      int32 T = eg.outputs[0].supervision.frames_per_sequence,
            B = eg.outputs[0].supervision.num_sequences,
            N = eg.outputs[0].supervision.label_dim; //num pdfs
      CuMatrix<BaseFloat> random_nnet_output(T*B, N),
                          nnet_output_deriv1(T*B, N),
                          nnet_output_deriv2(T*B, N);
      random_nnet_output.SetRandUniform();
      random_nnet_output.ApplyLogSoftMaxPerRow(random_nnet_output);
      pf.tac();

      CuMatrix<BaseFloat> obs_mat(T, N);
      for (int t = 0; t < obs_mat.NumRows(); t++)
	for (int j = 0; j < obs_mat.NumCols(); j++) {
	  int pdfid = j + 1;
	  obs_mat(t, j) = Log((float)((t+1)*(pdfid+1) % 4 + 1));
	}
      random_nnet_output = obs_mat;
       /*
      pf.tic("on-CPU");
      NumeratorComputation numerator(eg.outputs[0].supervision, random_nnet_output);
      BaseFloat num_logprob_weighted = numerator.Forward();
      std::cout << "num logprob weighted: " << num_logprob_weighted << "\n";
      numerator.Backward(&nnet_output_deriv1);
      pf.tac();
      // */

      pf.tic("on-GPU-my");
      ChainTrainingOptions opts;
      FullNumeratorComputation cunum(opts, ng, random_nnet_output);
      BaseFloat cu_num_logprob_weighted = cunum.Forward();
      std::cout << "cu num logprob weighted: " << cu_num_logprob_weighted << "\n";
      bool ok = true;
      ok = cunum.Backward(eg.outputs[0].supervision.weight, &nnet_output_deriv2);
      std::cout << "ok: " << ok << "\n";
      pf.tac();

      std::cout << "Profiling results:\n" << pf.toString() << "\n";

      //WriteKaldiObject(nnet_output_deriv1, "deriv1.txt", false);
      //WriteKaldiObject(nnet_output_deriv2, "deriv2.txt", false);
      //for (int i = 0; i < nnet_output_deriv1.NumRows(); i++)
      //  for (int j = 0; j < nnet_output_deriv1.NumCols(); j++)
      //    if ( abs(nnet_output_deriv1(i, j) - nnet_output_deriv2(i, j)) > 0.01 )
      //      std::cout << "i: " << i << ", j: " << j << " ,deriv1: "
      //                << nnet_output_deriv1(i, j) << " ,deriv2: " << nnet_output_deriv2(i, j) << "\n";

      //AssertEqual(nnet_output_deriv1, nnet_output_deriv2, 0.001);
    }

#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif

    KALDI_LOG << "Checked " << num_read << " egs.";
    return (num_written != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}

