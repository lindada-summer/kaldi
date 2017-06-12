#!/usr/bin/env python
import argparse
import sys
import os
import re
import errno
import subprocess
try:
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    do_plot = True
except ImportError:
    warnings.warn(
        """This script requires matplotlib and numpy.
        Please install them to generate plots.""")
    do_plot = False


parser = argparse.ArgumentParser(description="""Convert input matrix to posteriorgram""")
parser.add_argument('--outdir', default='.')
parser.add_argument('--transcript', default='---')
parser.add_argument('--title-prefix', default='')
parser.add_argument('--phones')
args = parser.parse_args()


def make_postgram(mat, utt, transcript, phones, pdffile):
    T = mat.shape[0]
    f = T / 70.0 
    fig = plt.figure(figsize=(8*f, 6.25*f))
    plt.imshow(np.transpose(mat), vmin=mat.min(), vmax=mat.max(),
               interpolation='nearest', cmap='gray')
    plt.xlabel(transcript)
    if phones:
      plt.yticks( range(len(phones)), tuple(phones), fontsize=8 )
    plt.grid(True, color='w', alpha=0.5)
    fig.suptitle(args.title_prefix + ': ' + utt)
    plt.savefig(pdffile)



######################################
######################################
mat = []
i = 0
utt = ''
while True:
    line = sys.stdin.readline().strip()
    if not line:
        break
    if line.find('[') != -1:
        utt = line[:line.find('[')].strip()
        continue
    parts = line.split()
    finished = False
    if parts[-1] == ']':
        parts = parts[:-1]
        finished = True
    row = [float(p) for p in parts]
    mat += [row]
    if finished:
        break

print('Generating postgram for utt: {}'.format(utt))

nneto = np.array(mat)
phones = None
if args.phones:
  phones = args.phones.split(',')[:-1]
  if len(phones) != nneto.shape[1]:
    print('Phone labels dim mismatch nnet output columns')
    sys.exit(1)

make_postgram(nneto, utt, args.transcript, phones, args.outdir+'/'+utt+'.pdf')

