'''
File name: apply_model.py
Author: Frank Zalkow
Date: 2020
License: MIT

This file is part of the following repository:
  https://github.com/fzalkow/ctc_chroma

If you use code of this reposiory, please cite:
  Frank Zalkow and Meinard Müller:
  Using Weakly Aligned Score–Audio Pairs to Train Deep Chroma Models for Cross-Modal Music Retrieval.
  In Proceedings of the International Society for Music Information Retrieval Conference, Montréal, Canada, 2020.
'''

import glob
import os
import argparse

import tqdm
import numpy as np
import librosa

import ctc_chroma


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model-id', help='model identifier', required=True)
    parser.add_argument('-i', '--input', help='input directory', required=True)
    parser.add_argument('-o', '--output', help='output directory', required=True)
    args = parser.parse_args()

    model_id = args.model_id
    input = args.input
    output = args.output

    assert os.path.exists(input)
    os.makedirs(output, exist_ok=True)

    model = ctc_chroma.models.get_model(model_id)

    all_files = sorted(glob.glob(os.path.join(input, '*.wav')))

    for fn_wav in tqdm.tqdm(all_files):
        hcqt, times, freqs = ctc_chroma.features.compute_hcqt_median(fn_wav, feature_rate=25)
        hcqt_norm = librosa.util.normalize(hcqt.T, norm=2, fill=True, axis=1)
        probabilities_ctc = model.predict(hcqt_norm[np.newaxis, :, :, :])[0, :, :].T
        chroma_ctc = librosa.util.normalize(probabilities_ctc[:-1, :], norm=2, fill=True, axis=0)

        fn_npz = os.path.join(output, os.path.splitext(os.path.basename(fn_wav))[0])
        np.savez_compressed(fn_npz, chroma=chroma_ctc, time_ax=times)
