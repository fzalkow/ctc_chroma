'''
File name: ctc_chroma/features.py
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

import os
import argparse
import glob
import multiprocessing as mp

import librosa
from tqdm import tqdm
import numpy as np


def get_hcqt_params():
    """Hack to always use the same parameters :)
    Function from https://github.com/rabitt/ismir2017-deepsalience"""
    bins_per_octave = 36
    n_octaves = 6
    harmonics = [0.5, 1, 2, 3, 4, 5]
    sr = 22050
    fmin = 32.7
    hop_length = 256
    return bins_per_octave, n_octaves, harmonics, sr, fmin, hop_length


def compute_hcqt(audio_fpath):
    """Compute the harmonic CQT from a given audio file
    Function from https://github.com/rabitt/ismir2017-deepsalience"""
    (bins_per_octave, n_octaves, harmonics,
     sr, f_min, hop_length) = get_hcqt_params()
    y, fs = librosa.load(audio_fpath, sr=sr)

    cqt_list = []
    shapes = []
    for h in harmonics:
        cqt = librosa.cqt(y, sr=fs, hop_length=hop_length, fmin=f_min*float(h), n_bins=bins_per_octave*n_octaves,
                          bins_per_octave=bins_per_octave)
        cqt_list.append(cqt)
        shapes.append(cqt.shape)

    shapes_equal = [s == shapes[0] for s in shapes]
    if not all(shapes_equal):
        min_time = np.min([s[1] for s in shapes])
        new_cqt_list = []
        for i in range(len(cqt_list)):
            new_cqt_list.append(cqt_list[i][:, :min_time])
        cqt_list = new_cqt_list

    log_hcqt = ((1.0 / 80.0) * librosa.core.amplitude_to_db(np.abs(np.array(cqt_list)), ref=np.max)) + 1.0

    return log_hcqt


def get_time_grid(n_time_frames):
    """Get the hcqt time grid
    Function from https://github.com/rabitt/ismir2017-deepsalience"""
    (_, _, _, sr, _, hop_length) = get_hcqt_params()
    time_grid = librosa.core.frames_to_time(range(n_time_frames), sr=sr, hop_length=hop_length)
    return time_grid


def get_freq_grid():
    """Get the hcqt frequency grid
    Function from https://github.com/rabitt/ismir2017-deepsalience"""
    (bins_per_octave, n_octaves, _, _, f_min, _) = get_hcqt_params()
    freq_grid = librosa.cqt_frequencies(bins_per_octave*n_octaves, f_min, bins_per_octave=bins_per_octave)
    return freq_grid


def compute_and_save_hcqt(fn_wav, fn_out, feature_rate):
    hcqt, times, freqs = compute_hcqt_median(fn_wav, feature_rate)

    # interpolate times
    eps = np.finfo(float).eps
    new_times = np.arange(times[0], times[-1] + eps, 1.0/feature_rate)  # add eps to include last time point

    hcqt_interpolated = median_bin_salience(hcqt, times, new_times, axis=-1)

    np.savez_compressed(fn_out, f_pitch=hcqt_interpolated, f_pitch_ax_time=new_times, f_pitch_ax_freq=freqs)


def compute_hcqt_median(fn_wav, feature_rate):
    hcqt = compute_hcqt(fn_wav)
    times = get_time_grid(len(hcqt[0][0]))
    freqs = get_freq_grid()

    # interpolate times
    eps = np.finfo(float).eps
    new_times = np.arange(times[0], times[-1] + eps, 1.0/feature_rate)  # add eps to include last time point

    return median_bin_salience(hcqt, times, new_times, axis=-1), new_times, freqs


def median_bin_salience(salience, old_times, new_times, axis=1):

    idx = []
    for i in range(len(new_times)):
        start_time = new_times[i]
        idx.append(np.searchsorted(old_times, start_time))
    idx.append(len(old_times))

    new_X = librosa.util.sync(salience, idx, aggregate=np.median, pad=False, axis=axis)

    return new_X


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--indir')
    parser.add_argument('-o', '--outdir')
    parser.add_argument('--rate', help='Frame Rate', type=int, default=25)
    parser.add_argument('--mode', help='DEVELOP or PRODUCTION', default='PRODUCTION')
    args = parser.parse_args()

    # python 00_extract_features_bittner/extract_features_input.py -i
    # /home/fzalkow/mountpoints/groupmm/Projects/BM/Subsets/barlow-morgenstern/03_BM-medium/data_WAV_WCM
    # -o data/BM_hcqt25 --time-binning median --rate 25

    indir = args.indir
    feature_rate = args.rate
    outdir = args.outdir
    assert os.path.exists(indir)
    assert args.mode in ['DEVELOP', 'PRODUCTION']
    os.makedirs(outdir, exist_ok=True)

    fn_audios = sorted(glob.glob(os.path.join(indir, '*.wav')))

    if args.mode == 'DEVELOP':
        base = os.path.splitext(os.path.basename(fn_audios[0]))[0]
        fn_new = os.path.join(outdir, base + '.npz')
        compute_and_save_hcqt(fn_audios[0], fn_new, feature_rate)

    if args.mode == 'PRODUCTION':
        n_workers = 16  # mp.cpu_count()
        pool = mp.Pool(n_workers)
        amount_of_jobs = len(fn_audios)
        i = 0

        with tqdm(total=amount_of_jobs, desc='Execute jobs', dynamic_ncols=True) as pbar:

            for fn_audio in fn_audios:

                base = os.path.splitext(os.path.basename(fn_audio))[0]
                fn_new = os.path.join(outdir, base + '.npz')
                if not os.path.exists(fn_new):
                    pool.apply_async(compute_and_save_hcqt, args=(fn_audio, fn_new, feature_rate),
                                     callback=lambda _: pbar.update(),
                                     error_callback=lambda x: tqdm.write(str(x)))
                else:
                    pbar.update()
            pool.close()  # no more jobs will be added
            pool.join()
