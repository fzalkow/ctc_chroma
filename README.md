# Using Weakly Aligned Score–Audio Pairs to Train Deep Chroma Models for Cross-Modal Music Retrieval

## Background

This repository contains accompanying code for the following paper.
If you use code from this repository, please consider citing the paper.

```
Frank Zalkow and Meinard Müller: Using Weakly Aligned Score–Audio Pairs to Train Deep Chroma Models for Cross-Modal Music Retrieval. In Proceedings of the International Society for Music Information Retrieval Conference, Montréal, Canada, 2020.
```

There is an accompanying website for the paper.

https://www.audiolabs-erlangen.de/resources/MIR/2020-ISMIR-ctc-chroma

## Usage

You can install the code in this repository with pip:

```
pip install ctc_chroma
```

There are two ways to use the models of this repository. The first way is to use a [Jupyter notebook](apply_model.ipynb). This notebook applies the model and visualizes its output. The second way is to use a script to batch process audio files in a folder. This script can be executed like this:

```
python apply_model.py -m MODEL_ID -i INPUT -o OUTPUT
```

Here, `INPUT` is a directory with audio files, `OUTPUT` is a directory for the output files, and `MODEL_ID` specifies the model variant. There are ten model variants contained in the repository, due to different training and validation splits. The identifiers for the variants used in the paper are `train123valid4`, `train234valid5`, `train345valid1`, `train451valid2`, and `train512valid3`. Furthermore, we provide models where we used more training data (without having a left-out test set in our dataset.) The identifiers for these additional models are `train1234valid5`, `train2345valid1`, `train3451valid2`, `train4512valid3`, and `train5123valid4`. It may not be fair to test the latter models with the dataset used in the paper. However, we recommend using them for audio files outside our dataset.

## Recordings

For making it easy to directly try out the code of this repository, we included two excerpts from public domain recordings, which we downloaded from [Musopen](https://musopen.org). The excerpts correspond to the musical sections that are used for the figures in the paper (Figure 3 and 4). However, different performances (not public domain) have been used to generate the figures in the paper. Below you find a small table with details for the excerpts.

| Filename                               | Composer  | Work                            | Performer                            | Description                  |
|----------------------------------------|-----------|---------------------------------|--------------------------------------|------------------------------|
| Beethoven_Op067-01_DavidHighSchool.wav | Beethoven | Symphony no. 5, op. 67          | Davis High School Symphony Orchestra | First movement, first theme  |
| Beethoven_Op002-2-01_Pitman.wav        | Beethoven | Piano Sonata no. 2, op. 2 no. 2 | Paul Pitman                          | First movement, second theme |


## Acknowledgements

Frank Zalkow and Meinard Müller are supported by the German Research Foundation (DFG-MU 2686/11-1, MU 2686/12-1). We thank Daniel Stoller for fruitful discussions on the CTC loss, and Michael Krause for proof-reading the manuscript. We also thank Stefan Balke and Vlora Arifi-Müller as well as all students involved in the annotation work, especially Lena Krauß and Quirin Seilbeck. The International Audio Laboratories Erlangen are a joint institution of the Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU) and Fraunhofer Institute for Integrated Circuits IIS. The authors gratefully acknowledge the compute resources and support provided by the Erlangen Regional Computing Center (RRZE).
