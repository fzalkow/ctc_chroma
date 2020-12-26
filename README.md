# Using Weakly Aligned Score–Audio Pairs to Train Deep Chroma Models for Cross-Modal Music Retrieval

## Background

This repository contains accompanying code for the following papers. If you use code from this repository, please consider citing them.

    [1]: Frank Zalkow and Meinard Müller: Using Weakly Aligned Score–Audio Pairs to Train Deep Chroma Models for Cross-Modal Music Retrieval. In Proceedings of the International Society for Music Information Retrieval Conference, Montréal, Canada, 2020.

    [2]: Frank Zalkow and Meinard Müller: CTC-Based Learning of Deep Chroma Features for Cross-Modal Music Retrieval. Currently under review.

There is an accompanying website for the paper.

https://www.audiolabs-erlangen.de/resources/MIR/2020-ISMIR-ctc-chroma

## Usage

You can install the code in this repository with pip:

    pip install ctc_chroma

There are two ways to use the models of this repository. The first way is to use a [Jupyter notebook](apply_model.ipynb). This notebook applies the model and visualizes its output. The second way is to use a script to batch process audio files in a folder. This script can be executed like this:

    python apply_model.py -m MODEL_ID -i INPUT -o OUTPUT

Here, `INPUT` is a directory with audio files, `OUTPUT` is a directory for the output files, and `MODEL_ID` specifies the model variant. There are several model variants contained in the repository. These variants are due to different versions of the training data (used in the papers [1] and [2], respectively), due to different training and validation splits, and due to different training procedures. The following table specifies all model identifiers of the repository.

| Model Identifier         | Used in Paper | Training Procedure             |
| ------------------------ | ------------- | ------------------------------ |
| v1_ctc_train1234valid5   | [1]           | CTC                            |
| v1_ctc_train123valid4    | [1]           | CTC                            |
| v1_ctc_train2345valid1   | [1]           | CTC                            |
| v1_ctc_train234valid5    | [1]           | CTC                            |
| v1_ctc_train3451valid2   | [1]           | CTC                            |
| v1_ctc_train345valid1    | [1]           | CTC                            |
| v1_ctc_train4512valid3   | [1]           | CTC                            |
| v1_ctc_train451valid2    | [1]           | CTC                            |
| v1_ctc_train5123valid4   | [1]           | CTC                            |
| v1_ctc_train512valid3    | [1]           | CTC                            |
| v2_ctc_train123valid4    | [2]           | CTC                            |
| v2_ctc_train234valid5    | [2]           | CTC                            |
| v2_ctc_train345valid1    | [2]           | CTC                            |
| v2_ctc_train451valid2    | [2]           | CTC                            |
| v2_ctc_train512valid3    | [2]           | CTC                            |
| v2_linear_train123valid4 | [2]           | Crossentropy, linear alignment |
| v2_linear_train234valid5 | [2]           | Crossentropy, linear alignment |
| v2_linear_train345valid1 | [2]           | Crossentropy, linear alignment |
| v2_linear_train451valid2 | [2]           | Crossentropy, linear alignment |
| v2_linear_train512valid3 | [2]           | Crossentropy, linear alignment |
| v2_strong_train123valid4 | [2]           | Crossentropy, strong alignment |
| v2_strong_train234valid5 | [2]           | Crossentropy, strong alignment |
| v2_strong_train345valid1 | [2]           | Crossentropy, strong alignment |
| v2_strong_train451valid2 | [2]           | Crossentropy, strong alignment |
| v2_strong_train512valid3 | [2]           | Crossentropy, strong alignment |

## Recordings

For making it easy to directly try out the code of this repository, we included two excerpts from public domain recordings, which we downloaded from [Musopen](https://musopen.org). The excerpts correspond to the musical sections that are used for the figures in the paper (Figure 3 and 4). However, different performances (not public domain) have been used to generate the figures in the paper. Below you find a small table with details for the excerpts.

| Filename                               | Composer  | Work                            | Performer                            | Description                  |
| -------------------------------------- | --------- | ------------------------------- | ------------------------------------ | ---------------------------- |
| Beethoven_Op067-01_DavidHighSchool.wav | Beethoven | Symphony no. 5, op. 67          | Davis High School Symphony Orchestra | First movement, first theme  |
| Beethoven_Op002-2-01_Pitman.wav        | Beethoven | Piano Sonata no. 2, op. 2 no. 2 | Paul Pitman                          | First movement, second theme |

## Acknowledgements

Frank Zalkow and Meinard Müller are supported by the German Research Foundation (DFG-MU 2686/11-1, MU 2686/12-1). We thank Daniel Stoller for fruitful discussions on the CTC loss, and Michael Krause for proof-reading the manuscript. We also thank Stefan Balke and Vlora Arifi-Müller as well as all students involved in the annotation work, especially Lena Krauß and Quirin Seilbeck. The International Audio Laboratories Erlangen are a joint institution of the Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU) and Fraunhofer Institute for Integrated Circuits IIS. The authors gratefully acknowledge the compute resources and support provided by the Erlangen Regional Computing Center (RRZE).
