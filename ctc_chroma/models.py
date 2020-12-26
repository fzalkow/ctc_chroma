'''
File name: ctc_chroma/models.py
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

from tensorflow.keras.layers import Input, Conv2D, Lambda, LeakyReLU, TimeDistributed, Dense, Concatenate
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K


IDS = [
    'v1_ctc_train1234valid5',
    'v1_ctc_train123valid4',
    'v1_ctc_train2345valid1',
    'v1_ctc_train234valid5',
    'v1_ctc_train3451valid2',
    'v1_ctc_train345valid1',
    'v1_ctc_train4512valid3',
    'v1_ctc_train451valid2',
    'v1_ctc_train5123valid4',
    'v1_ctc_train512valid3',
    'v2_ctc_train123valid4',
    'v2_ctc_train234valid5',
    'v2_ctc_train345valid1',
    'v2_ctc_train451valid2',
    'v2_ctc_train512valid3',
    'v2_linear_train123valid4',
    'v2_linear_train234valid5',
    'v2_linear_train345valid1',
    'v2_linear_train451valid2',
    'v2_linear_train512valid3',
    'v2_strong_train123valid4',
    'v2_strong_train234valid5',
    'v2_strong_train345valid1',
    'v2_strong_train451valid2',
    'v2_strong_train512valid3'
]


def get_model_noweigts():
    num_pitches = 216
    num_harmonics = 6

    input_data = Input(shape=(None, num_pitches, num_harmonics), name='input_1')
    y1 = Conv2D(64, (3, 3), padding='same', activation=LeakyReLU(), name='bendy1')(input_data)
    y2 = Conv2D(32, (3, 3), padding='same', activation=LeakyReLU(), name='bendy2')(y1)
    y3 = Conv2D(32, (3, 3), padding='same', activation=LeakyReLU(), name='smoothy1')(y2)
    y4 = Conv2D(32, (3, 3), padding='same', activation=LeakyReLU(), name='smoothy2')(y3)
    y5 = Conv2D(8, (3, 42), padding='same', activation=LeakyReLU(), name='distribute')(y4)
    y6 = Conv2D(1, (1, 1), padding='same', activation='sigmoid', name='squishy')(y5)
    y6a = Lambda(lambda batch: K.squeeze(batch, axis=-1))(y6)  # output shape (batch, time, pitch)

    chroma_pooling = Dense(12, activation='linear', use_bias=False, trainable=False)
    eps_pooling = Dense(1, activation='linear')

    y_chroma = TimeDistributed(chroma_pooling, name='chroma_pool')(y6a)
    y_eps = TimeDistributed(eps_pooling, name='eps_pool')(y6a)

    y7 = Concatenate(axis=-1, name='concat_chroma_eps')([y_chroma, y_eps])
    y8 = Lambda(lambda batch: K.softmax(batch, axis=-1), name='softmax')(y7)

    return Model(inputs=input_data, outputs=y8)


def get_model(model_id):
    if model_id not in IDS:
        raise RuntimeError(f'Model identifier "{model_id}" is not valid.')

    model = get_model_noweigts()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    path_model = os.path.join(dir_path, 'models', 'model_' + model_id)
    model.load_weights(path_model)
    return model
