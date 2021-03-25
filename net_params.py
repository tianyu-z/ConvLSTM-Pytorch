from collections import OrderedDict
from ConvRNN import CGRU_cell, CLSTM_cell


# build model
# in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride=v[3], padding=v[4]
def get_params(inputshape):
    # input: inputshape is a list [H, W] of the dataset
    H, W = inputshape
    convlstm_encoder_params = [
        [
            OrderedDict({"conv1_leaky_1": [1, 16, 3, 1, 1]}),
            OrderedDict({"conv2_leaky_1": [64, 64, 3, 2, 1]}),
            OrderedDict({"conv3_leaky_1": [96, 96, 3, 2, 1]}),
        ],
        [
            CLSTM_cell(shape=(H, W), input_channels=16, filter_size=5, num_features=64),
            CLSTM_cell(
                shape=(int(H / 2), int(W / 2)),
                input_channels=64,
                filter_size=5,
                num_features=96,
            ),
            CLSTM_cell(
                shape=(int(H / 4), int(W / 4)),
                input_channels=96,
                filter_size=5,
                num_features=96,
            ),
        ],
    ]

    convlstm_decoder_params = [
        [
            OrderedDict({"deconv1_leaky_1": [96, 96, 4, 2, 1]}),
            OrderedDict({"deconv2_leaky_1": [96, 96, 4, 2, 1]}),
            OrderedDict(
                {"conv3_leaky_1": [64, 16, 3, 1, 1], "conv4_leaky_1": [16, 1, 1, 1, 0]}
            ),
        ],
        [
            CLSTM_cell(
                shape=(int(H / 4), int(H / 4)),
                input_channels=96,
                filter_size=5,
                num_features=96,
            ),
            CLSTM_cell(
                shape=(int(H / 2), int(H / 2)),
                input_channels=96,
                filter_size=5,
                num_features=96,
            ),
            CLSTM_cell(shape=(H, W), input_channels=96, filter_size=5, num_features=64),
        ],
    ]

    convgru_encoder_params = [
        [
            OrderedDict({"conv1_leaky_1": [1, 16, 3, 1, 1]}),
            OrderedDict({"conv2_leaky_1": [64, 64, 3, 2, 1]}),
            OrderedDict({"conv3_leaky_1": [96, 96, 3, 2, 1]}),
        ],
        [
            CGRU_cell(shape=(H, W), input_channels=16, filter_size=5, num_features=64),
            CGRU_cell(
                shape=(int(H / 2), int(H / 2)),
                input_channels=64,
                filter_size=5,
                num_features=96,
            ),
            CGRU_cell(
                shape=(int(H / 4), int(H / 4)),
                input_channels=96,
                filter_size=5,
                num_features=96,
            ),
        ],
    ]

    convgru_decoder_params = [
        [
            OrderedDict({"deconv1_leaky_1": [96, 96, 4, 2, 1]}),
            OrderedDict({"deconv2_leaky_1": [96, 96, 4, 2, 1]}),
            OrderedDict(
                {"conv3_leaky_1": [64, 16, 3, 1, 1], "conv4_leaky_1": [16, 1, 1, 1, 0]}
            ),
        ],
        [
            CGRU_cell(
                shape=(int(H / 4), int(H / 4)),
                input_channels=96,
                filter_size=5,
                num_features=96,
            ),
            CGRU_cell(
                shape=(int(H / 2), int(H / 2)),
                input_channels=96,
                filter_size=5,
                num_features=96,
            ),
            CGRU_cell(shape=(H, W), input_channels=96, filter_size=5, num_features=64),
        ],
    ]
    return [
        convlstm_encoder_params,
        convlstm_decoder_params,
        convgru_encoder_params,
        convgru_decoder_params,
    ]

