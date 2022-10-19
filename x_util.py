import json

with open('./models/ae_model_arch.json') as f:
    hparams = json.loads(f.read())
    for item in hparams:
        print(item, hparams[item])


# def construct_params_for_ae():
#     """
#     Constructs the hyperparameters for auto encoder.
#     Returns a dictionary.
#     Below is the reference.
#     hparams : :obj:`dict`
#             - 'model_type' (:obj:`int`): 'conv' | 'linear'
#             - 'model_class' (:obj:`str`): 'ae'
#             - 'y_pixels' (:obj:`int`)
#             - 'x_pixels' (:obj:`int`)
#             - 'n_input_channels' (:obj:`int`)
#             - 'n_ae_latents' (:obj:`int`)
#             - 'fit_sess_io_layers' (:obj:`bool`): fit session-specific input/output layers
#             - 'ae_encoding_x_dim' (:obj:`list`)
#             - 'ae_encoding_y_dim' (:obj:`list`)
#             - 'ae_encoding_n_channels' (:obj:`list`)
#             - 'ae_encoding_kernel_size' (:obj:`list`)
#             - 'ae_encoding_stride_size' (:obj:`list`)
#             - 'ae_encoding_x_padding' (:obj:`list`)
#             - 'ae_encoding_y_padding' (:obj:`list`)
#             - 'ae_encoding_layer_type' (:obj:`list`)
#             - 'ae_decoding_x_dim' (:obj:`list`)
#             - 'ae_decoding_y_dim' (:obj:`list`)
#             - 'ae_decoding_n_channels' (:obj:`list`)
#             - 'ae_decoding_kernel_size' (:obj:`list`)
#             - 'ae_decoding_stride_size' (:obj:`list`)
#             - 'ae_decoding_x_padding' (:obj:`list`)
#             - 'ae_decoding_y_padding' (:obj:`list`)
#             - 'ae_decoding_layer_type' (:obj:`list`)
#             - 'ae_decoding_starting_dim' (:obj:`list`)
#             - 'ae_decoding_last_FF_layer' (:obj:`bool`)
#     the following is the default setting.
#     """
#     res = {
#         'model_type': 'conv',  # (: obj:`int`): 'conv' | 'linear'
#         'model_class': 'ae',  # (: obj:`str`): 'ae'
#         'ae_input_dim': [1, 256, 256],
#         'y_pixels': 256,  # (: obj:`int`)
#         'x_pixels': 256,  # (: obj:`int`)
#         'n_input_channels': 1,  # (: obj:`int`)
#         'n_ae_latents': 12,  # (: obj:`int`)
#         'fit_sess_io_layers': False,  # (:obj:`bool`) fit session-specific input/output layers
#         'ae_encoding_x_dim': [128, 64, 32, 16, 4],  # (: obj:`list`)
#         'ae_encoding_y_dim': [128, 64, 32, 16, 4],  # (: obj:`list`)
#         'ae_encoding_n_channels': [32, 64, 128, 256, 512],  # (: obj:`list`)
#         'ae_encoding_kernel_size': [5, 5, 5, 5, 5],  # (: obj:`list`)
#         'ae_encoding_stride_size': [2, 2, 2, 2, 5],  # (: obj:`list`)
#         'ae_encoding_x_padding': [[1, 2], [1, 2], [1, 2], [1, 2], [2, 2]],  # (: obj:`list`)
#         'ae_encoding_y_padding': [[1, 2], [1, 2], [1, 2], [1, 2], [2, 2]],  # (: obj:`list`)
#         'ae_encoding_layer_type': ['conv', 'conv', 'conv', 'conv', 'conv'],  # (: obj:`list`)
#         'ae_decoding_x_dim': [16, 32, 64, 128, 256],  # (: obj:`list`)
#         'ae_decoding_y_dim': [16, 32, 64, 128, 256],  # (: obj:`list`)
#         'ae_decoding_n_channels': [256, 128, 64, 32, 1],  # (: obj:`list`)
#         'ae_decoding_kernel_size': [5, 5, 5, 5, 5],  # (: obj:`list`)
#         'ae_decoding_stride_size': [5, 2, 2, 2, 2],  # (: obj:`list`)
#         'ae_decoding_x_padding': [[2, 2], [1, 2], [1, 2], [1, 2], [1, 2]],  # (: obj:`list`)
#         'ae_decoding_y_padding': [[2, 2], [1, 2], [1, 2], [1, 2], [1, 2]],  # (: obj:`list`)
#         'ae_decoding_layer_type': ['convtranspose', 'convtranspose', 'convtranspose', 'convtranspose', 'convtranspose'],  # (: obj:`list`)
#         'ae_decoding_starting_dim': [512, 4, 4],  # (: obj:`list`)
#         'ae_decoding_last_FF_layer': 0  # (: obj:`bool`)
#         'ae_batch_norm': False,
#         'ae_batch_norm_momentum':
#     }
#     return res
