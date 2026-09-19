import torch
from models.convlstm import ConvLSTMModel
from models.cnn_lstm import CNNLSTMModel
from models.transformer import SpatioTemporalTransformer
from models.lstm_baseline import LSTMBaseline


def get_model(config):

    model_name = config["model"]["name"]

    if model_name == "convlstm":
        return ConvLSTMModel(config)

    elif model_name == "cnn_lstm":
        return CNNLSTMModel(config)

    elif model_name == "transformer":
        return SpatioTemporalTransformer(config)

    elif model_name == "lstm":
        return LSTMBaseline(config)

    else:
        raise ValueError(f"Unknown model: {model_name}")