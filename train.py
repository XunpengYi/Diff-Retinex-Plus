# flake8: noqa
import os.path as osp
import warnings
import archs
import data
import losses
import models
from train_pipline import train_pipeline

warnings.filterwarnings("ignore", category=UserWarning)

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir))
    train_pipeline(root_path)
