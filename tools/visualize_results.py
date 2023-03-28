from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint

import numpy as np
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import _init_paths
from config import cfg
from config import update_config
from core.loss import JointsMSELoss
from core.function import validate
from utils.utils import create_logger
from utils.transforms import flip_back

import dataset
import models


def parse_args():
    parser = argparse.ArgumentParser(description="Train keypoints network")
    # general
    parser.add_argument("--cfg", help="experiment configure file name", required=True, type=str)
    parser.add_argument("opts", help="Modify config options using the command-line",  default=None, nargs=argparse.REMAINDER)
    parser.add_argument("--modelDir", help="model directory", type=str, default="")
    parser.add_argument("--logDir", help="log directory", type=str, default="")
    parser.add_argument("--dataDir", help="data directory", type=str, default="")
    parser.add_argument("--prevModelDir", help="prev Model directory", type=str, default="")
    return parser.parse_args()


def main():
    args = parse_args()
    update_config(cfg, args)

    logger, final_output_dir, _tb_log_dir = create_logger(cfg, args.cfg, 'valid')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = eval(f"models.{cfg.MODEL.NAME}.get_pose_net")(cfg, is_train=False)

    if cfg.TEST.MODEL_FILE:
        logger.info("=> loading model from {}".format(cfg.TEST.MODEL_FILE))
        ckpt_state_dict = torch.load(cfg.TEST.MODEL_FILE)
        model.load_state_dict(ckpt_state_dict, strict=True)
    else:
        model_state_file = os.path.join(final_output_dir, "final_state.pth")
        logger.info("=> loading model from {}".format(model_state_file))
        model.load_state_dict(torch.load(model_state_file))

    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    val_dataset = eval(f"dataset.cfg.DATASET.DATASET")(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
        transforms.Compose([transforms.ToTensor(), normalize])
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=True
    )

    model.eval()

    with torch.no_grad():
        for i, (input_img, *_) in enumerate(val_loader):
            # compute output
            outputs = model(input_img)
            if isinstance(outputs, list):
                output = outputs[-1]
            else:
                output = outputs

            print(output)

            if cfg.TEST.FLIP_TEST:
                input_flipped = np.flip(input_img.cpu().numpy(), 3).copy()
                input_flipped = torch.from_numpy(input_flipped).cuda()
                outputs_flipped = model(input_flipped)

                if isinstance(outputs_flipped, list):
                    output_flipped = outputs_flipped[-1]
                else:
                    output_flipped = outputs_flipped

                output_flipped = flip_back(output_flipped.cpu().numpy(), val_dataset.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()

                output = (output + output_flipped) * 0.5

            print(output)


if __name__ == "__main__":
    main()
