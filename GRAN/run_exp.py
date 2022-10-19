import os
import sys
import torch
import logging
import traceback
import numpy as np
from validation import validation_score
from pprint import pprint

from runner.gran_runner import *
from utils.logger import setup_logging
from utils.arg_helper import parse_arguments, get_config

torch.set_printoptions(profile="full")

# DISCLAIMER:
# Boilerplate parts of this code file were originally forked from
# https://github.com/lrjconan/GRAN


def main():
    args = parse_arguments()
    # args.test= True
    config = get_config(args.config_file, is_test=args.test)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    config.use_gpu = config.use_gpu and torch.cuda.is_available()

    # log info
    log_file = os.path.join(config.save_dir, "log_exp_{}.txt".format(config.run_id))
    logger = setup_logging(args.log_level, log_file)
    logger.info("Writing log file to {}".format(log_file))
    logger.info("Exp instance id = {}".format(config.run_id))
    logger.info("Exp comment = {}".format(args.comment))
    logger.info("Config =")
    print(">" * 80)
    pprint(config)
    print("<" * 80)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # The GPU id to use
    os.environ["CUDA_VISIBLE_DEVICES"] = str(3)
    # Run the experiment
    try:
        runner = eval(config.runner)(config)
        print("args.test", args.test)
        if not args.test:
            runner.train()
        else:
            runner.test()
    except:
        logger.error(traceback.format_exc())

    sys.exit(0)


if __name__ == "__main__":
    main()
