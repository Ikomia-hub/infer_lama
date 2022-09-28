#!/usr/bin/env python3

# Example command:
# ./bin/predict.py \
#       model.path=<path to checkpoint, prepared by make_checkpoint.py> \
#       indir=<path to input data> \
#       outdir=<where to store predicts>

from ikomia import core, dataprocess
import copy

# Your imports below
import base64, os
#from IPython.display import HTML, Image
#from base64 import b64decode
#import matplotlib.pyplot as plt
import numpy as np


# Loading from predict.py
import logging
import os
import sys
import traceback

from infer_lama.saicinpainting.evaluation.utils import move_to_device # OK
from infer_lama.saicinpainting.evaluation.refinement import refine_predict # OK

# Set up envrironment variables 
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'



import cv2
# Module for creation hierarchical configuration
import hydra
import numpy as np
import torch
# Lib for process meters/progress bar
import tqdm
# Lib for markup language 
import yaml
# YAML based hierarchical configuration system, handle config from multiple sources
from omegaconf import OmegaConf

# Default collate function used by the DataLoader class
# Checks what type of data your Dataset returns and tries to combine into a batch like (x_batch, y_batch)
from torch.utils.data._utils.collate import default_collate 
# Set up dataset config, return dataset
from infer_lama.saicinpainting.training.data.datasets import make_default_val_dataset 
# Load checkpoint return model
from infer_lama.saicinpainting.training.trainers import load_checkpoint 
# Logger warning
from infer_lama.saicinpainting.utils import register_debug_signal_handlers 



LOGGER = logging.getLogger(__name__)


@hydra.main(config_path='../configs/prediction', config_name='default.yaml')
def main(predict_config: OmegaConf):
    try:
        register_debug_signal_handlers()  # kill -10 <pid> will result in traceback dumped into log
        # Loading config file and model checkpoint 
        device = torch.device(predict_config.device)

        train_config_path = os.path.join(predict_config.model.path, 'config.yaml')
        with open(train_config_path, 'r') as f:
            train_config = OmegaConf.create(yaml.safe_load(f))
        
        train_config.training_model.predict_only = True
        train_config.visualizer.kind = 'noop'
        # train_config is a dictionary, loaded as object with OmegaConf

        out_ext = predict_config.get('out_ext', '.png')
        
        #checkpoint_path: podel path (.../best.ckpt)
        checkpoint_path = os.path.join(predict_config.model.path, 
                                       'models', 
                                       predict_config.model.checkpoint)
        model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location=self.device)
        model.freeze()

        if not predict_config.get('refine', False):
            model.to(device)

        if not predict_config.indir.endswith('/'):
            predict_config.indir += '/'

       
        # Mask images 
        dataset = make_default_val_dataset(predict_config.indir, **predict_config.dataset)
        #tqdm.trange generates progress bar
        for img_i in tqdm.trange(len(dataset)):
            mask_fname = dataset.mask_filenames[img_i]
            cur_out_fname = os.path.join(
                predict_config.outdir, 
                os.path.splitext(mask_fname[len(predict_config.indir):])[0] + out_ext
            )
            os.makedirs(os.path.dirname(cur_out_fname), exist_ok=True)
            # dataset[img_i] is a dict containing arrays dtype = float32 with the keys ['image', 'mask', 'unpad_to_size'].
            # default_collate function, just converts array of structures to structures of array: makes tensors?.
            batch = default_collate([dataset[img_i]])
            # Batch is a dictionary with tensors of (['image', 'mask', 'unpad_to_size'])
            if predict_config.get('refine', False):
                assert 'unpad_to_size' in batch, "Unpadded size is required for the refinement"
                # image unpadding is taken care of in the refiner, so that output image
                # is same size as the input image
                cur_res = refine_predict(batch, model, **predict_config.refiner)
                cur_res = cur_res[0].permute(1,2,0).detach().cpu().numpy()
            else:
                with torch.no_grad():
                    batch = move_to_device(batch, device)
                    batch['mask'] = (batch['mask'] > 0) * 1
                    batch = model(batch)                    
                    cur_res = batch[predict_config.out_key][0].permute(1, 2, 0).detach().cpu().numpy()
                    unpad_to_size = batch.get('unpad_to_size', None)
                    if unpad_to_size is not None:
                        orig_height, orig_width = unpad_to_size
                        cur_res = cur_res[:orig_height, :orig_width]

            cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
            cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
            cv2.imwrite(cur_out_fname, cur_res)

    except KeyboardInterrupt:
        LOGGER.warning('Interrupted by user')
    except Exception as ex:
        LOGGER.critical(f'Prediction failed due to {ex}:\n{traceback.format_exc()}')
        sys.exit(1)


if __name__ == '__main__':
    main()
