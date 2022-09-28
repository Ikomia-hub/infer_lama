# Copyright (C) 2021 Ikomia SAS
# Contact: https://www.ikomia.com
#
# This file is part of the IkomiaStudio software.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import copy
import os
import numpy as np
import cv2
import torch
import requests
from ikomia import core, dataprocess
from distutils.util import strtobool
from omegaconf import OmegaConf
from skimage import img_as_float
from urllib.parse import urlencode
from zipfile import ZipFile

from kornia.geometry.transform import resize
from infer_lama.saicinpainting.evaluation.utils import move_to_device
from infer_lama.saicinpainting.training.trainers import load_checkpoint
from infer_lama.saicinpainting.evaluation.data import pad_tensor_to_modulo
from infer_lama.saicinpainting.evaluation.refinement import refine_predict


# --------------------
# - Class to handle the process parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class InferLamaParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        self.cuda = torch.cuda.is_available()
        self.method = "default"
        self.iter = 15
        self.ini_res = 100
        # Place default value initialization here
        self.predict_config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                                "lama-config", "predict_config.yaml")
        self.checkpoint_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                                "big-lama", "models", "best.ckpt")
        self.train_config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                                "lama-config", "train_config.yaml")
        self.update = False

    def setParamMap(self, param_map):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        self.cuda = strtobool(param_map["cuda"])
        self.method = param_map["method"]
        self.iter = int(param_map["iter"])
        self.ini_res = int(param_map["ini_res"])
        self.predict_config_path = param_map["predict_config_path"]
        self.checkpoint_path = param_map["checkpoint_path"]
        self.train_config_path = param_map["train_config_file"]
        self.update = True

    def getParamMap(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = core.ParamMap()
        param_map["cuda"] = str(self.cuda)
        param_map["method"] = self.method
        param_map["iter"] = str(self.iter)
        param_map["ini_res"] = str(self.ini_res)
        param_map["predict_config_path"] = self.predict_config_path
        param_map["checkpoint_path"] = self.checkpoint_path
        param_map["train_config_file"] = self.train_config_path
        return param_map


# --------------------
# - Class which implements the process
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class InferLama(dataprocess.C2dImageTask):

    def __init__(self, name, param):
        dataprocess.C2dImageTask.__init__(self, name)

        # Detect if we have a GPU available
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Add input/output of the process here
        self.model = None
        # Config
        self.predict_config_path = None
        self.checkpoint_path = None
        self.train_config_path = None

        # Create parameters class
        if param is None:
            self.setParam(InferLamaParam())
        else:
            self.setParam(copy.deepcopy(param))

    def getProgressSteps(self):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 1

    def infer(self, src_image):

       # Check input image format and generate binary mask
        src_ini = src_image

        if src_image.dtype == 'uint8':
            imagef = img_as_float(src_image)
            graph_input = self.getInput(1)
            if graph_input.isDataAvailable():
                self.createGraphicsMask(imagef.shape[1], imagef.shape[0], graph_input)
                binimg = self.getGraphicsMask(0)
            else:
                raise Exception("No graphic input set.")
        else:
            raise Exception("Input image should be of type unint8 (range: 0 through 255 decimal).")

        # Convert mask to tensor
        if binimg.ndim == 3:
            binimg = np.transpose(binimg, (2, 0, 1))
        binimg = binimg.astype('float32') / 255
        binimg_3d = np.expand_dims(binimg, axis=0)
        binimg_3d = torch.as_tensor([binimg_3d])

        # Convert input image to tensor
        if src_image.ndim == 3:
            src_image = np.transpose(src_image, (2, 0, 1))
        src_image = src_image.astype('float32') / 255
        src_image = torch.as_tensor([src_image])

        param = self.getParam()

        # Resizing if image too large

        if param.ini_res < 100:
            h_orig, w_orig = src_ini.shape[0], src_ini.shape[1]
            img_limit_size = (h_orig * w_orig) * param.ini_res * 0.01
            resizing = True
            ratio = np.sqrt(img_limit_size / float(h_orig * w_orig))
            h_red, w_red = int(h_orig * ratio), int(w_orig * ratio)
            print(f"Resizing image from unpainting from {(h_orig, w_orig)} to {(h_red, w_red)}...")
            binimg_3d = resize(
                        binimg_3d, (h_red, w_red),
                        interpolation = 'bilinear', align_corners = False)
            src_image = resize(
                        src_image, (h_red, w_red),
                        interpolation = 'bilinear', align_corners = False)
        else:
            resizing = False

        # Pad to tensor
        binimg_3d = pad_tensor_to_modulo(binimg_3d, 8)
        src_image_pad = pad_tensor_to_modulo(src_image, 8)

        # Reference image size
        unpad_to_size = [torch.as_tensor([src_image.shape[2]]),
                        torch.as_tensor([src_image.shape[3]])]

        # Prepare the input into batch of one for inference
        batch = dict(image = src_image_pad, mask = binimg_3d, unpad_to_size = unpad_to_size)

        predict_config = OmegaConf.load(param.predict_config_path)

        # Config for refine inpainting
        if param.method == "refine":
            predict_config.refine = True
            predict_config.refiner.gpu_ids = "0,"
            predict_config.refiner.n_iters = int(param.iter)
            predict_config.refiner.px_budget = src_image.shape[2] * src_image.shape[3]

        # Inpainting
        if predict_config.get('refine', False):
            assert 'unpad_to_size' in batch, "Unpadded size is required for the refinement"
            batch = move_to_device(batch, self.device)
            batch = self.model(batch)
            cur_res = refine_predict(
                    batch, self.model, **predict_config.refiner,
                    devices = self.device)
            cur_res = cur_res[0].permute(1,2,0).detach().cpu().numpy()
            cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
            cur_res = cv2.resize(
                    cur_res,(src_ini.shape[1], src_ini.shape[0]),
                    interpolation = cv2.INTER_LINEAR)
        else:
            with torch.no_grad():
                batch = move_to_device(batch, self.device)
                batch = self.model(batch)                
                cur_res = batch[predict_config.out_key][0].permute(1, 2, 0).detach().cpu().numpy()
                orig_height, orig_width = unpad_to_size
                cur_res = cur_res[:orig_height, :orig_width]
                cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
                if resizing is True:
                    cur_res = cv2.resize(
                            cur_res, (src_ini.shape[1], src_ini.shape[0]),
                            interpolation = cv2.INTER_LINEAR)

        # To fix
        # Inpainted_img = self.applyGraphicsMask(src_ini, cur_res, 0)
        output = self.getOutput(0)
        output.setImage(cur_res)

    def run(self):
        # Core function of your process
        # Call beginTaskRun for initialization
        self.beginTaskRun()

        # Get input :
        image_in = self.getInput(0)

        # Get image from input/output (numpy array):
        src_image = image_in.getImage()

        param = self.getParam()

        #Downloading the model
        if not os.path.isfile(param.checkpoint_path):
            print("Downloading the model, this will take a few minutes.")
            base_url = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?'
            public_key = 'https://disk.yandex.ru/d/ouP6l8VJ0HpMZg'
            final_url = base_url + urlencode(dict(public_key=public_key))
            response = requests.get(final_url)
            download_url = response.json()['href']
            download_response = requests.get(download_url)
            to_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)))
            with open(os.path.join(to_folder, "big-lama_model.zip") , 'wb') as f:
                f.write(download_response.content)
            with ZipFile(os.path.join(to_folder, "big-lama_model.zip"), 'r') as f:
                f.extractall(path=to_folder)
            os.remove(os.path.join(to_folder, "big-lama_model.zip"))

        # Load model
        if param.update or self.model is None:
            self.device = torch.device("cuda") if param.cuda else torch.device("cpu")
            train_config = OmegaConf.load(param.train_config_path)
            self.model = load_checkpoint(
                        train_config, param.checkpoint_path,
                        strict = False, map_location=self.device)
            self.model.freeze()
            self.model.to(self.device)
            param.update = False
            print("Will run on {}".format(self.device.type))

        self.infer(src_image)

        # Step progress bar:
        self.emitStepProgress()

        # Call endTaskRun to finalize process
        self.endTaskRun()


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class InferLamaFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "infer_lama"
        self.info.shortDescription = "Inpainting using Fourier convolutions by Samsung Research."
        self.info.description = "This plugin propose inference for large (irregular) "\
                                "mask inpainting (Lama). The model implementation "\
                                "is based on fast Fourier Convolution network."\
                                "There are two inference methods available: default and refine."\
                                "The refine method requires a large amount of memory therefore"\
                                "the image resolution can be manually adjusted " \
                                "according to your available memory."

        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Inpainting"
        self.info.version = "1.0.0"
        self.info.iconPath = "icons/icon.png"
        self.info.authors = "Suvorov, Roman and Logacheva, Elizaveta and Mashikhin,"\
                            "Anton and Remizova, Anastasia and Ashukha,"\
                            "Arsenii and Silvestrov, Aleksei and Kong, Naejin and Goka,"\
                            "Harshith and Park, Kiwoong and Lempitsky, Victor."
        self.info.article = "Resolution-robust Large Mask Inpainting with Fourier Convolutions"
        self.info.journal = "arXiv preprint arXiv:2109.07161"
        self.info.year = 2021
        self.info.license = "Apache license 2.0"
        # URL of documentation
        self.info.documentationLink = "https://doi.org/10.48550/arXiv.2109.07161"
        # Code source repository
        self.info.repository = "https://github.com/geomagical/lama-with-refiner/tree/refinement"
        # Keywords used for search
        self.info.keywords = "LaMa, mask inpainting, Fourier convolution, Samsung AI"

    def create(self, param=None):
        # Create process object
        return InferLama(self.info.name, param)
