# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os

import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize
from torch import nn
from torchvision.io import read_image, ImageReadMode

from nvflare.apis.dxo import from_shareable, DataKind, DXO
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants
from custom.pt_constants import PTConstants
from custom.pylon.pylon import PylonConfig
from custom.pylon.utils.pretrain import *

import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from sklearn import metrics, model_selection, preprocessing
from PIL import Image
import pandas as pd
import numpy as np

labels_col = ['Cardiomegaly','Effusion','Edema']
classes = 3
image_w = 256
image_h = 256

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def dicom2array(path, voi_lut=True, fix_monochrome=True):
    """Convert DICOM file to numy array
    
    Args: 
        path (str): Path to the DICOM file to be converted
        voi_lut (bool): Whether or not VOI LUT is available
        fix_monochrome (bool): Whether or not to apply MONOCHROME fix
        
    Returns:
        Numpy array of the respective DICOM file
    """
    
    # Use the pydicom library to read the DICOM file
    dicom = pydicom.read_file(path)
    
    # VOI LUT (if available by DICOM device) is used to transform raw DICOM data to "human-friendly" view
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array
        
    # Depending on this value, X-ray may look inverted - fix that
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data
        
    # Normalize the image array
    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    
    return data

class XRayDataset(Dataset):
    def __init__(self, df, image_dir, transform=None, target_transform=None):
        self.img_files = df['Image'].tolist()
        self.img_labels = df[labels_col].values.tolist()
        self.transform = transform
        self.target_transform = target_transform
        self.image_dir = image_dir

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.img_files[idx])
        if '.dcm' in self.img_files[idx]:
          image = dicom2array(img_path)
          image = torch.tensor(image)/255
          image = image.unsqueeze(0)
        else:
          image = read_image(img_path, mode=ImageReadMode.GRAY)/255

        label = self.img_labels[idx]
        if self.transform:
            image = image[0].numpy()
            aug = self.transform(image=image)
            image = torch.from_numpy(aug["image"])
            image = image.unsqueeze(0)
        image = image.unsqueeze(0)
        image = F.interpolate(image, size=image_w)
        image = image[0]
        image = image.expand(3, -1, -1)
        return image, label

class Validator(Executor):

    def __init__(self, validate_task_name=AppConstants.TASK_VALIDATION):
        super(Validator, self).__init__()

        self._validate_task_name = validate_task_name

        # Setup the model
        net_conf = PylonConfig(
            n_in=1,
            n_out=classes,
            up_type='2layer',
            freeze='enc',
        )
        self.model = net_conf.make_model()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model.to(self.device)
        self.batch_size = 32
        self.loss = nn.BCEWithLogitsLoss()

    def setup(self, fl_ctx):
        self.client_name = fl_ctx.get_identity_name()
        site = ""
        if self.client_name == "site-c":
            site = "c"
        else:
            site = "n"

        # Training setup
        PATH_NAME = os.path.join(BASE_DIR, f'data_{site}')
        IMAGE_PATH = os.path.join(PATH_NAME, 'selected_xray', 'aj-sira')
        test = pd.read_csv(os.path.join(PATH_NAME, 'label', f'test_{site}.csv'))

        test_dataset = XRayDataset(test, IMAGE_PATH)

        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False,
                                      num_workers=0)

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        if task_name == self._validate_task_name:
            self.setup(fl_ctx)
            model_owner = "?"
            try:
                try:
                    dxo = from_shareable(shareable)
                except:
                    self.log_error(fl_ctx, "Error in extracting dxo from shareable.")
                    return make_reply(ReturnCode.BAD_TASK_DATA)

                # Ensure data_kind is weights.
                if not dxo.data_kind == DataKind.WEIGHTS:
                    self.log_exception(fl_ctx, f"DXO is of type {dxo.data_kind} but expected type WEIGHTS.")
                    return make_reply(ReturnCode.BAD_TASK_DATA)

                # Extract weights and ensure they are tensor.
                model_owner = shareable.get_header(AppConstants.MODEL_OWNER, "?")
                weights = {k: torch.as_tensor(v, device=self.device) for k, v in dxo.data.items()}

                # Get validation accuracy
                val_loss = self.do_validation(weights, abort_signal)
                if abort_signal.triggered:
                    return make_reply(ReturnCode.TASK_ABORTED)

                self.log_info(fl_ctx, f"Loss when validating {model_owner}'s model on"
                                      f" {fl_ctx.get_identity_name()}"f's data: {val_loss}')

                dxo = DXO(data_kind=DataKind.METRICS, data={'val_loss': val_loss})
                return dxo.to_shareable()
            except:
                self.log_exception(fl_ctx, f"Exception in validating model from {model_owner}")
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)
        else:
            return make_reply(ReturnCode.TASK_UNKNOWN)

    def do_validation(self, weights, abort_signal):
        self.model.load_state_dict(weights)

        self.model.eval()
        losses = []
        with torch.no_grad():
            for i, (images, labels) in enumerate(self.test_loader):
                if abort_signal.triggered: return 0
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                outputs = outputs.pred
                loss = self.loss(outputs, labels)
                losses.append(loss.item())
        return np.mean(losses)