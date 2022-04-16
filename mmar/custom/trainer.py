import os

import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut

import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, models
from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision.io import read_image, ImageReadMode
import torch.nn.functional as F

from nvflare.apis.dxo import from_shareable, DXO, DataKind, MetaKey
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReturnCode, ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.model import make_model_learnable, model_learnable_to_dxo
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.pt.pt_fed_utils import PTModelPersistenceFormatManager
from custom.pt_constants import PTConstants
from net import Net

import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from sklearn import metrics, model_selection, preprocessing
from PIL import Image
from albumentations import HorizontalFlip, Rotate, RandomBrightnessContrast, Flip, Compose, RandomResizedCrop
from typing import List, Optional, Dict, Generator, NamedTuple, Any, Tuple, Union, Mapping
import pandas as pd
import numpy as np

### LABELS
labels_col = ['Cardiomegaly','Effusion','Edema']
classes = 3 # number of findings
image_w = 256
image_h = 256

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

def augment(p=0.5):
    return Compose([
        RandomResizedCrop(image_h,image_w,scale=(0.7, 1.0), p=1.0),
        HorizontalFlip(p=0.5),
        RandomBrightnessContrast(0.5,0.5,p=0.5),
        Rotate(90, border_mode=0, p=0.5),
    ], p=p)

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

class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True

class Trainer(Executor):

    def __init__(self, lr=1e-3, epochs=50, train_task_name=AppConstants.TASK_TRAIN,
                 submit_model_task_name=AppConstants.TASK_SUBMIT_MODEL, exclude_vars=None):
        super(Trainer, self).__init__()

        self._lr = lr
        self._epochs = epochs
        self._train_task_name = train_task_name
        self._submit_model_task_name = submit_model_task_name
        self._exclude_vars = exclude_vars
        self.batch_size = 32
        self.transform = augment()

    def setup(self, fl_ctx):
        self.client_name = fl_ctx.get_identity_name()

        ### PATH
        site = ""
        if self.client_name == "site-c":
            site = "c"
        else:
            site = "n"

        # Training setup
        PATH_NAME = os.path.join(BASE_DIR, f'data_{site}')
        IMAGE_PATH = os.path.join(PATH_NAME, 'selected_xray', 'aj-sira')
        train = pd.read_csv(os.path.join(PATH_NAME, 'label', f'train_{site}.csv'))
        val = pd.read_csv(os.path.join(PATH_NAME, 'label', f'val_{site}.csv'))

        train_dataset = XRayDataset(train, IMAGE_PATH) # transform?
        val_dataset = XRayDataset(val, IMAGE_PATH)
        self._train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False,
                                        num_workers=0)
        self._val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False,
                                      num_workers=0)

        self.train_size = len(train)
        self.val_size = len(val)
        self.total_step = len(self._train_loader) * self._epochs
        self._n_iterations = len(self._train_loader)

        self.model = Net()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.loss = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self._lr)

        self._default_train_conf = {"train": {"model": type(self.model).__name__}}
        self.persistence_manager = PTModelPersistenceFormatManager(
            data=self.model.state_dict(), default_train_conf=self._default_train_conf)

    def local_train(self, fl_ctx, weights, abort_signal):
        self.model.load_state_dict(state_dict=weights)

        # Basic training
        min_val_loss = 10
        for epoch in range(self._epochs):
            # Train
            self.model.train()
            losses = []
            for i, (images, labels) in enumerate(self._train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                with torch.set_grad_enabled(True):
                    outputs = self.model(images)
                    outputs = outputs.pred
                    # print(outputs)
                    loss = self.loss(outputs, labels)

                    # Backward and optimize
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    losses.append(loss.item())
            self.logger.info(f"Local\n epochs ${epoch}\ntrain_loss: ${np.mean(losses)}")

            # Validate
            self.model.eval()
            losses = []
            for i, (images, labels) in enumerate(self._val_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                # forward
                with torch.set_grad_enabled(False):
                    outputs = self.model(images)
                    outputs = outputs.pred
                    loss = self.loss(outputs, labels)
                    losses.append(loss.item())
            val_loss = np.mean(losses)
            self.logger.info(f"Local\n epochs ${epoch}\nval_loss: ${val_loss}")
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                torch.save(self.model.state_dict(), "./best_model.pt")
                self.logger.info(f"Model saved as current val_loss is : {val_loss}")
            self.early_stopping(val_loss)
            if self.early_stopping.early_stop:
                self.logger.info(f"Early stopping at val_loss : {val_loss}")
                break

        tmp_dict = torch.load("./best_model.pt")
        self.model.load_state_dict(tmp_dict)

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        try:
            if task_name == self._train_task_name:
                self.setup(fl_ctx)
                # Get model weights
                try:
                    dxo = from_shareable(shareable)
                except:
                    self.log_error(fl_ctx, "Unable to extract dxo from shareable.")
                    return make_reply(ReturnCode.BAD_TASK_DATA)

                # Ensure data kind is weights.
                if not dxo.data_kind == DataKind.WEIGHTS:
                    self.log_error(fl_ctx, f"data_kind expected WEIGHTS but got {dxo.data_kind} instead.")
                    return make_reply(ReturnCode.BAD_TASK_DATA)

                # Convert weights to tensor. Run training
                torch_weights = {k: torch.as_tensor(v) for k, v in dxo.data.items()}
                self.early_stopping = EarlyStopping(patience=10)
                self.local_train(fl_ctx, torch_weights, abort_signal)

                # Check the abort_signal after training.
                # local_train returns early if abort_signal is triggered.
                if abort_signal.triggered:
                    return make_reply(ReturnCode.TASK_ABORTED)

                # Save the local model after training.
                self.save_local_model(fl_ctx)

                # Get the new state dict and send as weights
                new_weights = self.model.state_dict()
                new_weights = {k: v.cpu().numpy() for k, v in new_weights.items()}

                outgoing_dxo = DXO(data_kind=DataKind.WEIGHTS, data=new_weights,
                                   meta={MetaKey.NUM_STEPS_CURRENT_ROUND: self._n_iterations})
                return outgoing_dxo.to_shareable()
            elif task_name == self._submit_model_task_name:
                # Load local model
                ml = self.load_local_model(fl_ctx)

                # Get the model parameters and create dxo from it
                dxo = model_learnable_to_dxo(ml)
                return dxo.to_shareable()
            else:
                return make_reply(ReturnCode.TASK_UNKNOWN)
        except:
            self.log_exception(fl_ctx, f"Exception in simple trainer.")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

    def save_local_model(self, fl_ctx: FLContext):
        run_dir = fl_ctx.get_engine().get_workspace().get_run_dir(fl_ctx.get_prop(ReservedKey.RUN_NUM))
        models_dir = os.path.join(run_dir, PTConstants.PTModelsDir)
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        model_path = os.path.join(models_dir, PTConstants.PTLocalModelName)

        ml = make_model_learnable(self.model.state_dict(), {})
        self.persistence_manager.update(ml)
        torch.save(self.persistence_manager.to_persistence_dict(), model_path)

    def load_local_model(self, fl_ctx: FLContext):
        run_dir = fl_ctx.get_engine().get_workspace().get_run_dir(fl_ctx.get_prop(ReservedKey.RUN_NUM))
        models_dir = os.path.join(run_dir, PTConstants.PTModelsDir)
        if not os.path.exists(models_dir):
            return None
        model_path = os.path.join(models_dir, PTConstants.PTLocalModelName)

        self.persistence_manager = PTModelPersistenceFormatManager(data=torch.load(model_path),
                                                                   default_train_conf=self._default_train_conf)
        ml = self.persistence_manager.to_model_learnable(exclude_vars=self._exclude_vars)
        return ml