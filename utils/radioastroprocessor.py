import numpy as np
import sys
import torch
from astropy.io import fits
from astropy.visualization import ZScaleInterval
import torchvision.transforms.functional as TF
import torch.nn as nn


_ZSCALE_INTERVAL = ZScaleInterval()

class RadioFoundationProcessor:
    """
    Preprocessing pipeline for radio astronomy images, ensuring telescope independence
    and compatibility with a foundation model.
    """

    def __init__(self, image_size=224):
        """
        Parameters
        ----------
        image_size : int
            Output spatial size (H=W=image_size) after optional resizing.
        """
        self.image_size = image_size

    def __call__(self, fits_path):
        # 1. Load FITS and handle NaNs/Infs
        try:
            with fits.open(fits_path, memmap=True) as hdul:
                data = hdul[0].data.astype(np.float32)
                data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception as e:
            msg = f"RadioFoundationProcessor: failed to read FITS '{fits_path}': {e}"
            print(msg, file=sys.stderr, flush=True)
            raise OSError(msg) from e

        tensor = self._pipeline(data) #run processing

        # 7. Resize to arbitrary model input size
        tensor = TF.resize(tensor, [self.image_size, self.image_size], interpolation=TF.InterpolationMode.BICUBIC)

        return tensor

    def process_numpy(self, data, resize=False):
        """
        Process a numpy array (single-channel radio image) through the same pipeline as __call__.
        """
        # 1. Clean up NaNs
        data = np.array(data, dtype=np.float32)
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        tensor = self._pipeline(data) #run processing

        # 7. Optional resize to arbitrary model input size
        if resize:
            tensor = TF.resize(tensor, [self.image_size, self.image_size], interpolation=TF.InterpolationMode.BICUBIC)

        return tensor

    def _pipeline(self, data):
        """
        Core processing pipeline:

        1. Assume NaNs/Infs have already been cleaned by the caller.
        2. Apply Astropy's ZScaleInterval per image, mapping the result to [0,1].
        3. Convert to a 3-channel tensor for downstream models.

        Notes:
            - Absolute flux calibration is not preserved; this is intended
              for morphology-focused tasks.
        """
        # Astropy zscale stretch to [0,1]
        data = self.zscale_image(data)

        # 3-channel tensor in [0,1]
        tensor = torch.from_numpy(data.astype(np.float32)).unsqueeze(0).repeat(3, 1, 1).contiguous().float()
        return tensor

    @staticmethod
    def zscale_image(image):
        """
        Astropy ZScaleInterval stretch clipped to [0, 1].

        Parameters
        ----------
        image : np.ndarray
            Input 2D array.

        Returns
        -------
        np.ndarray
            Float32 array scaled to [0, 1].
        """
        img = np.asarray(image, dtype=np.float32)

        # Work only on finite values
        finite_mask = np.isfinite(img)
        if not np.any(finite_mask):
            return np.zeros_like(img, dtype=np.float32)

        # Guard against pathological constant images before applying zscale.
        vals = img[finite_mask]
        if np.all(vals == vals.flat[0]):
            return np.zeros_like(img, dtype=np.float32)

        img = np.asarray(_ZSCALE_INTERVAL(img, clip=True), dtype=np.float32)
        img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
        return np.clip(img, 0.0, 1.0)

class RangeAdapter(nn.Module):
    def forward(self, x):
        # x is the processor’s output in [0,1]
        return x.mul(4.0).sub(2.0)

if __name__ == '__main__':
    # Defaults: noise-dominated, multi-telescope cutouts (EMU-like), using a simple percentile-based z-scale to [0,1]
    processor = RadioFoundationProcessor(
        image_size=224,
        debug_stats=False,
    )
    processed_numpy = processor.process_numpy(data, resize=False) #data is a numpy image OR
    processed_tensor = processor(fits_path) # path to fits file, output as a tensor
