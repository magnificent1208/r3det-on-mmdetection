from ..builder import PIPELINES
from .transforms import Resize, RandomFlip, RandomCrop
import numpy as np


@PIPELINES.register_module()
class RResize(Resize):
    """
        Resize images & rotated bbox
        Inherit Resize pipeline class to handle rotated bboxes
    """

    def __init__(self,
                 img_scale=None,
                 multiscale_mode='range',
                 ratio_range=None):
        super(RResize, self).__init__(img_scale=img_scale,
                                      multiscale_mode=multiscale_mode,
                                      ratio_range=ratio_range,
                                      keep_ratio=True)

    def _resize_bboxes(self, results):
        for key in results.get('bbox_fields', []):
            bboxes = results[key]
            orig_shape = bboxes.shape
            bboxes = bboxes.reshape((-1, 5))
            w_scale, h_scale, _, _ = results['scale_factor']
            bboxes[:, 0] *= w_scale
            bboxes[:, 1] *= h_scale
            bboxes[:, 2:4] *= np.sqrt(w_scale * h_scale)
            results[key] = bboxes.reshape(orig_shape)


@PIPELINES.register_module()
class RRandomFlip(RandomFlip):
    """Flip the image & bbox & mask.

    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.

    Args:
        flip_ratio (float, optional): The flipping probability.
    """

    def bbox_flip(self, bboxes, img_shape, direction):
        """Flip bboxes horizontally or vertically.

        Args:
            bboxes(ndarray): shape (..., 5*k)
            img_shape(tuple): (height, width)
        """
        assert bboxes.shape[-1] % 5 == 0
        orig_shape = bboxes.shape
        bboxes = bboxes.reshape((-1, 5))
        flipped = bboxes.copy()
        if direction == 'horizontal':
            flipped[:, 0] = img_shape[1] - bboxes[:, 0] - 1
        elif direction == 'vertical':
            flipped[:, 1] = img_shape[0] - bboxes[:, 1] - 1
        else:
            raise ValueError(
                'Invalid flipping direction "{}"'.format(direction))
        rotated_flag = (bboxes[:, 4] != -np.pi / 2)
        flipped[rotated_flag, 4] = -np.pi / 2 - bboxes[rotated_flag, 4]
        flipped[rotated_flag, 2] = bboxes[rotated_flag, 3],
        flipped[rotated_flag, 3] = bboxes[rotated_flag, 2]
        return flipped.reshape(orig_shape)
