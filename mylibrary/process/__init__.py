from ..nets import FeatureExtractor
from ..utils.loader_util import get_pixel_params_mask

def gen_extractor(src, is_calibrate, weight_path, num_cls):
    (pixel_mean, pixel_std) = get_pixel_params_mask(sources=src, vid_stride=3, count=5, threshold=(235, 235, 235)) if is_calibrate else ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    extractor = FeatureExtractor(
        model_name='osnet_ain_x1_0',
        model_path=weight_path,
        verbose=False,
        num_classes=num_cls,
        pixel_norm=True,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
        device='cuda'
    )
    return extractor