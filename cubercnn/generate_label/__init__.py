# Only import llm_generated_prior directly to avoid detectron2 dependency
try:
    from .llm_generated_prior import SUNRGBD, KITTI, ARKitScenes, nuScenes
    __all__ = ['SUNRGBD', 'KITTI', 'ARKitScenes', 'nuScenes']
except ImportError:
    __all__ = []