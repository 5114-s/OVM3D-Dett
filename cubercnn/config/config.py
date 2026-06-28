# Copyright (c) Meta Platforms, Inc. and affiliates
from detectron2.config import CfgNode as CN

def get_cfg_defaults(cfg):

    # A list of category names which will be used
    cfg.DATASETS.CATEGORY_NAMES = []

    # The category names which will be treated as ignore
    # e.g., not counting as background during training
    # or as false positives during evaluation.
    cfg.DATASETS.IGNORE_NAMES = []

    # Should the datasets appear with the same probabilty
    # in batches (e.g., the imbalance from small and large
    # datasets will be accounted for during sampling)
    cfg.DATALOADER.BALANCE_DATASETS = False

    # The thresholds for when to treat a known box
    # as ignore based on too heavy of truncation or 
    # too low of visibility in the image. This affects
    # both training and evaluation ignores.
    cfg.DATASETS.TRUNCATION_THRES = 0.99
    cfg.DATASETS.VISIBILITY_THRES = 0.01
    cfg.DATASETS.MIN_HEIGHT_THRES = 0.00
    cfg.DATASETS.MAX_DEPTH = 1e8

    # Whether modal 2D boxes should be loaded, 
    # or if the full 3D projected boxes should be used.
    cfg.DATASETS.MODAL_2D_BOXES = False

    # Whether truncated 2D boxes should be loaded, 
    # or if the 3D full projected boxes should be used.
    cfg.DATASETS.TRUNC_2D_BOXES = True

    # Threshold used for matching and filtering boxes
    # inside of ignore regions, within the RPN and ROIHeads
    cfg.MODEL.RPN.IGNORE_THRESHOLD = 0.5

    # Configuration for cube head
    cfg.MODEL.ROI_CUBE_HEAD = CN()
    cfg.MODEL.ROI_CUBE_HEAD.NAME = "CubeHead"
    cfg.MODEL.ROI_CUBE_HEAD.POOLER_RESOLUTION = 7
    cfg.MODEL.ROI_CUBE_HEAD.POOLER_SAMPLING_RATIO = 0
    cfg.MODEL.ROI_CUBE_HEAD.POOLER_TYPE = "ROIAlignV2"

    # Settings for the cube head features
    cfg.MODEL.ROI_CUBE_HEAD.NUM_CONV = 0
    cfg.MODEL.ROI_CUBE_HEAD.CONV_DIM = 256
    cfg.MODEL.ROI_CUBE_HEAD.NUM_FC = 2
    cfg.MODEL.ROI_CUBE_HEAD.FC_DIM = 1024
    
    # the style to predict Z with currently supported
    # options --> ['direct', 'sigmoid', 'log', 'clusters']
    cfg.MODEL.ROI_CUBE_HEAD.Z_TYPE = "direct"

    # the style to predict pose with currently supported
    # options --> ['6d', 'euler', 'quaternion']
    cfg.MODEL.ROI_CUBE_HEAD.POSE_TYPE = "6d"

    # Whether to scale all 3D losses by inverse depth
    cfg.MODEL.ROI_CUBE_HEAD.INVERSE_Z_WEIGHT = False

    # Virtual depth puts all predictions of depth into
    # a shared virtual space with a shared focal length. 
    cfg.MODEL.ROI_CUBE_HEAD.VIRTUAL_DEPTH = True
    cfg.MODEL.ROI_CUBE_HEAD.VIRTUAL_FOCAL = 512.0

    # If true, then all losses are computed using the 8 corners
    # such that they are all in a shared scale space. 
    # E.g., their scale correlates with their impact on 3D IoU.
    # This way no manual weights need to be set.
    cfg.MODEL.ROI_CUBE_HEAD.DISENTANGLED_LOSS = True

    # When > 1, the outputs of the 3D head will be based on
    # a 2D scale clustering, based on 2D proposal height/width.
    # This parameter describes the number of bins to cluster.
    cfg.MODEL.ROI_CUBE_HEAD.CLUSTER_BINS = 1

    # Whether batch norm is enabled during training. 
    # If false, all BN weights will be frozen. 
    cfg.MODEL.USE_BN = True

    # Whether to predict the pose in allocentric space. 
    # The allocentric space may correlate better with 2D 
    # images compared to egocentric poses. 
    cfg.MODEL.ROI_CUBE_HEAD.ALLOCENTRIC_POSE = True

    # Whether to use chamfer distance for disentangled losses
    # of pose. This avoids periodic issues of rotation but 
    # may prevent the pose "direction" from being interpretable.
    cfg.MODEL.ROI_CUBE_HEAD.CHAMFER_POSE = True

    # Should the prediction heads share FC features or not. 
    # These include groups of uv, z, whl, pose.
    cfg.MODEL.ROI_CUBE_HEAD.SHARED_FC = True

    # Check for stable gradients. When inf is detected, skip the update. 
    # This prevents an occasional bad sample from exploding the model. 
    # The threshold below is the allows percent of bad samples. 
    # 0.0 is off, and 0.01 is recommended for minor robustness to exploding.
    cfg.MODEL.STABILIZE = 0.01
    
    # Whether or not to use the dimension priors
    cfg.MODEL.ROI_CUBE_HEAD.DIMS_PRIORS_ENABLED = True

    # How prior dimensions should be computed? 
    # The supported modes are ["exp", "sigmoid"]
    # where exp is unbounded and sigmoid is bounded
    # between +- 3 standard deviations from the mean.
    cfg.MODEL.ROI_CUBE_HEAD.DIMS_PRIORS_FUNC = 'exp'

    # weight for confidence loss. 0 is off.
    cfg.MODEL.ROI_CUBE_HEAD.USE_CONFIDENCE = 1.0

    # Loss weights for XY, Z, Dims, Pose
    cfg.MODEL.ROI_CUBE_HEAD.LOSS_W_3D = 1.0
    cfg.MODEL.ROI_CUBE_HEAD.LOSS_W_XY = 1.0
    cfg.MODEL.ROI_CUBE_HEAD.LOSS_W_Z = 1.0
    cfg.MODEL.ROI_CUBE_HEAD.LOSS_W_DIMS = 1.0
    cfg.MODEL.ROI_CUBE_HEAD.LOSS_W_POSE = 1.0
    cfg.MODEL.ROI_CUBE_HEAD.USE_PSEUDO_WEIGHT = False
    cfg.MODEL.ROI_CUBE_HEAD.USE_FACTORIZED_PSEUDO_WEIGHT = False

    cfg.MODEL.DLA = CN()

    # Supported types for DLA backbones are...
    # dla34, dla46_c, dla46x_c, dla60x_c, dla60, dla60x, dla102x, dla102x2, dla169
    cfg.MODEL.DLA.TYPE = 'dla34'

    # Only available for dla34, dla60, dla102
    cfg.MODEL.DLA.TRICKS = False

    # A joint loss for the disentangled loss.
    # All predictions are computed using a corner
    # or chamfers loss depending on chamfer_pose!
    # Recommened to keep this weight small: [0.05, 0.5]
    cfg.MODEL.ROI_CUBE_HEAD.LOSS_W_JOINT = 1.0

    # sgd, adam, adam+amsgrad, adamw, adamw+amsgrad
    cfg.SOLVER.TYPE = 'sgd'

    cfg.MODEL.RESNETS.TORCHVISION = True
    cfg.TEST.DETECTIONS_PER_IMAGE = 100

    cfg.TEST.VISIBILITY_THRES = 1/2.0
    cfg.TEST.TRUNCATION_THRES = 1/2.0

    cfg.INPUT.RANDOM_FLIP = "horizontal"

    # Optional metric depth input. When enabled, DatasetMapper3D loads a cached
    # depth map and the 3D RoI head may use it as a point-map feature.
    cfg.INPUT.USE_DEPTH = False
    cfg.INPUT.DEPTH_ROOT = ""
    cfg.INPUT.DEPTH_ALLOW_SENSOR_FALLBACK = True
    cfg.INPUT.USE_PSEUDO_MASK = False
    cfg.INPUT.PSEUDO_MASK_ROOT = ""
    cfg.INPUT.PSEUDO_MASK_SIZE = 28
    cfg.INPUT.PSEUDO_MASK_MATCH_IOU_THRESHOLD = 0.05
    cfg.INPUT.PSEUDO_MASK_ALLOW_REUSE = False
    cfg.INPUT.USE_GROUND_MASK = False
    cfg.INPUT.GROUND_MASK_ROOT = ""

    # When True, we will use localization uncertainty
    # as the new IoUness score in the RPN.
    cfg.MODEL.RPN.OBJECTNESS_UNCERTAINTY = 'IoUness'

    # If > 0.0 this is the scaling factor that will be applied to
    # an RoI 2D box before doing any pooling to give more context. 
    # Ex. 1.5 makes width and height 50% larger. 
    cfg.MODEL.ROI_CUBE_HEAD.SCALE_ROI_BOXES = 0.0

    # OVMono3D/DetAny3D-inspired depth adapter for the 3D RoI feature. The
    # adapter is zero-initialized so enabling it starts from the original model.
    cfg.MODEL.ROI_CUBE_HEAD.USE_DEPTH_ROI = False
    cfg.MODEL.ROI_CUBE_HEAD.DEPTH_ADAPTER_SCALE = 1.0

    # Training-only depth consistency regularizer. This uses cached metric depth
    # to regularize predicted object depth without adding a test-time dependency.
    cfg.MODEL.ROI_CUBE_HEAD.USE_DEPTH_CONSISTENCY_LOSS = False
    cfg.MODEL.ROI_CUBE_HEAD.LOSS_W_DEPTH_CONSISTENCY = 0.05
    cfg.MODEL.ROI_CUBE_HEAD.DEPTH_CONSISTENCY_MIN_PIXELS = 16
    cfg.MODEL.ROI_CUBE_HEAD.DEPTH_CONSISTENCY_CENTER_CROP = 0.75
    cfg.MODEL.ROI_CUBE_HEAD.DEPTH_CONSISTENCY_MODE = "center"
    cfg.MODEL.ROI_CUBE_HEAD.DEPTH_CONSISTENCY_PERCENTILE = 0.35

    # MonoDGP-style Region Segmentation Head (RSH). It distills cached
    # SAM/SAM2 masks into a lightweight RoI foreground map, then uses a
    # zero-initialized adapter to inject foreground-aware features.
    cfg.MODEL.ROI_CUBE_HEAD.USE_REGION_SEGMENTATION_HEAD = False
    cfg.MODEL.ROI_CUBE_HEAD.RSH_HIDDEN_DIM = 128
    cfg.MODEL.ROI_CUBE_HEAD.RSH_MASK_SIZE = 28
    cfg.MODEL.ROI_CUBE_HEAD.RSH_FEATURE_SCALE = 1.0
    cfg.MODEL.ROI_CUBE_HEAD.RSH_DETACH_MASK_FEATURE = True
    cfg.MODEL.ROI_CUBE_HEAD.LOSS_W_REGION_SEGMENTATION = 0.10
    cfg.MODEL.ROI_CUBE_HEAD.RSH_USE_DEPTH_GUIDANCE = True
    cfg.MODEL.ROI_CUBE_HEAD.RSH_DEPTH_MASK_THRESHOLD = 0.30
    cfg.MODEL.ROI_CUBE_HEAD.RSH_USE_PSEUDO_WEIGHT = False

    # DetAny3D-style zero-initialized residual prediction branches. These do not
    # change the initial outputs but can learn small corrections during training.
    cfg.MODEL.ROI_CUBE_HEAD.USE_ZERO_INIT_RESIDUAL = False
    cfg.MODEL.ROI_CUBE_HEAD.RESIDUAL_SCALE_XY = 1.0
    cfg.MODEL.ROI_CUBE_HEAD.RESIDUAL_SCALE_Z = 1.0
    cfg.MODEL.ROI_CUBE_HEAD.RESIDUAL_SCALE_DIMS = 1.0
    cfg.MODEL.ROI_CUBE_HEAD.RESIDUAL_SCALE_POSE = 1.0

    # Multi-modal latent-box interpreter.
    cfg.MODEL.ROI_CUBE_HEAD.USE_GEOMETRY_INTERPRETER = False
    cfg.MODEL.ROI_CUBE_HEAD.GEOMETRY_NUM_HYPOTHESES = 8
    cfg.MODEL.ROI_CUBE_HEAD.GEOMETRY_HIDDEN_DIM = 256
    cfg.MODEL.ROI_CUBE_HEAD.GEOMETRY_NUM_LAYERS = 3
    cfg.MODEL.ROI_CUBE_HEAD.GEOMETRY_NUM_HEADS = 8
    cfg.MODEL.ROI_CUBE_HEAD.GEOMETRY_RESIDUAL_SCALE = 0.25
    cfg.MODEL.ROI_CUBE_HEAD.LOSS_W_MULTI_HYPOTHESIS = 0.20
    cfg.MODEL.ROI_CUBE_HEAD.GEOMETRY_SELECTION_TEMPERATURE = 0.20
    cfg.MODEL.ROI_CUBE_HEAD.GEOMETRY_ORACLE_WEIGHT = 0.25
    cfg.MODEL.ROI_CUBE_HEAD.GEOMETRY_PROJECTION_WEIGHT = 1.0
    cfg.MODEL.ROI_CUBE_HEAD.GEOMETRY_SILHOUETTE_WEIGHT = 2.0
    cfg.MODEL.ROI_CUBE_HEAD.GEOMETRY_DEPTH_WEIGHT = 2.0
    cfg.MODEL.ROI_CUBE_HEAD.GEOMETRY_POINT_WEIGHT = 1.5
    cfg.MODEL.ROI_CUBE_HEAD.GEOMETRY_GROUND_WEIGHT = 0.25
    cfg.MODEL.ROI_CUBE_HEAD.LOSS_W_GEOMETRY_CLOSURE = 0.15
    cfg.MODEL.ROI_CUBE_HEAD.GEOMETRY_APPLY_TO_PREDICTION = False
    cfg.MODEL.ROI_CUBE_HEAD.GEOMETRY_APPLY_IN_INFERENCE = False
    cfg.MODEL.ROI_CUBE_HEAD.GEOMETRY_MIN_DIMENSION = 0.03
    cfg.MODEL.ROI_CUBE_HEAD.GEOMETRY_MAX_DIMENSION = 8.0
    cfg.MODEL.ROI_CUBE_HEAD.SHAPE_MEMORY_CAPACITY = 2048
    cfg.MODEL.ROI_CUBE_HEAD.SHAPE_MEMORY_TOPK = 8
    cfg.MODEL.ROI_CUBE_HEAD.SHAPE_MEMORY_MOMENTUM = 0.9
    cfg.MODEL.ROI_CUBE_HEAD.SHAPE_MEMORY_MIN_CONFIDENCE = 0.65
    cfg.MODEL.ROI_CUBE_HEAD.SHAPE_MEMORY_MAX_UPDATES = 32
    cfg.MODEL.ROI_CUBE_HEAD.SHAPE_PROTOTYPE_BLEND = 0.25

    # Frozen DINOv2 multi-layer feature encoder.
    cfg.MODEL.ROI_CUBE_HEAD.DINOV2_CHECKPOINT = \
        "/data/ZhaoX/ovmono3d/checkpoints/dinov2_vitb14_pretrain.pth"
    cfg.MODEL.ROI_CUBE_HEAD.DINOV2_IMAGE_SIZE = 336
    cfg.MODEL.ROI_CUBE_HEAD.DINOV2_OUTPUT_DIM = 128
    cfg.MODEL.ROI_CUBE_HEAD.DINOV2_LAYERS = (2, 5, 8, 11)
    cfg.MODEL.ROI_CUBE_HEAD.DINOV2_CHUNK_SIZE = 2

    # Differentiable cuboid render-back supervision.
    cfg.MODEL.ROI_CUBE_HEAD.USE_DIFFERENTIABLE_RENDERER = False
    cfg.MODEL.ROI_CUBE_HEAD.RENDER_SIZE = 28
    cfg.MODEL.ROI_CUBE_HEAD.RENDER_EDGE_SOFTNESS = 0.02
    cfg.MODEL.ROI_CUBE_HEAD.RENDER_DEPTH_TEMPERATURE = 0.08
    cfg.MODEL.ROI_CUBE_HEAD.LOSS_W_RENDER_SILHOUETTE = 0.10
    cfg.MODEL.ROI_CUBE_HEAD.LOSS_W_RENDER_DEPTH = 0.05

    # In-training EMA teacher. The teacher uses oracle 2D boxes and only
    # replaces uncertain 3D targets after warmup.
    cfg.MODEL.EMA_TEACHER = CN()
    cfg.MODEL.EMA_TEACHER.ENABLED = False
    cfg.MODEL.EMA_TEACHER.DECAY = 0.999
    cfg.MODEL.EMA_TEACHER.WARMUP_ITERS = 3000
    cfg.MODEL.EMA_TEACHER.UPDATE_INTERVAL = 1
    cfg.MODEL.EMA_TEACHER.MIN_SCORE = 0.35
    cfg.MODEL.EMA_TEACHER.MIN_GEOMETRY_SCORE = 0.35
    cfg.MODEL.EMA_TEACHER.MAX_BLEND = 0.35

    # weight path specifically for pretraining (no checkpointables will be loaded)
    cfg.MODEL.WEIGHTS_PRETRAIN = ''
