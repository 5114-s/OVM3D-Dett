# Copyright (c) Meta Platforms, Inc. and affiliates
import logging
import os
import sys
import numpy as np
import copy
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import (
    default_argument_parser, 
    default_setup, 
    default_writers, 
    launch
)
from detectron2.solver import build_lr_scheduler
from detectron2.utils.events import EventStorage
from detectron2.utils.logger import setup_logger

logger = logging.getLogger("cubercnn")

sys.dont_write_bytecode = True
sys.path.append(os.getcwd())
np.set_printoptions(suppress=True)

from cubercnn.solver import build_optimizer, freeze_bn, PeriodicCheckpointerOnlyOne
from cubercnn.config import get_cfg_defaults
from cubercnn.data import (
    load_omni3d_json,
    DatasetMapper3D,
    build_detection_train_loader,
    build_detection_test_loader,
    get_omni3d_categories,
    simple_register
)
from cubercnn.evaluation import (
    Omni3DEvaluator, Omni3Deval,
    Omni3DEvaluationHelper,
    inference_on_dataset
)
from cubercnn.modeling.proposal_generator import RPNWithIgnore
from cubercnn.modeling.roi_heads import ROIHeads3D_Text
from cubercnn.modeling.meta_arch import RCNN3D_text, build_model
from cubercnn.modeling.backbone import build_dla_from_vision_fpn_backbone
from cubercnn import util, vis, data
import cubercnn.vis.logperf as utils_logperf

from transformers import BertTokenizer, BertModel

MAX_TRAINING_ATTEMPTS = 10


def unwrap_model(model):
    return model.module if isinstance(model, DistributedDataParallel) else model


def build_ema_teacher(model):
    student = unwrap_model(model)
    student_dino = getattr(student.roi_heads, "dino_encoder", None)
    shared_dino_backbone = (
        getattr(student_dino, "model", None)
        if student_dino is not None
        else None
    )
    if shared_dino_backbone is not None:
        student_dino.model = torch.nn.Identity()
    try:
        teacher = copy.deepcopy(student)
    finally:
        if shared_dino_backbone is not None:
            student_dino.model = shared_dino_backbone
    if shared_dino_backbone is not None:
        teacher.roi_heads.dino_encoder.model = shared_dino_backbone
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher


@torch.no_grad()
def update_ema_teacher(teacher, model, decay):
    student = unwrap_model(model)
    teacher_parameters = dict(teacher.named_parameters())
    for name, student_parameter in student.named_parameters():
        teacher_parameter = teacher_parameters.get(name)
        if teacher_parameter is None or teacher_parameter is student_parameter:
            continue
        teacher_parameter.mul_(decay).add_(
            student_parameter.detach(),
            alpha=1.0 - decay,
        )
    teacher_buffers = dict(teacher.named_buffers())
    for name, student_buffer in student.named_buffers():
        teacher_buffer = teacher_buffers.get(name)
        if teacher_buffer is None or teacher_buffer is student_buffer:
            continue
        teacher_buffer.copy_(student_buffer.detach())


def project_rotation_matrix(matrix):
    u, _, vh = torch.linalg.svd(matrix)
    rotation = u @ vh
    determinant = torch.det(rotation)
    if determinant < 0:
        u[:, -1] *= -1
        rotation = u @ vh
    return rotation


@torch.no_grad()
def apply_ema_teacher_targets(
    cfg,
    teacher,
    data,
    text_embeddings,
    iteration,
):
    warmup = int(cfg.MODEL.EMA_TEACHER.WARMUP_ITERS)
    if iteration < warmup:
        return 0, 0.0
    max_blend = float(cfg.MODEL.EMA_TEACHER.MAX_BLEND)
    min_score = float(cfg.MODEL.EMA_TEACHER.MIN_SCORE)
    ramp = min(1.0, (iteration - warmup + 1) / max(warmup, 1))
    device = next(teacher.parameters()).device

    teacher_inputs = []
    valid_indices = []
    for sample in data:
        instances = sample.get("instances")
        if instances is None:
            indices = torch.empty(0, dtype=torch.long)
            boxes = torch.empty((0, 4), device=device)
            classes = torch.empty(0, dtype=torch.long, device=device)
            render_masks = torch.empty((0, 28, 28), device=device)
        else:
            classes_all = instances.gt_classes
            valid = (
                (classes_all >= 0)
                & (classes_all < cfg.MODEL.ROI_HEADS.NUM_CLASSES)
            )
            indices = torch.nonzero(valid, as_tuple=False).flatten()
            boxes = instances.gt_boxes.tensor[indices].to(device)
            classes = classes_all[indices].to(device)
            render_masks = (
                instances.gt_render_masks[indices].to(device)
                if instances.has("gt_render_masks")
                else torch.empty((len(indices), 28, 28), device=device)
            )
        valid_indices.append(indices)
        teacher_sample = {
            key: value
            for key, value in sample.items()
            if key != "instances"
        }
        teacher_sample["oracle2D"] = {
            "gt_bbox2D": boxes,
            "gt_classes": classes,
            "gt_render_masks": render_masks,
        }
        teacher_inputs.append(teacher_sample)

    if not teacher_inputs:
        return 0, 0.0

    predictions = teacher.inference(
        teacher_inputs,
        text_embeddings,
        do_postprocess=False,
    )
    updated = 0
    blend_values = []
    for sample, indices, prediction in zip(data, valid_indices, predictions):
        if indices.numel() == 0 or len(prediction) == 0:
            continue
        count = min(indices.numel(), len(prediction))
        indices = indices[:count]
        instances = sample["instances"]
        scores = prediction.scores[:count].detach().cpu()
        centers = prediction.pred_center_cam[:count].detach().cpu()
        center_2d = prediction.pred_center_2D[:count].detach().cpu()
        dimensions = prediction.pred_dimensions[:count].detach().cpu()
        poses = prediction.pred_pose[:count].detach().cpu()
        geometry_scores = (
            prediction.pred_geometry_score[:count].detach().cpu()
            if prediction.has("pred_geometry_score")
            else torch.ones(count)
        )
        geometry_confidence = (
            prediction.pred_geometry_confidence[:count].detach().cpu()
            if prediction.has("pred_geometry_confidence")
            else torch.ones(count)
        )
        current_height = max(int(sample["image"].shape[1]), 1)
        image_scale = float(sample["height"]) / current_height
        center_2d = center_2d / max(image_scale, 1e-6)

        target_boxes = instances.gt_boxes3D
        target_poses = instances.gt_poses
        if instances.has("gt_pseudo_weight_joint"):
            source_confidence = instances.gt_pseudo_weight_joint[indices]
        elif instances.has("gt_pseudo_weight"):
            source_confidence = instances.gt_pseudo_weight[indices]
        else:
            source_confidence = torch.ones(count)

        for local_index, target_index in enumerate(indices.tolist()):
            score = float(scores[local_index])
            geometry_score = float(
                geometry_scores[local_index] * geometry_confidence[local_index]
            )
            center = centers[local_index]
            dims = dimensions[local_index]
            pose = poses[local_index]
            if (
                score < min_score
                or geometry_score < float(cfg.MODEL.EMA_TEACHER.MIN_GEOMETRY_SCORE)
                or not torch.isfinite(center).all()
                or not torch.isfinite(dims).all()
                or not torch.isfinite(pose).all()
                or center[2] <= 0.05
                or torch.any(dims <= 0.01)
            ):
                continue
            blend = (
                max_blend
                * ramp
                * score
                * geometry_score
                * (1.0 - 0.5 * float(source_confidence[local_index]))
            )
            blend = float(np.clip(blend, 0.0, max_blend))
            if blend <= 0:
                continue

            old_box = target_boxes[target_index]
            old_center = old_box[6:9]
            old_dims = old_box[3:6].clamp(min=0.01)
            blended_center = (1.0 - blend) * old_center + blend * center
            blended_dims = torch.exp(
                (1.0 - blend) * torch.log(old_dims)
                + blend * torch.log(dims.clamp(min=0.01))
            )
            blended_center_2d = (
                (1.0 - blend) * old_box[:2]
                + blend * center_2d[local_index]
            )
            target_boxes[target_index, :2] = blended_center_2d
            target_boxes[target_index, 2] = blended_center[2]
            target_boxes[target_index, 3:6] = blended_dims
            target_boxes[target_index, 6:9] = blended_center
            blended_pose = (
                (1.0 - blend) * target_poses[target_index]
                + blend * pose
            )
            target_poses[target_index] = project_rotation_matrix(blended_pose)
            updated += 1
            blend_values.append(blend)

    mean_blend = float(np.mean(blend_values)) if blend_values else 0.0
    return updated, mean_blend


def do_test(cfg, model, text_embeddings, iteration='final', storage=None):
        
    filter_settings = data.get_filter_settings_from_cfg(cfg)    
    filter_settings['visibility_thres'] = cfg.TEST.VISIBILITY_THRES
    filter_settings['truncation_thres'] = cfg.TEST.TRUNCATION_THRES
    filter_settings['min_height_thres'] = 0.0625
    filter_settings['max_depth'] = 1e8

    dataset_names_test = cfg.DATASETS.TEST
    only_2d = cfg.MODEL.ROI_CUBE_HEAD.LOSS_W_3D == 0.0
    output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", 'iter_{}'.format(iteration))

    eval_helper = Omni3DEvaluationHelper(
        dataset_names_test, 
        filter_settings, 
        output_folder, 
        iter_label=iteration,
        only_2d=only_2d,
    )

    for dataset_name in dataset_names_test:
        """
        Cycle through each dataset and test them individually.
        This loop keeps track of each per-image evaluation result, 
        so that it doesn't need to be re-computed for the collective.
        """

        '''
        Distributed Cube R-CNN inference
        '''
        data_loader = build_detection_test_loader(cfg, dataset_name)
        results_json = inference_on_dataset(model, data_loader, text_embeddings)

        if comm.is_main_process():
            
            '''
            Individual dataset evaluation
            '''
            eval_helper.add_predictions(dataset_name, results_json)
            eval_helper.save_predictions(dataset_name)
            eval_helper.evaluate(dataset_name)

            '''
            Optionally, visualize some instances
            '''
            # instances = torch.load(os.path.join(output_folder, dataset_name, 'instances_predictions.pth'))
            # log_str = vis.visualize_from_instances(
            #     instances, data_loader.dataset, dataset_name, 
            #     cfg.INPUT.MIN_SIZE_TEST, os.path.join(output_folder, dataset_name), 
            #     MetadataCatalog.get('omni3d_model').thing_classes, iteration
            # )
            # logger.info(log_str)

    if comm.is_main_process():
        
        '''
        Summarize each Omni3D Evaluation metric
        '''  
        eval_helper.summarize_all()


def do_train(cfg, model, dataset_id_to_unknown_cats, dataset_id_to_src, text_embeddings, resume=False):

    max_iter = cfg.SOLVER.MAX_ITER
    do_eval = cfg.TEST.EVAL_PERIOD > 0

    model.train()

    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    # bookkeeping
    checkpointer = DetectionCheckpointer(model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler)    
    periodic_checkpointer = PeriodicCheckpointerOnlyOne(checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter)
    writers = default_writers(cfg.OUTPUT_DIR, max_iter) if comm.is_main_process() else []
    
    # create the dataloader
    data_mapper = DatasetMapper3D(cfg, is_train=True)
    data_loader = build_detection_train_loader(cfg, mapper=data_mapper, dataset_id_to_src=dataset_id_to_src)

    # give the mapper access to dataset_ids
    data_mapper.dataset_id_to_unknown_cats = dataset_id_to_unknown_cats

    if cfg.MODEL.WEIGHTS_PRETRAIN != '':
        
        # load ONLY the model, no checkpointables.
        checkpointer.load(cfg.MODEL.WEIGHTS_PRETRAIN, checkpointables=[])

    # determine the starting iteration, if resuming
    start_iter = (checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1)
    iteration = start_iter
    ema_teacher = (
        build_ema_teacher(model)
        if cfg.MODEL.EMA_TEACHER.ENABLED
        else None
    )

    logger.info("Starting training from iteration {}".format(start_iter))

    if not cfg.MODEL.USE_BN:
        freeze_bn(model)

    world_size = comm.get_world_size()

    # if the loss diverges for more than the below TOLERANCE
    # as a percent of the iterations, the training will stop.
    # This is only enabled if "STABILIZE" is on, which 
    # prevents a single example from exploding the training. 
    iterations_success = 0
    iterations_explode = 0
    
    # when loss > recent_loss * TOLERANCE, then it could be a
    # diverging/failing model, which we should skip all updates for.
    TOLERANCE = 4.0         

    GAMMA = 0.02            # rolling average weight gain
    recent_loss = None      # stores the most recent loss magnitude

    data_iter = iter(data_loader)

    # model.parameters() is surprisingly expensive at 150ms, so cache it
    named_params = list(model.named_parameters())

    with EventStorage(start_iter) as storage:
        
        while True:

            data = next(data_iter)
            storage.iter = iteration

            if ema_teacher is not None:
                ema_updated, ema_blend = apply_ema_teacher_targets(
                    cfg,
                    ema_teacher,
                    data,
                    text_embeddings,
                    iteration,
                )
                if comm.is_main_process():
                    storage.put_scalar(
                        "ema_teacher/updated_targets",
                        ema_updated,
                        smoothing_hint=False,
                    )
                    storage.put_scalar(
                        "ema_teacher/mean_blend",
                        ema_blend,
                        smoothing_hint=False,
                    )

            # forward
            loss_dict = model(data, text_embeddings)
            losses = sum(loss_dict.values())

            # reduce
            loss_dict_reduced = {k: v.item() for k, v in allreduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        
            # sync up
            comm.synchronize()

            if recent_loss is None:

                # init recent loss fairly high
                recent_loss = losses_reduced*2.0

            # Is stabilization enabled, and loss high or NaN?
            diverging_model = cfg.MODEL.STABILIZE > 0 and \
                        (losses_reduced > recent_loss*TOLERANCE or \
                            not (np.isfinite(losses_reduced)) or np.isnan(losses_reduced))

            if diverging_model:
                # clip and warn the user.
                losses = losses.clip(0, 1) 
                logger.warning('Skipping gradient update due to higher than normal loss {:.2f} vs. rolling mean {:.2f}, Dict-> {}'.format(
                    losses_reduced, recent_loss, loss_dict_reduced
                ))
            else:
                # compute rolling average of loss
                recent_loss = recent_loss * (1-GAMMA) + losses_reduced*GAMMA
            
            if comm.is_main_process():
                # send loss scalars to tensorboard.
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)
        
            # backward and step
            optimizer.zero_grad()
            losses.backward()

            # if the loss is not too high, 
            # we still want to check gradients.
            if not diverging_model:

                if cfg.MODEL.STABILIZE > 0:
                    
                    for name, param in named_params:

                        if param.grad is not None:
                            diverging_model = torch.isnan(param.grad).any() or torch.isinf(param.grad).any()
                        
                        if diverging_model:
                            logger.warning('Skipping gradient update due to inf/nan detection, loss is {}'.format(loss_dict_reduced))
                            break

            # convert exploded to a float, then allreduce it, 
            # if any process gradients have exploded then we skip together.
            diverging_model = torch.tensor(float(diverging_model)).cuda()

            if world_size > 1:
                dist.all_reduce(diverging_model)

            # sync up
            comm.synchronize()

            if diverging_model > 0:
                optimizer.zero_grad()
                iterations_explode += 1

            else:
                optimizer.step()
                if (
                    ema_teacher is not None
                    and (iteration + 1) % int(cfg.MODEL.EMA_TEACHER.UPDATE_INTERVAL) == 0
                ):
                    update_ema_teacher(
                        ema_teacher,
                        model,
                        float(cfg.MODEL.EMA_TEACHER.DECAY),
                    )
                storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
                iterations_success += 1

            total_iterations = iterations_success + iterations_explode

            # Only retry if we have trained sufficiently long relative
            # to the latest checkpoint, which we would otherwise revert back to.
            retry = (iterations_explode / total_iterations) >= cfg.MODEL.STABILIZE \
                    and (total_iterations > cfg.SOLVER.CHECKPOINT_PERIOD*1/2)
            
            # Important for dist training. Convert to a float, then allreduce it, 
            # if any process gradients have exploded then we must skip together.
            retry = torch.tensor(float(retry)).cuda()
            
            if world_size > 1:
                dist.all_reduce(retry)

            # sync up
            comm.synchronize()

            # any processes need to retry
            if retry > 0:

                # instead of failing, try to resume the iteration instead. 
                logger.warning('!! Restarting training at {} iters. Exploding loss {:d}% of iters !!'.format(
                    iteration, int(100*(iterations_explode / (iterations_success + iterations_explode)))
                ))

                # send these to garbage, for ideally a cleaner restart.
                del data_mapper
                del data_loader
                del optimizer
                del checkpointer
                del periodic_checkpointer
                del ema_teacher
                return False
                
            scheduler.step()

            # Evaluate only when the loss is not diverging.
            if not (diverging_model > 0) and \
                (do_eval and ((iteration + 1) % cfg.TEST.EVAL_PERIOD) == 0 and iteration != (max_iter - 1)):

                logger.info('Starting test for iteration {}'.format(iteration+1))
                do_test(cfg, model, text_embeddings, iteration=iteration+1, storage=storage)
                comm.synchronize()
                
                if not cfg.MODEL.USE_BN: 
                    freeze_bn(model)

            # Flush events
            if iteration - start_iter > 5 and ((iteration + 1) % 20 == 0 or iteration == max_iter - 1):
                for writer in writers:
                    writer.write()
            
            # Do not bother checkpointing if there is potential for a diverging model.
            if not (diverging_model > 0) and \
                (iterations_explode / total_iterations) < 0.5*cfg.MODEL.STABILIZE:
                periodic_checkpointer.step(iteration)

            iteration += 1

            if iteration >= max_iter:
                break
    
    # success
    return True

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.set_new_allowed(True)
    get_cfg_defaults(cfg)

    config_file = args.config_file
    
    # store locally if needed
    if config_file.startswith(util.CubeRCNNHandler.PREFIX):    
        config_file = util.CubeRCNNHandler._get_local_path(util.CubeRCNNHandler, config_file)

    cfg.merge_from_file(config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)

    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="cubercnn")
    
    filter_settings = data.get_filter_settings_from_cfg(cfg)

    for dataset_name in cfg.DATASETS.TRAIN:
        simple_register(dataset_name, filter_settings, folder_name=cfg.DATASETS.FOLDER_NAME, filter_empty=True)
    
    dataset_names_test = cfg.DATASETS.TEST

    for dataset_name in dataset_names_test:
        if not(dataset_name in cfg.DATASETS.TRAIN):
            simple_register(dataset_name, filter_settings, folder_name=cfg.DATASETS.FOLDER_NAME, filter_empty=False)
    
    return cfg


def main(args):
    
    cfg = setup(args)

    logger.info('Preprocessing Training Datasets')

    filter_settings = data.get_filter_settings_from_cfg(cfg)

    priors = None

    if args.eval_only:
        category_path = os.path.join(util.file_parts(args.config_file)[0], 'category_meta.json')

        # store locally if needed
        if category_path.startswith(util.CubeRCNNHandler.PREFIX):
            category_path = util.CubeRCNNHandler._get_local_path(util.CubeRCNNHandler, category_path)

        metadata = util.load_json(category_path)

        # register the categories
        thing_classes = metadata['thing_classes']
        id_map = {int(key):val for key, val in metadata['thing_dataset_id_to_contiguous_id'].items()}
        MetadataCatalog.get('omni3d_model').thing_classes = thing_classes
        MetadataCatalog.get('omni3d_model').thing_dataset_id_to_contiguous_id  = id_map

    else: 

        # setup and join the data.
        dataset_paths = [os.path.join('datasets', cfg.DATASETS.FOLDER_NAME, name + '.json') for name in cfg.DATASETS.TRAIN]
        datasets = data.Omni3D(dataset_paths, filter_settings=filter_settings)

        # determine the meta data given the datasets used. 
        data.register_and_store_model_metadata(datasets, cfg.OUTPUT_DIR, filter_settings)

        thing_classes = MetadataCatalog.get('omni3d_model').thing_classes
        dataset_id_to_contiguous_id = MetadataCatalog.get('omni3d_model').thing_dataset_id_to_contiguous_id
        
        '''
        It may be useful to keep track of which categories are annotated/known
        for each dataset in use, in case a method wants to use this information.
        '''

        infos = datasets.dataset['info']

        if type(infos) == dict:
            infos = [datasets.dataset['info']]

        dataset_id_to_unknown_cats = {}
        possible_categories = set(i for i in range(cfg.MODEL.ROI_HEADS.NUM_CLASSES + 1))
        
        dataset_id_to_src = {}

        for info in infos:
            dataset_id = info['id']
            known_category_training_ids = set()

            if not dataset_id in dataset_id_to_src:
                dataset_id_to_src[dataset_id] = info['source']

            for id in info['known_category_ids']:
                if id in dataset_id_to_contiguous_id:
                    known_category_training_ids.add(dataset_id_to_contiguous_id[id])
            
            # determine and store the unknown categories.
            unknown_categories = possible_categories - known_category_training_ids
            dataset_id_to_unknown_cats[dataset_id] = unknown_categories

            # log the per-dataset categories
            logger.info('Available categories for {}'.format(info['name']))
            logger.info([thing_classes[i] for i in (possible_categories & known_category_training_ids)])

        # compute priors given the training data.
        priors = util.compute_priors(cfg, datasets)


    # Load the BERT model and obtain the text embeddings of the categories.
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    thing_classes.append('None')
    texts = thing_classes

    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    outputs = model(**inputs)

    encoded_texts = outputs.last_hidden_state
    text_embeddings = encoded_texts[:,1:-1,:].mean(dim=1).cuda().detach()

    '''
    The training loops can attempt to train for N times.
    This catches a divergence or other failure modes. 
    '''

    remaining_attempts = MAX_TRAINING_ATTEMPTS
    while remaining_attempts > 0:

        # build the training model.
        model = build_model(cfg, priors=priors)

        if remaining_attempts == MAX_TRAINING_ATTEMPTS:
            # log the first attempt's settings.
            logger.info("Model:\n{}".format(model))

        if args.eval_only:
            # skip straight to eval mode
            DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
                cfg.MODEL.WEIGHTS, resume=args.resume
            )
            return do_test(cfg, model, text_embeddings)

        # setup distributed training.
        distributed = comm.get_world_size() > 1
        if distributed:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], 
                broadcast_buffers=False, find_unused_parameters=True
            )

        # train full model, potentially with resume.
        if do_train(cfg, model, dataset_id_to_unknown_cats, dataset_id_to_src, text_embeddings, resume=args.resume):
            break
        else:

            # allow restart when a model fails to train.
            remaining_attempts -= 1
            del model

    if remaining_attempts == 0:
        # Exit if the model could not finish without diverging. 
        raise ValueError('Training failed')

    return do_test(cfg, model, text_embeddings)

def allreduce_dict(input_dict, average=True):
    """
    Reduce the values in the dictionary from all processes so that process with rank
    0 has the reduced results.
    Args:
        input_dict (dict): inputs to be reduced. All the values must be scalar CUDA Tensor.
        average (bool): whether to do average or sum
    Returns:
        a dict with the same keys as input_dict, after reduction.
    """
    world_size = comm.get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
