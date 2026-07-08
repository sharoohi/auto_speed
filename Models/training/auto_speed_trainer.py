import contextlib
import os
import copy
import shutil
import warnings
import tyro
import torch
import tqdm
import yaml

from typing import Callable, Optional, Tuple
from contextlib import nullcontext
from pathlib import Path
from dataclasses import dataclass

from torch.utils import data
from torch.profiler import ProfilerActivity

import auto_speed_util as util
from Models.data_utils.load_data_auto_speed import LoadDataAutoSpeed
from Models.model_components.auto_speed.auto_speed_network import AutoSpeedNetwork

from torch.utils.tensorboard import SummaryWriter
import time
from datetime import datetime

warnings.filterwarnings("ignore")


@dataclass
class AutoSpeedTrainingArgs:
    """
    Parameters for a single model training run.

    Used by single training (this script) and HPO, where some of these may
    vary between runs.

    """
    dataset: Path
    """dataset directory path"""
    input_width: int = 1024
    input_height: int = 512
    batch_size: int = 32
    val_batch_size: int = 4
    local_rank: int = 0
    version: str = "n"
    runs_dir: Path = Path("runs")
    epochs: int = 150
    world_size: int = 1
    disable_pbar: bool = False
    do_profile: bool = False
    """Whether to enable pytorch profiling"""
    use_preprocessed: bool = False
    """Whether to use preprocessed numpy files instead of jpg images"""
    warmup_epochs: int = 1
    """ Number of epochs to skip from throughput metrics"""
    do_compile: bool = False
    compile_backend: str = "inductor"
    compile_mode: str = "default"
    compile_dynamic: bool = False
    compile_fullgraph: bool = False
    fp32: bool = False
    """
    Use full float precision (fp32) instead of the default behaviour of switching the model to
    fp16 and using automatic mixed precision with gradient scaling. Increases memory usage, but
    can in some circumstances improve performance.
    """
    disable_vectorized_loss: bool = False
    """
    Use the original implementation of building the GT tensor for
    loss computation. The vectorized version (default) is faster,
    but we support the original version for comparison.
    """
    checkpoint_path: Optional[str] = None

    def __post_init__(self):
        self.distributed: bool = self.world_size > 1
        self.runs_dir = self.runs_dir.absolute()


@dataclass
class AutoSpeedSingleRunTrainingArgs(AutoSpeedTrainingArgs):
    """
    Parameters for a single-run training call only (not HPO).
    """
    config: Tuple[Path, ...] = (Path("../config/auto_speed.yaml"),)
    """Config YAML specifying hyperparameters for the model"""
    profile: bool = False
    """Print model parameter count and FLOPs before training"""
    output_subdir: Optional[str] = None
    """
    Specify an exact output directory name to write results to within runs_dir, overwriting if it already exists.
    Otherwise a new unique subdirectory will be created
    """


class Logger:
    # This should be changed to use proper logging, not a custom invention
    def __init__(self, log_file_path: Path, log_to_stdout: bool = True):
        self.log_file_path = log_file_path
        self.log_to_stdout = log_to_stdout

    def write_log(self, msg, stdout=None):
        """
        Prints immediately to console (optional) and appends to a text file.

        Prints all messages if `log_to_stdout=True`. Overwrite for individual messages using `stdout`.

        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        msg = f"{timestamp} {msg}"
        if stdout is None:
            stdout = self.log_to_stdout
        if stdout:
            print(msg, flush=True)
        try:
            with open(self.log_file_path, "a") as f:
                f.write(msg + "\n")
        except OSError as e:
            warnings.warn(f"Failed to write to log file {self.log_file_path}: {e}")


def train(
        args: AutoSpeedTrainingArgs,
        params,
        run_dir: Path,
        log_writer,
        log_to_stdout: bool = True,
        epoch_callback: Optional[Callable] = None,
        train_folder: str = "train",
        val_folder: str = "val",
):
    logger = Logger(run_dir / "training.log", log_to_stdout=log_to_stdout)

    # Device check
    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(device_id)
        logger.write_log(f"Using GPU: {device_id} ({device_name})")
    else:
        logger.write_log("CUDA not available. Using CPU.")

    if args.checkpoint_path:
        model = AutoSpeedNetwork().load_model(version=args.version, num_classes=4, checkpoint_path=args.checkpoint_path)
    else:
        model = AutoSpeedNetwork().build_model(version=args.version, num_classes=4)
    model.cuda()

    # Optimizer
    accumulate = max(round(64 / (args.batch_size * args.world_size)), 1)
    params['weight_decay'] *= args.batch_size * args.world_size * accumulate / 64

    optimizer = torch.optim.SGD(util.set_params(model, params['weight_decay']),
                                params['min_lr'], params['momentum'], nesterov=True)

    # EMA
    ema = util.EMA(model) if args.local_rank == 0 else None

    input_data_dir = args.dataset / "images" / train_folder if not args.use_preprocessed else args.dataset / "images" / f"{train_folder}_preprocessed"
    logger.write_log(f"Data dir: {input_data_dir} ({'' if args.use_preprocessed else 'not '}preprocessed)")
    filenames = [f.as_posix() for f in input_data_dir.rglob("*") if f.is_file()]

    sampler = None
    if len(filenames) == 0:
        logger.write_log(f"No files found in {input_data_dir}. Check your dataset path and that preprocessing has been run")
        raise IOError(f"No files found in {input_data_dir}. Check your dataset path and that preprocessing has been run")
    dataset = LoadDataAutoSpeed(
        filenames,
        args.input_width,
        args.input_height,
        params,
        augment=True,
        is_preprocessed=args.use_preprocessed,
    )

    if args.distributed:
        sampler = data.distributed.DistributedSampler(dataset)

    loader = data.DataLoader(
        dataset,
        args.batch_size,
        sampler is None,
        sampler,
        num_workers=8,
        pin_memory=True,
        collate_fn=LoadDataAutoSpeed.collate_fn,
    )

    # Scheduler
    num_steps = len(loader)
    scheduler = util.LinearLR(args, params, num_steps)

    if args.distributed:
        # DDP mode
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(module=model,
                                                          device_ids=[args.local_rank],
                                                          output_device=args.local_rank)

    # ===== torch.compile: Graph-level Code Generation =====
    # torch.compile traces the model, reduces Python overhead, and generates optimized kernels.
    # Benefits: 10-50% speedup per epoch (amortizes ~1-2 min first-epoch compilation cost).
    # ========================================================
    uncompiled_model = model
    if args.do_compile:
        if not hasattr(torch, "compile"):
            raise RuntimeError(
                "torch.compile is unavailable in this PyTorch version. "
                "Upgrade to PyTorch >= 2.0 or disable --do-compile."
            )

        compile_mode = None if args.compile_mode.lower() == "none" else args.compile_mode
        supported_modes = {"default", "reduce-overhead", "max-autotune", None}
        if compile_mode not in supported_modes:
            raise ValueError(
                f"Unsupported compile mode '{args.compile_mode}'. "
                "Use one of: default, reduce-overhead, max-autotune, none"
            )

        logger.write_log(
            f"torch.compile enabled: backend={args.compile_backend}, "
            f"mode={args.compile_mode}, dynamic={args.compile_dynamic}, fullgraph={args.compile_fullgraph}. "
            f"(First epoch will be slower due to compilation)"
        )
        
        # Wrap model with torch.compile
        # Note: Compile wraps the entire model after DDP setup for proper gradient flow.
        model = torch.compile(
            model,
            backend=args.compile_backend,           # Code generation backend (inductor/aot_eager/...)
            mode=compile_mode,                      # Speed vs. overhead tradeoff (default/reduce-overhead/max-autotune)
            dynamic=args.compile_dynamic,           # Allow variable batch sizes (slower, but no recompile)
            fullgraph=args.compile_fullgraph,       # Compile entire graph (fail on unsupported ops)
        )
    
    # Setup profiler
    profiler_out_dir = f"{run_dir}/profiler"
    if args.do_profile and args.local_rank == 0:
        if not os.path.exists(profiler_out_dir):
            os.makedirs(profiler_out_dir)
        print(f"Profiling enabled saving profiles to {profiler_out_dir}")
        profiling_context = torch.profiler.profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=False,  # Disable to reduce trace size
                profile_memory=False,  # Disable to reduce trace size
                with_stack=True,
                schedule=torch.profiler.schedule(wait=8, warmup=2, active=10, repeat=1),
                on_trace_ready=lambda prof: prof.export_chrome_trace(
                    f"{profiler_out_dir}/autospeed_trace_bs{args.batch_size}_gpus{args.world_size}.json"
                )
            )
    else:
        profiling_context = nullcontext()

    best = 0
    if args.fp32:
        # Keep everything in full precision
        amp_scale = None
        logger.write_log("FP32 training: disabling AMP and using only float32")
    else:
        # Use mixed precision with gradient scaling
        amp_scale = torch.amp.GradScaler()
    criterion = util.ComputeLoss(model, params, vectorized=not args.disable_vectorized_loss)

    total_samples_all_epochs = 0
    total_compute_time_all_epochs = 0.0
    total_workload_time_all_epochs = 0.0

    for epoch in range(args.epochs):
        if args.disable_pbar:
            logger.write_log(f"Starting epoch {epoch+1}/{args.epochs}")
        model.train()
        if sampler is not None:
            sampler.set_epoch(epoch)
        # if args.epochs - epoch == 10:
        #     loader.dataset.mosaic = False

        data_iter = enumerate(loader)
        p_bar = None

        if args.local_rank == 0 and not args.disable_pbar:
            print(('\n' + '%10s' * 5) % ('epoch', 'memory', 'box', 'cls', 'dfl'))
            p_bar = tqdm.tqdm(data_iter, total=num_steps)
            data_iter = p_bar

        optimizer.zero_grad()
        avg_box_loss = util.AverageMeter()
        avg_cls_loss = util.AverageMeter()
        avg_dfl_loss = util.AverageMeter()
        
        epoch_samples = 0
        epoch_compute_time = 0.0
        epoch_workload_start = time.perf_counter()

        with profiling_context as prof:
            for i, (samples, targets) in data_iter:
                start_time = time.perf_counter()
                step = i + num_steps * epoch
                scheduler.step(step, optimizer)

                if args.fp32:
                    samples = samples.cuda().float()
                else:
                    samples = samples.cuda().half()
                samples /= 255

                # Forward
                with contextlib.nullcontext() if args.fp32 else torch.amp.autocast('cuda'):
                    outputs = model(samples)  # forward
                    loss_box, loss_cls, loss_dfl = criterion(outputs, targets)

                # Don't track averages if we won't be outputting them
                if not args.disable_pbar:
                    avg_box_loss.update(loss_box.item(), samples.size(0))
                    avg_cls_loss.update(loss_cls.item(), samples.size(0))
                    avg_dfl_loss.update(loss_dfl.item(), samples.size(0))

                loss_box *= args.batch_size  # loss scaled by batch_size
                loss_cls *= args.batch_size  # loss scaled by batch_size
                loss_dfl *= args.batch_size  # loss scaled by batch_size
                loss_box *= args.world_size  # gradient averaged between devices in DDP mode
                loss_cls *= args.world_size  # gradient averaged between devices in DDP mode
                loss_dfl *= args.world_size  # gradient averaged between devices in DDP mode

                loss = loss_box + loss_cls + loss_dfl

                # Backward
                if amp_scale is not None:
                    amp_scale.scale(loss).backward()
                else:
                    loss.backward()

                # Optimize
                if step % accumulate == 0:
                    # amp_scale.unscale_(optimizer)  # unscale gradients
                    # util.clip_gradients(model)  # clip gradients
                    if amp_scale is not None:
                        amp_scale.step(optimizer)  # optimizer.step
                        amp_scale.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()
                    if ema:
                        ema.update(uncompiled_model)

                torch.cuda.synchronize()

                if prof is not None:
                    prof.step()

                batch_time = time.perf_counter() - start_time
                local_batch_size = samples.size(0)
                global_batch_size = local_batch_size * (args.world_size if args.distributed else 1)
                epoch_samples += global_batch_size
                epoch_compute_time += batch_time

                # Log
                if args.local_rank == 0 and p_bar is not None:
                    memory = f'{torch.cuda.memory_reserved() / 1E9:.4g}G'  # (GB)
                    # Throughput uses actual batch size (handles short final batch).
                    # Report global throughput in DDP (samples/sec across all ranks).
                    fps = global_batch_size / batch_time if batch_time > 0 else 0
                    s = ('%10s' * 2 + '%10.3g' * 3 + '%10.3g') % (f'{epoch + 1}/{args.epochs}', memory,
                                                            avg_box_loss.avg, avg_cls_loss.avg, avg_dfl_loss.avg, fps)
                    p_bar.set_description(s)
                    logger.write_log(s, stdout=False)
                if args.disable_pbar and (i + 1) % 100 == 0:
                    logger.write_log(f"Epoch {epoch + 1}: processed {i + 1}/{num_steps} batches")

        if args.local_rank == 0:
            epoch_workload_time = time.perf_counter() - epoch_workload_start
            avg_compute_fps = epoch_samples / epoch_compute_time if epoch_compute_time > 0 else 0.0
            avg_workload_fps = epoch_samples / epoch_workload_time if epoch_workload_time > 0 else 0.0
            
            # Only accumulate metrics after warmup phase (skip first N epochs for steady-state measurement)
            if epoch >= args.warmup_epochs:
                total_samples_all_epochs += epoch_samples
                total_compute_time_all_epochs += epoch_compute_time
                total_workload_time_all_epochs += epoch_workload_time

            warmup_note = " (warmup, excluded from overall average)" if epoch < args.warmup_epochs else ""
            logger.write_log(
                f"Epoch {epoch+1}: Average FPS (compute-only): {avg_compute_fps:.2f} | "
                f"Average FPS (full workload incl. dataloader): {avg_workload_fps:.2f}{warmup_note}"
            )
            log_writer.add_scalar("Throughput/FPS_compute_only", avg_compute_fps, epoch + 1)
            log_writer.add_scalar("Throughput/FPS_full_workload", avg_workload_fps, epoch + 1)

            # mAP
            if args.disable_pbar:
                logger.write_log("Running validation")
            last = val(args, params, run_dir, ema.ema, log_to_stdout=log_to_stdout, split=val_folder)

            log_writer.add_scalar("Loss/box", avg_box_loss.avg, epoch + 1)
            log_writer.add_scalar("Loss/cls", avg_cls_loss.avg, epoch + 1)
            log_writer.add_scalar("Loss/dfl", avg_dfl_loss.avg, epoch + 1)

            log_writer.add_scalar("Metrics/mAP", last[0], epoch + 1)
            log_writer.add_scalar("Metrics/mAP@50", last[1], epoch + 1)
            log_writer.add_scalar("Metrics/Recall", last[2], epoch + 1)
            log_writer.add_scalar("Metrics/Precision", last[3], epoch + 1)

            # Update best mAP
            if last[0] > best:
                best = last[0]

            # Save model
            save = {
                'epoch': epoch + 1,
                'model': copy.deepcopy(ema.ema),
            }
            # Save last, best and delete
            torch.save(save, f=f'{run_dir}/weights/last.pt')
            if best == last[0]:
                torch.save(save, f=f'{run_dir}/weights/best.pt')
            del save
        
        # Epoch completed
        if epoch_callback is not None:
            try:
                epoch_callback(epoch + 1, last[0], last[1], last[2], last[3])
            except TypeError:
                epoch_callback()

    if args.local_rank == 0:
        warmup_note = f" (after {args.warmup_epochs} warmup epoch(s))" if args.warmup_epochs > 0 else ""
        if total_compute_time_all_epochs > 0:
            overall_compute_fps = total_samples_all_epochs / total_compute_time_all_epochs
            logger.write_log(f"Overall Average Training Throughput (compute-only FPS){warmup_note}: {overall_compute_fps:.2f}")
        if total_workload_time_all_epochs > 0:
            overall_workload_fps = total_samples_all_epochs / total_workload_time_all_epochs
            logger.write_log(f"Overall Average Training Throughput (full workload FPS){warmup_note}: {overall_workload_fps:.2f}")
        util.strip_optimizer(f'{run_dir}/weights/best.pt')  # strip optimizers
        util.strip_optimizer(f'{run_dir}/weights/last.pt')  # strip optimizers


@torch.no_grad()
def val(args: AutoSpeedTrainingArgs, params, run_dir, model=None, log_to_stdout=True, split="val"):
    logger = Logger(run_dir / "validation.log", log_to_stdout=log_to_stdout)

    # Device check
    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(device_id)
        logger.write_log(f"Validation using GPU: {device_id} ({device_name})")
    else:
        logger.write_log("CUDA not available. Using CPU.")

    current_dir = args.dataset / "images" / split
    filenames = [f.as_posix() for f in current_dir.rglob("*") if f.is_file()]

    dataset = LoadDataAutoSpeed(filenames, args.input_width, args.input_height, params, augment=False)
    loader = data.DataLoader(
        dataset, 
        batch_size=args.val_batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True,
        collate_fn=LoadDataAutoSpeed.collate_fn
    )

    if not model:
        # In pytorch > 2.5 by default weights_only=True
        model = torch.load(f=f'{run_dir}/weights/best.pt', map_location='cuda', weights_only=False)
        model = model['model'].float().fuse()

    if args.fp32:
        model.float()
    else:
        model.half()
    model.eval()

    # Configure
    iou_v = torch.linspace(start=0.5, end=0.95, steps=10).cuda()  # iou vector for mAP@0.5:0.95
    n_iou = iou_v.numel()

    m_pre = 0
    m_rec = 0
    map50 = 0
    mean_ap = 0
    metrics = []
    if not args.disable_pbar:
        loader = tqdm.tqdm(loader, desc=('%10s' * 5) % ('', 'precision', 'recall', 'mAP50', 'mAP'))
    
    for samples, targets in loader:
        samples = samples.cuda()
        # uint8 to fp16/32
        samples = samples.float() if args.fp32 else samples.half()
        samples = samples / 255.  # 0 - 255 to 0.0 - 1.0
        _, _, h, w = samples.shape  # batch-size, channels, height, width
        scale = torch.tensor((w, h, w, h)).cuda()
        # Inference
        outputs = model(samples)
        # NMS
        outputs = util.non_max_suppression(outputs)
        # Metrics
        for i, output in enumerate(outputs):
            idx = targets['idx'] == i
            cls = targets['cls'][idx]
            box = targets['box'][idx]

            cls = cls.cuda()
            box = box.cuda()

            metric = torch.zeros(output.shape[0], n_iou, dtype=torch.bool).cuda()

            if output.shape[0] == 0:
                if cls.shape[0]:
                    metrics.append((metric, *torch.zeros((2, 0)).cuda(), cls.squeeze(-1)))
                continue
            # Evaluate
            if cls.shape[0]:
                target = torch.cat(tensors=(cls, util.wh2xy(box) * scale), dim=1)
                metric = util.compute_metric(output[:, :6], target, iou_v)
            # Append
            metrics.append((metric, output[:, 4], output[:, 5], cls.squeeze(-1)))

    # Compute metrics
    metrics = [torch.cat(x, dim=0).cpu().numpy() for x in zip(*metrics)]  # to numpy
    if len(metrics) and metrics[0].any():
        tp, fp, m_pre, m_rec, map50, mean_ap = util.compute_ap(*metrics, plot=False, names=params["names"])
    # Print results
    logger.write_log(('%10s' + '%10.3g' * 4) % ('', m_pre, m_rec, map50, mean_ap))
    # Return results
    if not args.fp32:
        # Back to full precision (from half)
        model.float()  # for training
    return mean_ap, map50, m_rec, m_pre


def profile(args: AutoSpeedTrainingArgs, params):
    import thop
    shape = (1, 3, args.input_height, args.input_width)
    model = AutoSpeedNetwork().build_model(version=args.version, num_classes=4)
    model.eval()
    model(torch.zeros(shape))

    x = torch.empty(shape)
    flops, num_params = thop.profile(model, inputs=[x], verbose=False)
    flops, num_params = thop.clever_format(nums=[2 * flops, num_params], format="%.3f")

    if args.local_rank == 0:
        print(f'Number of parameters: {num_params}')
        print(f'Number of FLOPs: {flops}')


def new_unique_subdir(base_dir: Path, prefix: str) -> Path:
    suffix_num = 0
    timestamp = datetime.now().strftime("%Y-%m-%d")
    while True:
        subdir_name = f"{prefix}-{timestamp}-{suffix_num:04d}"
        subdir = base_dir / subdir_name
        if not subdir.exists():
            subdir.mkdir(parents=True)
            return subdir
        suffix_num += 1


if __name__ == "__main__":
    args: AutoSpeedSingleRunTrainingArgs = tyro.cli(AutoSpeedSingleRunTrainingArgs)

    # Allow local rank and world size to be overridden by env vars
    args.local_rank = int(os.getenv('LOCAL_RANK', args.local_rank))
    args.world_size = int(os.getenv('WORLD_SIZE', args.world_size))

    # Prepare training directory
    args.runs_dir.mkdir(exist_ok=True, parents=True)
    if args.output_subdir is not None:
        run_dir = args.runs_dir / args.output_subdir
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(exist_ok=True)
    else:
        run_dir = new_unique_subdir(args.runs_dir, "run")
    print(f"Outputting training logs and weights to {run_dir}")
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(exist_ok=True, parents=True)
    log_writer = SummaryWriter(log_dir=run_dir)

    if args.distributed:
        torch.cuda.set_device(device=args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    if args.local_rank == 0:
        weights_dir.mkdir(exist_ok=True, parents=True)

    # Load the hyperparameters we will pass to model training
    params = {}
    for config_path in args.config:
        print(f"Loading config from {config_path}")
        with config_path.open("r", errors="ignore") as f:
            params.update(yaml.safe_load(f))

    util.setup_seed()
    util.setup_multi_processes()

    if args.profile:
        profile(args, params)
    train(args, params, run_dir, log_writer)

    # Clean
    if args.distributed:
        torch.distributed.destroy_process_group()
    torch.cuda.empty_cache()
