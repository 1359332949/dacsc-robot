if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import json
import hydra
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import pickle
import numpy as np
import shutil
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_transformer_hybrid_image_policy import DiffusionTransformerHybridImagePolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from accelerate.utils import broadcast_object_list
from accelerate.utils import InitProcessGroupKwargs
from datetime import timedelta
import time

# hide wandb warnings
import logging
logging.getLogger("wandb").setLevel(logging.ERROR)


OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainDiffusionTransformerHybridWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DiffusionTransformerHybridImagePolicy = hydra.utils.instantiate(cfg.policy)

        self.ema_model: DiffusionTransformerHybridImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        self.optimizer = self.model.get_optimizer(**cfg.optimizer)

        # configure training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_action_mse = None
        self.best_val_step = None
        self.best_val_state = None

    def _track_best_val(self, policy, action_mse, global_step):
        """Keep the val-selected weights for the final, one-shot test."""
        if self.best_val_action_mse is None or action_mse < self.best_val_action_mse:
            self.best_val_action_mse = float(action_mse)
            self.best_val_step = int(global_step)
            self.best_val_state = {
                k: v.detach().cpu().clone() for k, v in policy.state_dict().items()
            }

    @torch.no_grad()
    def _action_mse(self, policy, dataloader, device, max_steps=None, seed=12345):
        """Held-out action MSE from ``predict_action``, never training loss.

        The reduction is kept on-device and converted to a Python float only
        once, avoiding a synchronization for every validation batch.
        """
        if dataloader is None or len(dataloader) == 0:
            return None
        cpu_state = torch.random.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        was_training = policy.training
        mse_sum = None
        n_samples = 0
        n_batches = 0
        try:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            policy.eval()
            for i, batch in enumerate(dataloader):
                if max_steps is not None and i >= int(max_steps):
                    break
                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                pred = policy.predict_action(batch["obs"])["action_pred"]
                gt = batch["action"]
                mse = F.mse_loss(pred, gt)
                b = int(gt.shape[0])
                term = mse.detach() * b
                mse_sum = term if mse_sum is None else mse_sum + term
                n_samples += b
                n_batches += 1
        finally:
            policy.train(was_training)
            torch.random.set_rng_state(cpu_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)
        if mse_sum is None:
            return None
        return {
            "action_mse": float(mse_sum / max(n_samples, 1)),
            "samples": int(n_samples),
            "batches": int(n_batches),
        }

    def run(self):
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        timeout_handler = InitProcessGroupKwargs(timeout=timedelta(hours=3)) # have processes wait up to 3 hours for evals to finish
        cfg = copy.deepcopy(self.cfg)
        # set split_batches=True so that effective batch size stays the same regardless of num GPUs
        log_with = None if str(cfg.logging.get("mode", "online")) == "disabled" else "wandb"
        accelerator = Accelerator(log_with=log_with,
                                   kwargs_handlers=[ddp_kwargs, timeout_handler],
                                   split_batches=True)

        if cfg.training.debug:
            cfg.logging.project = "debug"

        if log_with is not None:
            wandb_cfg = OmegaConf.to_container(cfg.logging, resolve=True)
            wandb_cfg.pop('project')
            accelerator.init_trackers(
                project_name=cfg.logging.project,
                config=OmegaConf.to_container(cfg, resolve=True),
                init_kwargs={"wandb": wandb_cfg}
            )

        # wandb_run = wandb.init(
        #     dir=str(self.output_dir),
        #     config=OmegaConf.to_container(cfg, resolve=True),
        #     **cfg.logging
        # )
        # wandb.config.update(
        #     {
        #         "output_dir": self.output_dir,
        #     }
        # )

        if accelerator.is_main_process:
            data_to_share = self.output_dir
        else:
            data_to_share = None
        data_list = [data_to_share]
        broadcast_object_list(data_list)
        self._output_dir = data_list[0]

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                accelerator.print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        if "ckpt_path" in cfg.task and cfg.task.ckpt_path is not None:
            accelerator.print(f"Initializing from checkpoint {cfg.task.ckpt_path}")
            if cfg.training.resume:
                self.load_checkpoint(path=cfg.task.ckpt_path)
            else:
                self.load_checkpoint(path=cfg.task.ckpt_path, include_keys=[])

        # configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)

        dataloader_cfg = copy.deepcopy(OmegaConf.to_container(cfg.dataloader))
        if hasattr(dataset, "get_dataset_sampler"):
            sampler = dataset.get_dataset_sampler()
            if sampler is not None:
                dataloader_cfg["sampler"] = sampler
                dataloader_cfg["shuffle"] = False
                print("using custom dataloader sampler")
        train_dataloader = DataLoader(dataset, **dataloader_cfg)
        normalizer_path = os.path.join(self.output_dir, 'normalizer.pkl')
        if accelerator.is_main_process:
            normalizer = dataset.get_normalizer()
            pickle.dump(normalizer, open(normalizer_path, 'wb'))

        # load normalizer on all processes
        accelerator.wait_for_everyone()
        normalizer = pickle.load(open(normalizer_path, 'rb'))

        # configure validation and held-out test datasets
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)
        test_dataset = dataset.get_test_dataset() if hasattr(dataset, "get_test_dataset") else None
        test_dataloader = (DataLoader(test_dataset, **cfg.val_dataloader)
                           if test_dataset is not None and len(test_dataset) > 0 else None)
        print(f"DACSC split: val={len(val_dataset)} steps, "
              f"test={0 if test_dataset is None else len(test_dataset)} steps", flush=True)

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        if cfg.training.max_train_steps is None:
            num_training_steps = (len(train_dataloader) * cfg.training.num_epochs) // cfg.training.gradient_accumulate_every
        else:
            num_training_steps = (cfg.training.max_train_steps * cfg.training.num_epochs) // cfg.training.gradient_accumulate_every

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=num_training_steps,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        ### Soroush: this part is removed, created dynamically when needed and closed
        # # configure env
        # if cfg.training.rollout_every is not None:
        #     env_runner: BaseImageRunner
        #     env_runner = hydra.utils.instantiate(
        #         cfg.task.env_runner,
        #         output_dir=self.output_dir)
        #     assert isinstance(env_runner, BaseImageRunner)

        # # configure logging
        # wandb_run = wandb.init(
        #     dir=str(self.output_dir),
        #     config=OmegaConf.to_container(cfg, resolve=True),
        #     **cfg.logging
        # )
        # wandb.config.update(
        #     {
        #         "output_dir": self.output_dir,
        #     }
        # )

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # accelerator
        train_dataloader, val_dataloader, self.model, self.optimizer, lr_scheduler = accelerator.prepare(
            train_dataloader, val_dataloader, self.model, self.optimizer, lr_scheduler
        )
        train_dataloader_iter = iter(train_dataloader)
        print("dataloader length:", len(train_dataloader))
        print("dataset length:", len(dataset))
        device = self.model.device
        assert device.type == "cuda", (
            f"DACSC: 训练设备不是 CUDA，而是 {device!r}；拒绝静默 CPU 回退")
        print("DACSC training device:", device,
              "|", torch.cuda.get_device_name(device))
        if self.ema_model is not None:
            self.ema_model.to(device)

        # save batch for sampling
        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

            cfg.task.env_runner.n_test = 5
            cfg.task.env_runner.max_steps = 40

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                self.model.train()
                step_log = dict()
                # ========= train for this epoch ==========
                train_losses = list()
                # max_train_steps = cfg.training.max_train_steps
                # tqdm_kwargs = {}
                # if max_train_steps is not None:
                #     tqdm_kwargs["total"] = max_train_steps
                # with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}",
                #         leave=False, mininterval=cfg.training.tqdm_interval_sec, **tqdm_kwargs) as tepoch:
                #     for batch_idx, batch in enumerate(tepoch):

                assert cfg.training.max_train_steps is not None
                for batch_idx in tqdm.tqdm(range(cfg.training.max_train_steps), desc=f"Training epoch {self.epoch}", leave=False, mininterval=cfg.training.tqdm_interval_sec):
                    try:
                        batch = next(train_dataloader_iter)
                    except StopIteration:
                        # reset for next dataset pass
                        t1 = time.time()
                        print("Creating new train dataloader iterator")
                        train_dataloader_iter = iter(train_dataloader)
                        batch = next(train_dataloader_iter)
                        print(f"Done after {time.time() - t1} seconds.")
                    # device transfer
                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                    if train_sampling_batch is None:
                        train_sampling_batch = batch

                    # compute loss
                    raw_loss = self.model(batch)
                    loss = raw_loss / cfg.training.gradient_accumulate_every
                    loss.backward()

                    # step optimizer
                    if self.global_step % cfg.training.gradient_accumulate_every == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        lr_scheduler.step()

                    # update ema
                    if cfg.training.use_ema:
                        ema.step(accelerator.unwrap_model(self.model))

                    # logging
                    raw_loss_cpu = raw_loss.item()
                    # tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                    train_losses.append(raw_loss_cpu)
                    step_log = {
                        'train_loss': raw_loss_cpu,
                        'global_step': self.global_step,
                        'epoch': self.epoch,
                        'lr': lr_scheduler.get_last_lr()[0]
                    }

                    is_last_batch = (batch_idx == (len(train_dataloader)-1))
                    if not is_last_batch:
                        # log of last step is combined with validation and rollout
                        accelerator.log(step_log, step=self.global_step)
                        json_logger.log(step_log)
                        self.global_step += 1

                    # step-based held-out action MSE (never uses the test split)
                    val_every_steps = int(cfg.training.get("val_every_steps", 0) or 0)
                    if (val_every_steps > 0
                            and (batch_idx + 1) % val_every_steps == 0
                            and batch_idx + 1 < cfg.training.max_train_steps
                            and accelerator.is_main_process):
                        policy_step = accelerator.unwrap_model(self.model)
                        if cfg.training.use_ema:
                            policy_step = self.ema_model
                        val_step = self._action_mse(
                            policy_step, val_dataloader, device,
                            max_steps=cfg.training.max_val_steps)
                        if val_step is not None:
                            self._track_best_val(
                                policy_step, val_step["action_mse"], self.global_step)
                            json_logger.log({
                                "val_action_mse": val_step["action_mse"],
                                "val_action_mse_step": val_step["action_mse"],
                                "val_samples": val_step["samples"],
                                "val_batches": val_step["batches"],
                                "global_step": self.global_step,
                                "epoch": self.epoch,
                            })
                            print(f"[VAL_STEP] step={batch_idx + 1} "
                                  f"val_action_mse={val_step['action_mse']:.6f} "
                                  f"samples={val_step['samples']}", flush=True)
                        self.model.train()

                    # if (cfg.training.max_train_steps is not None) \
                    #     and batch_idx >= (cfg.training.max_train_steps-1):
                    #     break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss

                # ========= eval for this epoch ==========
                policy = accelerator.unwrap_model(self.model)
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # run rollout
                if cfg.training.rollout_every is not None:
                    if self.epoch > 0 and (self.epoch % cfg.training.rollout_every) == 0:
                        if accelerator.is_main_process:
                            env_runner: BaseImageRunner
                            env_runner = hydra.utils.instantiate(
                                cfg.task.env_runner,
                                output_dir=self.output_dir)
                            assert isinstance(env_runner, BaseImageRunner)

                            runner_log = env_runner.run(policy)
                            # log all
                            step_log.update(runner_log)

                            # close and discard env_runner
                            env_runner.close()
                            del env_runner

                        accelerator.wait_for_everyone()

                # run validation (DACSC: held-out styles 31-40; test is 41-50)
                if cfg.training.val_every is not None and (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad():
                        val_losses = list()
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}",
                                leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                loss = policy.compute_loss(batch)
                                val_losses.append(loss.detach().float())
                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps-1):
                                    break
                        if len(val_losses) > 0:
                            val_loss = torch.stack(val_losses).mean().item()
                            step_log['val_loss'] = val_loss
                        val_result = self._action_mse(
                            policy, val_dataloader, device,
                            max_steps=cfg.training.max_val_steps)
                        if val_result is not None:
                            self._track_best_val(
                                policy, val_result["action_mse"], self.global_step)
                            step_log['val_action_mse'] = val_result["action_mse"]
                            step_log['val_samples'] = val_result["samples"]
                            step_log['val_batches'] = val_result["batches"]
                            print(f"[VAL] val_action_mse={val_result['action_mse']:.6f} "
                                  f"samples={val_result['samples']}", flush=True)

                # run diffusion sampling on a training batch
                if (self.epoch % cfg.training.sample_every) == 0:
                    if accelerator.is_main_process:
                        with torch.no_grad():
                            # sample trajectory from training set, and evaluate difference
                            batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                            obs_dict = batch['obs']
                            gt_action = batch['action']

                            result = policy.predict_action(obs_dict)
                            pred_action = result['action_pred']
                            mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                            step_log['train_action_mse_error'] = mse.item()
                            del batch
                            del obs_dict
                            del gt_action
                            del result
                            del pred_action
                            del mse

                    accelerator.wait_for_everyone()

                # checkpoint
                if self.epoch >= 0 and (self.epoch % cfg.training.checkpoint_every) == 0:
                    if accelerator.is_main_process:
                        model_ddp = self.model
                        self.model = accelerator.unwrap_model(self.model)
                        # checkpointing
                        if cfg.checkpoint.save_last_ckpt:
                            self.save_checkpoint()
                        if cfg.checkpoint.save_last_snapshot:
                            self.save_snapshot()

                        # sanitize metric names
                        metric_dict = dict()
                        for key, value in step_log.items():
                            new_key = key.replace('/', '_')
                            metric_dict[new_key] = value

                        # We can't copy the last checkpoint here
                        # since save_checkpoint uses threads.
                        # therefore at this point the file might have been empty!
                        # Never rank a checkpoint without its held-out monitor
                        # metric; TopKCheckpointManager's fallback (-1) would
                        # otherwise look artificially optimal for mode=min.
                        if cfg.checkpoint.topk.monitor_key in metric_dict:
                            topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
                        else:
                            topk_ckpt_path = None

                        if topk_ckpt_path is not None:
                            self.save_checkpoint(path=topk_ckpt_path)

                        self.model = model_ddp

                    accelerator.wait_for_everyone()
                # ========= eval end for this epoch ==========

                # end of epoch
                # log of last step is combined with validation and rollout
                accelerator.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1

        # Final held-out test evaluation: once, after training, never used
        # for early stopping or checkpoint selection.
        if test_dataloader is not None and accelerator.is_main_process:
            policy = accelerator.unwrap_model(self.model)
            if cfg.training.use_ema:
                policy = self.ema_model
            if self.best_val_state is None:
                raise RuntimeError(
                    "held-out test requested without any val_action_mse; "
                    "refusing to select a model by train loss")
            policy.load_state_dict({
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in self.best_val_state.items()
            })
            test_result = self._action_mse(
                policy, test_dataloader, device,
                max_steps=cfg.training.get("max_test_steps",
                                           cfg.training.max_val_steps))
            if test_result is None:
                raise RuntimeError("held-out test split produced no batches")
            test_metrics = {
                "test_action_mse": test_result["action_mse"],
                "test_samples": test_result["samples"],
                "test_batches": test_result["batches"],
                "test_dataset_steps": len(test_dataset),
                "test_style_min": int(getattr(dataset, "_test_style_min", -1)),
                "test_style_max": int(getattr(dataset, "_test_style_max", -1)),
                "selected_by": "val_action_mse",
                "best_val_action_mse": self.best_val_action_mse,
                "best_val_step": self.best_val_step,
            }
            with open(os.path.join(self.output_dir, "test_metrics.json"), "w") as f:
                json.dump(test_metrics, f, indent=2)
            print(f"[TEST] test_action_mse={test_result['action_mse']:.6f} "
                  f"samples={test_result['samples']}", flush=True)
        accelerator.wait_for_everyone()
        accelerator.end_training()

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainDiffusionTransformerHybridWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
