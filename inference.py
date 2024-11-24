import logging
import os

import hydra
from hydra.utils import instantiate
from pytorch_lightning import Trainer, seed_everything
# from pytorch_lightning.plugins import DDPPlugin
import torch

from data.data_module import DataModule
from finetune_learner import Learner
from pytorch_lightning.loggers import WandbLogger, CSVLogger
import wandb


# static vars
# os.environ["WANDB_SILENT"] = "true"
logging.getLogger("lightning").propagate = False


@hydra.main(config_path="conf", config_name="config_test")
def main(cfg):
    if cfg.fix_seed:
        seed_everything(42, workers=True)

    # print("The SLURM job ID for this run is {}".format(os.environ["SLURM_JOB_ID"]))
    # cfg.slurm_job_id = os.environ["SLURM_JOB_ID"]

    # cfg.gpus = torch.cuda.device_count()
    cfg.gpus = 1
    print("num gpus:", cfg.gpus)

    print(f"Inside main() function")
    speaker = cfg.speaker
    finetune_type = cfg.finetune
    print(f"{cfg.finetune = }")

    # Parameters
    lr = cfg.optimizer.lr
    wd = cfg.optimizer.weight_decay
    # ds_args = cfg.data.dataset
    window = cfg.data.timemask_window
    stride = cfg.data.timemask_stride
    dropout = cfg.model.visual_backbone.dropout_rate
    beam_size = cfg.decode.beam_size

    project_name = f"raven_{speaker}_inference"
        # run_name = f"pretrained_perf_on_all_labels_beam{beam_size}"
    # run_name = f"pretrained_perf_on_val_reduced_labels_beam{beam_size}"
    run_name = f"pretrained_perf_on_test_reduced_labels_beam{beam_size}"

    cfg.log_folder = os.path.join(cfg.logging_dir, f"{project_name}/{run_name}")
    cfg.exp_dir = cfg.log_folder
    cfg.trainer.default_root_dir = cfg.log_folder
    os.makedirs(cfg.log_folder, exist_ok=True)
    print(f"\nLogging Directory: {cfg.log_folder}")

    # Logging Stuff
    loggers = []
    csv_logger = CSVLogger(
        save_dir=cfg.log_folder,
        flush_logs_every_n_steps=1
    )
    loggers.append(csv_logger)
    if cfg.wandb:
        wandb_logger = WandbLogger(
            name=run_name,
            project=project_name,
            # config=cfg,
            settings=wandb.Settings(code_dir='.')
        )
        loggers.append(wandb_logger)

    data_module = DataModule(cfg)
    learner = Learner(cfg)
    model = learner.model
    for name, param in model.named_parameters():
        print(f"{name} | {param.shape = }")

    print(cfg.trainer)
    trainer = Trainer(
        **cfg.trainer,
        max_epochs=10,
        devices=[1],
        logger=loggers,
        num_sanity_val_steps=0,
        # limit_train_batches=1,
        # limit_val_batches=1,
        # limit_test_batches=1
        # logger=wandb_logger,
        # strategy=DDPPlugin(find_unused_parameters=False) if cfg.gpus > 1 else None
    )

    # trainer.test(learner, datamodule=data_module)
    trainer.validate(learner, verbose=True, datamodule=data_module)
    # trainer.fit(learner, datamodule=data_module)


if __name__ == "__main__":
    main()
