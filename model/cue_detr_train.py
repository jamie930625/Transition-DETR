# dataset
from transformers import DetrConfig

# training and logging
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateFinder
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
import pytorch_lightning as pl
import wandb

import os, glob, argparse

# project modules
from cue_detr_data import CueTrainModule
from cue_detr_model import CuePointDetr, CuePredictionLogger


def train_cue_points(args):
    if args.gpus > 1:
        pl.seed_everything(1)

    # Setup wandb logger
    logger = setup_logger(args)
    print(f'\nStarting run {args.run_name}...\n')

    data_module = CueTrainModule(
        args.image_dir,
        args.batch_size,
        image_processor='facebook/detr-resnet-50',
        reuse_slice=args.slice_reuse,
        w_slice=args.slice_width,
        w_bbox=args.box_width,
        box_area=args.box_area,
        pipeline_test=args.test_pl,
        )

    # fill config if we start with random weights
    detr_config = None
    if args.bb_pt or args.no_pt:
        detr_config = DetrConfig(
            num_labels=2, # 🚀 [核心修改] 這裡改成 2 類別
            num_queries=args.num_queries,
            use_pretrained_backbone=args.bb_pt,
            auxiliary_loss=args.auxiliary_loss,
            class_cost=args.class_cost,
            bbox_cost=args.bbox_cost,
            giou_cost=args.giou_cost,
            bbox_loss_coefficient=args.bbox_loss_coefficient,
            giou_loss_coefficient=args.giou_loss_coefficient,
            eos_coefficient=args.eos_coefficient,
            )
        
    detr = CuePointDetr(
        lr=args.lr, lr_backbone=args.lr_bb,
        weight_decay=args.weight_decay,
        detr_checkpoint=args.finetune if args.fully_pt else None,
        config=detr_config,  # is None if we just use finetune
        )
    
    callbacks = rank_zero_only(setup_callbacks)(args, data_module)
        
    # TRAINER
    print('\nSetting up trainer...')
    if args.gpus > 1:
        trainer = Trainer(
            strategy='ddp',
            devices=args.gpus,
            accelerator='gpu',
            max_epochs=args.epochs,
            gradient_clip_val=args.grad_clip,
            logger=logger,
            log_every_n_steps=1,
            default_root_dir=args.checkpoint_dir,
            callbacks=callbacks,
            )
    else:
        trainer = Trainer(
            devices=1,
            accelerator='gpu',
            max_epochs=args.epochs,
            gradient_clip_val=args.grad_clip,
            logger=logger,
            log_every_n_steps=1,
            default_root_dir=args.checkpoint_dir,
            callbacks=callbacks,
            )
    assert trainer is not None, 'Trainer initialization failed.'
        
    print('\nStarting training...')
    trainer.fit(detr, data_module, ckpt_path=args.resume_from)
    
    print('\nSaving Model...')

    # save checkpoint and hf model (as alternative)
    dir = get_checkpoint_dir(args)
    trainer.save_checkpoint(f'{dir}{args.epochs}-epoch.ckpt')
    detr.model.save_pretrained(dir)

    wandb.finish()
    print('\nDone...')


'''
==========================================
            HELPER FUNCTIONS
==========================================
'''
def update_logger(logger, log: dict):
    logger.experiment.config.update(log)    


def setup_logger(args):
    wandb_logger = WandbLogger(
        project=args.wandb_project,
        entity=args.wandb_user,
        name=args.run_name,
        save_dir=args.wandb_run_dir)
    
    rank_zero_only(update_logger)(
        wandb_logger,
        {
            'learning_rate': args.lr,
            'learning_rate_backbone': args.lr_bb,
            'weight_decay': args.weight_decay,
            'n_epochs': args.epochs,
            'batch_size': args.batch_size,
            'grad_clip_value': args.grad_clip,
            'gpus_used': args.gpus,
        })

    return wandb_logger


def get_checkpoint_dir(args):
    dir = f'{args.checkpoint_dir}{args.run_name}/'
    os.makedirs(dir, exist_ok=True) # 🚀 修正：防止多卡訓練時創建資料夾衝突
    return dir


def setup_callbacks(args, data_module):
    callbacks = []
    
    # 🚀 [核心修改] 救援 Meeting 的無情存檔機器
    checkpoint_callback = ModelCheckpoint(
        save_top_k=-1,                 # 保留所有歷史存檔
        save_last=True,                # 隨時更新 latest
        every_n_train_steps=500,       # 每 500 步強制吐出一個 .ckpt 裝備
        dirpath=get_checkpoint_dir(args),
        filename="demo-step-{step:06d}",
        )
    callbacks.append(checkpoint_callback)
    
    # Image logger callback
    data_module.prepare_data()
    data_module.setup()
    image_logger_callback = CuePredictionLogger(args.slice_idx, data_module.val_dataset)
    callbacks.append(image_logger_callback)

    # LR finder callback
    if args.lr_finder:
        lr_finder = LearningRateFinder(min_lr=1e-7, max_lr=1e-4)
        callbacks.append(lr_finder)

    return callbacks



'''
==========================================
              MAIN FUNCTION
==========================================
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run cue point detection model.')

    # Model Run Type (only one possible at a time)
    run_mode = parser.add_mutually_exclusive_group(required=True)
    run_mode.add_argument('--train', action='store_true')

    # toggle pipeline testing using a subset of the data
    parser.add_argument('--test_pl', action='store_true', help='Toggle to test the pipeline on a subset of the data.')

    # Detr Configuration
    detr_version = parser.add_mutually_exclusive_group(required=True)
    detr_version.add_argument('--fully_pt', action='store_true', help='Initialize DETR with pretrained transformer and pretrained backbone weights.')
    detr_version.add_argument('--bb_pt', action='store_true', help='Initialize DETR with random transformer weights and pretrained backbone weights.')
    detr_version.add_argument('--no_pt', action='store_true', help='Initialize DETR with random transformer and backbone weights.')

    # Hyperparameters
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_bb', default=1e-5, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--grad_clip', default=0.1, type=float)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--gpus', default=4, type=int) # 🚀 [核心修改] 預設改為 4，解放你的 3090 算力！

    # Directories & Storage
    parser.add_argument('--image_dir', type=str, required=True, help='Directory containing training images.')
    parser.add_argument('--checkpoint_dir', type=str, required=True, help='Directory to save model checkpoints.')
    parser.add_argument('--finetune', type=str, help='Image processor to finetune (default: facebook/detr-resnet-50).')
    parser.add_argument('--resume_from', type=str, help='Checkpoint to resume training from.')

    # WandB
    parser.add_argument('--wandb_user', type=str)
    parser.add_argument('--wandb_project', type=str)
    parser.add_argument('--wandb_run_dir', type=str, help='Directory to save wandb run logs.')
    parser.add_argument('--run_name', type=str)
    parser.add_argument('--run_nr', type=int)
    parser.add_argument('--slice_idx', default=100, type=int, help='Dataset index for logged wandb image.')

    # Callbacks
    parser.add_argument('--lr_finder', action='store_true', help='Toggle to run lr finder.')

    # Box params
    parser.add_argument('--box_area', type=int)
    parser.add_argument('--box_width', type=int)
    parser.add_argument('--slice_width', type=int)
    parser.add_argument('--slice_reuse', action='store_true', help='Toggle to use the same slice around cue points.')

    # DETR config params
    parser.add_argument('--num_queries', default=20, type=int, help='Number of object queries per image (Modified to 20 for single point precision).')
    parser.add_argument('--auxiliary_loss', action='store_true', help='Toggle to use auxiliary decoding losses.')
    parser.add_argument('--class_cost', default=1, type=float, help='Relative weight of the classification error.')
    parser.add_argument('--bbox_cost', default=5, type=float, help='Relative weight of the L1 error of the bounding box.')
    parser.add_argument('--giou_cost', default=2, type=float, help='Relative weight of the generalized IoU loss.')
    parser.add_argument('--bbox_loss_coefficient', default=8, type=float, help='Relative weight of the L1 bounding box loss (Modified to 8 to penalize time shift).')
    parser.add_argument('--giou_loss_coefficient', default=1, type=float, help='Relative weight of the generalized IoU loss (Modified to 1 to reduce box shape penalty).')
    parser.add_argument('--eos_coefficient', default=0.1, type=float, help='Relative classification weight of the `no-object` class.')

    args = parser.parse_args()

    # Logging
    run_name = args.run_name if args.run_name else ""
    run_nr = args.run_nr if args.run_nr is not None else len(glob.glob(f'{args.checkpoint_dir}{run_name}_*'))
    run_name = f'{run_name}_{run_nr}'
    args.run_name = run_name

    if args.test_pl:
        print('\n===========================\nARGUMENTS PASSED:\n')
        for arg in vars(args):
            print(arg, getattr(args, arg))
        print('\n===========================\n')

    if args.train:
        train_cue_points(args)
