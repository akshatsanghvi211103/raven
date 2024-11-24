import os
import socket
import csv
import time
import wandb
from pytorch_lightning import LightningModule
import torch
from torch.optim.lr_scheduler import StepLR
import torchaudio

from espnet.asr.asr_utils import add_results_to_json, torch_load
from espnet.nets.batch_beam_search import BatchBeamSearch
from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E
from espnet.nets.pytorch_backend.lm.transformer import TransformerLM
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from espnet.nets.scorers.length_bonus import LengthBonus
from metrics import WER
from utils import ids_to_str, set_requires_grad, UNIGRAM1000_LIST

def compute_word_level_distance(seq1, seq2):
    return torchaudio.functional.edit_distance(seq1.lower().split(), seq2.lower().split())

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_weight_norms(model):
    module_dict = {
        'frontend3D': model.encoder.frontend.frontend3D,
        'frontend': model.encoder.frontend,
        'encoders': model.encoder.encoders,
        'encoder': model.encoder,
        'decoder': model.decoder,
        'ctc': model.ctc,
        'model': model
    }

    weight_norm_dict = {}
    for module_name in module_dict:
        num_params = 0
        params_sum = 0
        weight_norm = 0
        model_module = module_dict[module_name]

        for name, param in model_module.named_parameters():
            params_sum += torch.sum(param * param)
            num_params += param.numel()
        
        if num_params:
            weight_norm = torch.sqrt(params_sum)/num_params 
            weight_norm = weight_norm.item()
        
        # print(f"{module_name} | {num_params = } | {math.sqrt(params_sum) = }")
        weight_norm_dict[f"{module_name}_weight_norm"] = weight_norm * 1e6
    
    return weight_norm_dict

def get_grad_norms(model):
    module_dict = {
        'frontend3D': model.encoder.frontend.frontend3D,
        'frontend': model.encoder.frontend,
        'encoders': model.encoder.encoders,
        'encoder': model.encoder,
        'decoder': model.decoder,
        'ctc': model.ctc,
        'model': model
    }

    grad_norms_dict = {}
    for module_name in module_dict:
        num_params = 0 # only considering params with non None grads
        grads_sum = 0
        grad_norm = 0
        model_module = module_dict[module_name]

        for name, param in model_module.named_parameters():
            grad = param.grad
            if grad is not None:
                grads_sum += torch.sum(grad * grad)
                num_params += grad.numel()

        if num_params:
            grad_norm = torch.sqrt(grads_sum)/num_params 
            grad_norm = grad_norm.item()
        
        # print(f"{module_name} | {num_params = } | {math.sqrt(grads_sum) = }")
        grad_norms_dict[f"{module_name}_grad_norm"] = grad_norm * 1e6
    
    return grad_norms_dict

def get_lr(opt):
    return opt.param_groups[0]['lr']

def print_stats(stats_dict):
    print(f"\n{20*'-'}STATS{20*'-'}")
    for name, value in stats_dict.items():
        print(f"{name}: {value}")
    print(f"{50*'-'}\n")

class Learner(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg
        if self.cfg.data.modality == "audio":
            self.backbone_args = self.cfg.model.audio_backbone
        elif self.cfg.data.modality == "video":
            self.backbone_args = self.cfg.model.visual_backbone
        else:
            raise NotImplementedError
        self.model = self.load_model()

        self.ignore_id = -1

        self.beam_search = self.get_beam_search(self.model)
        self.wer = WER()

        self.log_dir = None
        self.epoch_loss = 0.0
        self.epoch_acc = 0.0
        self.epoch_loss_ctc = 0.0
        self.epoch_loss_att = 0.0
        self.epoch_size = 0.0
        self.print_every = 100
        self.cur_train_step = 0
        self.result_data = []
        self.speaker = self.cfg.speaker
        self.finetune_type = self.cfg.finetune
        self.hostname = socket.gethostname()

        if self.global_rank == 0:
            self.train_epoch_st = 0
            self.train_epoch_time = 0
            self.val_wer_results = []
            self.test_wer_results = []

            self.best_val_epoch = -1
            self.best_test_epoch = -1
            self.best_val_wer = 100
            self.test_wer_at_val = 100
            self.best_test_wer = 100
            self.val_wer_at_test = 100
            self.epoch_time_meter = AverageMeter()

    def load_model(self):
        if self.cfg.data.labels_type == "unigram1000":
            odim = len(UNIGRAM1000_LIST)
        else:
            raise NotImplementedError

        model = E2E(odim, self.backbone_args)

        if self.cfg.model.pretrained_model_path:
            print("Load pretrained model weights")
            ckpt = torch.load(
                self.cfg.model.pretrained_model_path,
                map_location=lambda storage, loc: storage,
            )
            model.load_state_dict(ckpt)

        return model

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW([{"name": "model", "params": self.model.parameters(), "lr": self.cfg.optimizer.lr}], weight_decay=self.cfg.optimizer.weight_decay, betas=(0.9, 0.98))
        # scheduler = WarmupCosineScheduler(optimizer, self.cfg.optimizer.warmup_epochs, self.cfg.trainer.max_epochs, len(self.trainer.datamodule.train_dataloader()))
        # scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
        scheduler = {"scheduler": scheduler, "interval": "epoch", "frequency": 1}
        return [optimizer], [scheduler]
        # return [optimizer]

    def get_beam_search(self, model):
        if getattr(self.cfg.data, "labels_type", "char") == "unigram1000":
            token_list = UNIGRAM1000_LIST
        else:
            raise NotImplementedError
        odim = len(token_list)
        self.token_list = token_list

        scorers = model.scorers()

        if self.cfg.decode.lm_weight and self.cfg.model.pretrained_lm_path:
            lm = TransformerLM(len(token_list), self.cfg.model.language_model)
            set_requires_grad(lm, False)
            print("Load pretrained language model weights")
            torch_load(self.cfg.model.pretrained_lm_path, lm)
        else:
            lm = None

        scorers["lm"] = lm
        scorers["length_bonus"] = LengthBonus(len(token_list))

        weights = dict(
            decoder=1.0 - self.cfg.decode.ctc_weight,
            ctc=self.cfg.decode.ctc_weight,
            lm=self.cfg.decode.lm_weight,
            length_bonus=self.cfg.decode.penalty,
        )
        beam_search = BatchBeamSearch(
            beam_size=self.cfg.decode.beam_size,
            vocab_size=len(token_list),
            weights=weights,
            scorers=scorers,
            sos=odim - 1,
            eos=odim - 1,
            token_list=token_list,
            pre_beam_score_key=None if self.cfg.decode.ctc_weight == 1.0 else "decoder",
        )

        return beam_search

    def forward(self, model, data, padding_mask, lengths, label):
        return model(data, padding_mask, lengths, label=label)

    def calculate_wer(self, data, padding_mask, labels):
        labels = labels.squeeze(1)
        data = data.squeeze(1)
        padding_mask = padding_mask
        # print(f"calc: {labels.shape = } | {data.shape = }")
        for idx, (vid, label, mask) in enumerate(zip(data, labels, padding_mask)):
            x = vid[mask].unsqueeze(0)
            feat, _ = self.model.encoder(x, None)
            # print(f"calc: {feat.shape = } | {feat.device = }")

            if isinstance(self.beam_search, BatchBeamSearch):
                nbest_hyps = self.beam_search(
                    x=feat.squeeze(0),
                    maxlenratio=self.cfg.decode.maxlenratio,
                    minlenratio=self.cfg.decode.minlenratio,
                )
            else:
                raise NotImplementedError
            
            # print(f"calc: {type(nbest_hyps) = }")

            nbest_hyps = [h.asdict() for h in nbest_hyps[: min(len(nbest_hyps), 1)]]
            transcription = add_results_to_json(nbest_hyps, self.token_list)
            transcription = transcription.replace("<eos>", "")

            label = label[label != self.ignore_id]
            groundtruth = ids_to_str(label, self.token_list)

            groundtruth = groundtruth.replace("▁", " ").strip()
            transcription = transcription.replace("▁", " ").strip()

            # self.wer.update(transcription, groundtruth)
            # print(f"{self.wer.compute() = }")
            return groundtruth, transcription

    def test_step(self, data, batch_idx):
        # print(f"{data.keys() = } | {batch_idx = }")
        # print(f"{data['data_lengths'] = }")
        # print(f"{data['data'].shape = }")

        lengths = torch.tensor(data["data_lengths"], device=data["data"].device)
        padding_mask = make_non_pad_mask(lengths).to(lengths.device) # (1, T)

        # print(f"{lengths = }")
        self.calculate_wer(data["data"], padding_mask, data["label"])

    def on_validation_epoch_start(self):
        self.num_val_loaders = len(self.trainer.val_dataloaders)
        self.total_length = [0 for i in range(self.num_val_loaders)]
        self.total_edit_distance = [0 for i in range(self.num_val_loaders)]
        self.result_data = [[] for i in range(self.num_val_loaders)]

        # Wrote the column names to the results csv file (only from global rank = 0)
        if self.loggers:
            results_dir = os.path.join(self.loggers[0].log_dir, f"results")
            self.results_dir = results_dir
            self.results_filepaths = ['' for idx in range(self.num_val_loaders)]

            # Making the results.csv files to analyse the generated output text
            os.makedirs(results_dir, exist_ok=True)
            for i in range(self.num_val_loaders):
                if i == 0:
                    results_filename = f"test_results_epoch{self.current_epoch}.csv"
                else:
                    results_filename = f"val{i}_results_epoch{self.current_epoch}.csv"

                results_fp = os.path.join(results_dir, results_filename)
                self.results_filepaths[i] = results_fp

                # Wrote the column names to the results csv file (only from global rank = 0)
                if self.global_rank == 0:
                    row_names = [
                        "Index",
                        "Video Path",
                        "Ground Truth Text",
                        "Predicted Text",
                        "Length",
                        "Word Distance",
                        "WER",
                        "Total Length",
                        "Total Word Distance",
                        "Final WER"
                    ]
                    with open(self.results_filepaths[i], mode='w') as file:
                        writer = csv.writer(file, delimiter=',')
                        writer.writerow(row_names)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # print(f"{data.keys() = } | {batch_idx = }")
        # print(f"{data['data_lengths'] = }")
        # print(f"{data['data'].shape = }")

        idx = dataloader_idx
        lengths = torch.tensor(batch["data_lengths"], device=batch["data"].device)
        padding_mask = make_non_pad_mask(lengths).to(lengths.device) # (1, T)

        # print(f"{lengths = }")
        gt_text, pred_text = self.calculate_wer(batch["data"], padding_mask, batch["label"])

        # Caluclating word distance
        word_distance = compute_word_level_distance(gt_text, pred_text)
        gt_len = len(pred_text.split())
        self.total_edit_distance[idx] += word_distance
        self.total_length[idx] += gt_len
        wer = self.total_edit_distance[idx]/self.total_length[idx]

        video_path = batch['video_paths'][0]
        data_id = batch['ids'][0]

        data_id = batch_idx
        # Printing Stats for this datapoint
        if self.cfg.verbose:
            print(f"\n{'*' * 70}"
                  + f"\n{dataloader_idx = }"
                #   + f"\n{data_id} video_path: {video_path}"
                  + f"\n{data_id} GT: {gt_text}"
                  + f"\n{data_id} Pred: {pred_text}"
                  + f"\n{data_id} dist = {word_distance}, len: {len(gt_text.split())}"
                  + f"\n{data_id} Sentence WER: {word_distance/len(gt_text.split())}"
                  + f"\n{data_id} Cur WER: {wer}"
                  + f"\n{'*' * 70}") 
        
        # Results.csv data for this epoch
        if self.loggers and self.result_data:
            gt_len = gt_len
            if gt_len == 0:
                gt_len = 0.1
            wd = word_distance
            data = [
                data_id,
                video_path,
                gt_text,
                pred_text,
                gt_len,
                wd,
                wd/gt_len,
                self.total_length[idx],
                self.total_edit_distance[idx],
                wer
            ]
            self.result_data[idx].append(data)

    def training_step(self, batch, batch_idx):
        B, C, T, H, W = batch['data'].shape
        lengths = torch.tensor(batch["data_lengths"], device=batch["data"].device)
        padding_mask = make_non_pad_mask(lengths).to(lengths.device) # (1, T)
        label = batch['label']

        inp = batch['data'].squeeze(1) # (B, T, H, W)
        inp = inp[padding_mask].unsqueeze(0) # (B, C=1, T, H, W)
        # print(f"{inp.shape = } | {lengths.shape = } | {padding_mask.shape = } | {label.shape = }")

        torch.cuda.empty_cache()
        out, loss, loss_ctc, loss_att, acc = self.forward(self.model, inp, None, lengths, label)
        # print(f"{loss = } | {loss_ctc = } | {loss_att = } | {acc = }")
        
        batch_size = 1
        self.epoch_loss += loss.item() * batch_size
        self.epoch_loss_ctc += loss_ctc.item() * batch_size
        self.epoch_loss_att += loss_att.item() * batch_size
        self.epoch_acc += acc * batch_size
        self.epoch_size += batch_size
        opt = self.optimizers()
        # self.cur_lr = get_lr(opt)
        if self.global_rank == 0:
            # print(f"{self.cur_train_step = } | {self.cur_lr = }")
            lr_dict = {'train_step': self.cur_train_step}
            # if self.cfg.wandb:
            #     wandb.log(lr_dict)

        self.cur_train_step += 1
        return loss
    
    def on_train_batch_start(self, batch, batch_idx):
        weight_norms_dict = get_weight_norms(self.model)
        if self.global_rank == 0:
            if self.cfg.wandb:
                wandb.log(weight_norms_dict)
            # self.log_dict(weight_norms_dict, on_step=True, on_epoch=False,
            #               logger=True)
        
        return super().on_train_batch_start(batch, batch_idx)

    def on_train_batch_end(self, out, batch, batch_idx):
        grad_norms_dict = get_grad_norms(self.model)
        if self.global_rank == 0:
            if self.cfg.wandb:
                wandb.log(grad_norms_dict)
            
            # TODO : This only logs at intervals of 50 for some reason (figure it out)
            # self.log_dict(grad_norms_dict, on_step=True, on_epoch=False,
            #               logger=True)
        
        return super().on_train_batch_end(out, batch, batch_idx)

    def on_train_epoch_start(self):
        if self.log_dir is None:
            if self.loggers:
                self.log_dir = self.loggers[0].log_dir
                print(f"{self.log_dir = }")
                if self.cfg.wandb and self.global_rank == 0:
                    wandb.log({'log_dir': self.log_dir})

        self.epoch_loss = torch.tensor(0.0).to(self.device)
        self.epoch_acc = torch.tensor(0.0).to(self.device)
        self.epoch_loss_ctc = torch.tensor(0.0).to(self.device)
        self.epoch_loss_att = torch.tensor(0.0).to(self.device)
        self.epoch_size = torch.tensor(0.0).to(self.device)
        if self.global_rank == 0:
            self.train_epoch_st = time.time()

        return super().on_train_epoch_start()

    def on_train_epoch_end(self) -> None:
        # Gathering the values across GPUs
        self.epoch_loss = self.all_gather(self.epoch_loss, sync_grads=False).sum()
        self.epoch_loss_ctc = self.all_gather(self.epoch_loss_ctc, sync_grads=False).sum()
        self.epoch_loss_att = self.all_gather(self.epoch_loss_att, sync_grads=False).sum()
        self.epoch_acc = self.all_gather(self.epoch_acc, sync_grads=False).sum()
        self.epoch_size = self.all_gather(self.epoch_size, sync_grads=False).sum()

        # Logging from process with global rank = 0
        if self.global_rank == 0:
            self.train_epoch_time = time.time() - self.train_epoch_st
            self.epoch_time_meter.update(self.train_epoch_time)

            log_dict = {
                "train_loss_epoch": self.epoch_loss/self.epoch_size,
                "train_loss_ctc_epoch": self.epoch_loss_ctc/self.epoch_size,
                "train_loss_att_epoch": self.epoch_loss_att/self.epoch_size,
                "train_decoder_acc_epoch": self.epoch_acc/self.epoch_size,
                "train_epoch_time": self.train_epoch_time,
                "avg_train_epoch_time": self.epoch_time_meter.avg,
                "total_train_epoch_time": self.epoch_time_meter.sum,
                "epoch": self.current_epoch
            }
            self.log_dict(log_dict, logger=True)
            print_stats(log_dict)

        return super().on_train_epoch_end()


    def on_validation_epoch_end(self):
        if self.global_rank == 0:
            for idx in range(self.num_val_loaders):
                if idx == 0:
                    val_type = "test"
                else:
                    val_type = f"val{idx}"

                wer = self.total_edit_distance[idx]/self.total_length[idx]
                log_dict = {'epoch': self.current_epoch}
                if idx == 0:
                    log_dict['wer_test_epoch'] = wer
                    self.test_wer_results.append(wer)
                else:
                    log_dict[f"wer_val{idx}_epoch"] = wer
                    self.val_wer_results.append(wer)

                if self.cfg.wandb:
                    wandb.log(log_dict)
                else:
                    self.log_dict(log_dict, logger=True)
                print_stats(log_dict)

                # Write the csv results file for the corresponding val dataloader
                if self.loggers and self.result_data[idx]:
                    with open(self.results_filepaths[idx], mode='a') as file:
                        writer = csv.writer(file, delimiter=',')
                        writer.writerows(self.result_data[idx])
                        print(f"{self.current_epoch = } Successfully written the results data at {self.results_filepaths[idx]}")
        
        # calculate the best val wer epoch
        if self.global_rank == 0 and self.trainer.state.fn == "fit":
            val_wer = self.val_wer_results[-1]
            test_wer = self.test_wer_results[-1]
            if self.best_val_epoch == -1 or val_wer < self.best_val_wer:
                self.best_val_epoch = self.current_epoch
                self.best_val_wer = val_wer
                self.test_wer_at_val = test_wer

            if self.best_test_epoch == -1 or test_wer < self.best_test_wer:
                self.best_test_epoch = self.current_epoch
                self.best_test_wer = test_wer
                self.val_wer_at_test = val_wer

    def on_train_end(self):
        # Logging the Best Val epoch, Val wer, Test wer at end of training
        if self.global_rank == 0:
            log_data = {
                'speaker': self.speaker,
                'best_val_epoch': self.best_val_epoch,
                'val_wer_best_val_epoch': self.best_val_wer,
                'test_wer_best_val_epoch': self.test_wer_at_val,
                'best_test_epoch': self.best_test_epoch,
                'val_wer_best_test_epoch': self.val_wer_at_test,
                'best_test_wer': self.best_test_wer
            }
            print_stats(log_data)

            if self.cfg.wandb:
                log_data.update({'wandb_id': wandb.run.id, 'wandb_name': wandb.run.name, 'wandb_url': wandb.run.url})
                wandb.log(log_data)

                row_data = [
                    self.speaker,
                    self.finetune_type,
                    self.hostname,
                    wandb.run.id,
                    wandb.run.name,
                    wandb.run.url,
                    self.best_val_epoch,
                    self.best_val_wer,
                    self.test_wer_at_val,
                    self.best_test_epoch,
                    self.val_wer_at_test,
                    self.best_test_wer
                ]

                # writing to the csv file
                csv_file_path = '/home2/akshat/raven/accented_speakers.csv'
                with open(csv_file_path, mode='a') as file:
                    writer = csv.writer(file, delimiter=',')
                    # writer = csv.DictWriter(file, fieldnames=log_data.keys())
                    writer.writerow(row_data)  # Append the row

    def on_test_epoch_end(self, outputs):
        wer = self.wer.compute()
        print(wer)
        self.log("wer", wer)
        self.wer.reset()
