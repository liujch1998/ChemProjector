import pickle
from typing import Any, Optional, Union

import numpy as np
import pytorch_lightning as pl
import torch
import transformers
import os
import wandb  # Logging generation examples as W&B table
import traceback
from omegaconf import OmegaConf

from chemprojector.chem.featurize import COR_START, COR_END, COR_MOL_START, COR_MOL_END, _COR_RXN_OFFSET, BOS, PAD, EOS, tokenizer
from chemprojector.chem.fpindex import FingerprintIndex
from chemprojector.chem.matrix import ReactantReactionMatrix
from chemprojector.data.common import ProjectionBatch, draw_batch, draw_input, draw_output
from chemprojector.utils.train import get_optimizer, get_scheduler, sum_weighted_losses

from .chemprojector import ChemProjector, draw_generation_results


class ChemProjectorWrapper(pl.LightningModule):
    def __init__(self, config, base_model: str, args: dict | None = None):
        super().__init__()
        if config.version != 2:
            raise ValueError("Only version 2 is supported")
        self.save_hyperparameters(
            {
                "config": OmegaConf.to_container(config),
                "args": args or {},
            }
        )
        if base_model == "llama3.1-8b-inst":
            model_name = 'meta-llama/Llama-3.1-8B-Instruct'
        elif base_model == "llama3.2-1b-inst":
            model_name = 'meta-llama/Llama-3.2-1B-Instruct'
        else:
            raise ValueError(f"Unsupported base model: {base_model}")
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name,
            token=os.environ['HF_TOKEN'],
        )

    @property
    def config(self):
        return OmegaConf.create(self.hparams["config"])

    @property
    def args(self):
        return OmegaConf.create(self.hparams.get("args", {}))

    def setup(self, stage: str) -> None:
        super().setup(stage)

        # Load chem data
        with open(self.config.chem.rxn_matrix, "rb") as f:
            self.rxn_matrix: ReactantReactionMatrix = pickle.load(f)

        with open(self.config.chem.fpindex, "rb") as f:
            self.fpindex: FingerprintIndex = pickle.load(f)

    def configure_optimizers(self):
        optimizer = get_optimizer(self.config.train.optimizer, self.model)
        if "scheduler" in self.config.train:
            scheduler = get_scheduler(self.config.train.scheduler, optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": "val/loss",
            }
        return optimizer

    def training_step(self, batch: ProjectionBatch, batch_idx: int):
        loss = self.model(**batch).loss
        # loss_dict, aux_dict = self.model.get_loss_shortcut(batch, warmup=self.current_epoch == 0)
        # loss_sum = sum_weighted_losses(loss_dict, self.config.train.loss_weights)

        self.log("train/loss", loss, on_step=True, prog_bar=True, logger=True)
        # self.log_dict({f"train/loss_{k}": v for k, v in loss_dict.items()}, on_step=True, logger=True)

        # if "fp_select" in aux_dict:
        #     fp_select: torch.Tensor = aux_dict["fp_select"]
        #     fp_ratios: dict[str, float] = {}
        #     for i in range(int(fp_select.max().item()) + 1):
        #         ratio = (fp_select == i).float().mean().nan_to_num(0.0)
        #         fp_ratios[f"fp_select/{i}"] = ratio.item()
        #     self.log_dict(fp_ratios, on_step=True, logger=True)
        return loss

    def validation_step(self, batch: ProjectionBatch, batch_idx: int) -> Any:
        # print('input_ids:', batch['input_ids'].size(), batch['input_ids'].tolist())
        # print('attention_mask:', batch['attention_mask'].size(), batch['attention_mask'].tolist())
        # print('labels:', batch['labels'].size(), batch['labels'].tolist())
        # exit()

        loss = self.model(**batch).loss
        # loss_dict, _ = self.model.get_loss_shortcut(batch)
        # loss_weight = self.config.train.get("val_loss_weights", self.config.train.loss_weights)
        # loss_sum = sum_weighted_losses(loss_dict, loss_weight)

        self.log("val/loss", loss, on_step=False, prog_bar=True, logger=True, sync_dist=True)
        # self.log_dict({f"val/loss_{k}": v for k, v in loss_dict.items()}, on_step=False, logger=True, sync_dist=True)

        # Generate
        if self.args.get("visualize", True) and batch_idx == 0:
            max_smiles_len = 192
            max_cor_len = 1024
            gen_cfg = transformers.GenerationConfig(
                max_new_tokens=max_cor_len,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=PAD,
                eos_token_id=EOS,
            )
            input_ids = batch['input_ids'][:, :max_smiles_len]
            gold_output_ids = batch['input_ids'][:, max_smiles_len:]
            with torch.inference_mode():
                model_output_ids = self.model.generate(input_ids, generation_config=gen_cfg)
            model_output_ids = model_output_ids[:, input_ids.size(1):]
            local_rows: list[dict[str, Any]] = []

            for b in range(len(input_ids)):
                row_dict = {
                    'input_ids': ' '.join(map(str, input_ids[b].tolist())),
                    'gold_output_ids': ' '.join(map(str, gold_output_ids[b].tolist())),
                    'model_output_ids': ' '.join(map(str, model_output_ids[b].tolist())),
                    'input_str': '',
                    'gold_output_str': '',
                    'model_output_str': '',
                    'input_im': None,
                    'gold_output_im': None,
                    'model_output_im': None,
                }

                try:
                    input_str = ''
                    bos_index = (input_ids[b] == BOS).nonzero()[0, 0].item()
                    for i in range(bos_index + 1, input_ids.size(1)):
                        if input_ids[b, i] == COR_MOL_START:
                            input_str += '[MOL:START] '
                        elif input_ids[b, i] == COR_MOL_END:
                            input_str += ' [MOL:END]'
                        else:
                            input_str += tokenizer.decode(input_ids[b, i]).lstrip(' ')
                    row_dict['input_str'] = input_str
                except Exception:
                    row_dict['input_str'] = 'invalid'

                try:
                    row_dict['input_im'] = wandb.Image(draw_input(input_ids[b][bos_index + 1:].tolist()))
                except Exception as e:
                    pass
                    # row_dict['input_im'] = f'invalid: {e}\n{traceback.format_exc()}'

                try:
                    gold_output_str = ''
                    gold_eos_index = (gold_output_ids[b] == EOS).nonzero()[0, 0].item()
                    assert gold_output_ids[b, 0] == COR_START
                    assert gold_output_ids[b, gold_eos_index - 1] == COR_END
                    for i in range(gold_eos_index):
                        if gold_output_ids[b, i] == COR_START:
                            gold_output_str += '[COR:START]'
                        elif gold_output_ids[b, i] == COR_END:
                            gold_output_str += ' [COR:END]'
                        elif gold_output_ids[b, i] == COR_MOL_START:
                            gold_output_str += ' [MOL:START] '
                        elif gold_output_ids[b, i] == COR_MOL_END:
                            gold_output_str += ' [MOL:END]'
                        elif _COR_RXN_OFFSET + 1 <= gold_output_ids[b, i]:
                            gold_output_str += f' [RXN:{gold_output_ids[b, i] - _COR_RXN_OFFSET}]'
                        else:
                            gold_output_str += tokenizer.decode(gold_output_ids[b, i]).lstrip(' ')
                    row_dict['gold_output_str'] = gold_output_str
                except Exception as e:
                    row_dict['gold_output_str'] = f'invalid'

                try:
                    row_dict['gold_output_im'] = wandb.Image(draw_output(gold_output_ids[b][:gold_eos_index].tolist(), self.rxn_matrix))
                except Exception as e:
                    pass
                    # row_dict['gold_output_im'] = f'invalid: {e}\n{traceback.format_exc()}'

                try:
                    model_output_str = ''
                    model_eos_index = (model_output_ids[b] == EOS).nonzero()[0, 0].item()
                    assert model_output_ids[b, 0] == COR_START
                    assert model_output_ids[b, model_eos_index - 1] == COR_END
                    for i in range(model_eos_index):
                        if model_output_ids[b, i] == COR_START:
                            model_output_str += '[COR:START]'
                        elif model_output_ids[b, i] == COR_END:
                            model_output_str += ' [COR:END]'
                        elif model_output_ids[b, i] == COR_MOL_START:
                            model_output_str += ' [MOL:START] '
                        elif model_output_ids[b, i] == COR_MOL_END:
                            model_output_str += ' [MOL:END]'
                        elif _COR_RXN_OFFSET + 1 <= model_output_ids[b, i]:
                            model_output_str += f' [RXN:{model_output_ids[b, i] - _COR_RXN_OFFSET}]'
                        else:
                            model_output_str += tokenizer.decode(model_output_ids[b, i]).lstrip(' ')
                    row_dict['model_output_str'] = model_output_str
                except Exception:
                    row_dict['model_output_str'] = 'invalid'

                try:
                    row_dict['model_output_im'] = wandb.Image(draw_output(model_output_ids[b][:model_eos_index].tolist(), self.rxn_matrix))
                except Exception as e:
                    pass
                    # row_dict['model_output_im'] = f'invalid: {e}\n{traceback.format_exc()}'

                local_rows.append(row_dict)

            aggregated_rows = local_rows
            if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
                gathered: list[list[dict[str, Any]]] = [None] * torch.distributed.get_world_size()  # type: ignore
                torch.distributed.all_gather_object(gathered, local_rows)
                aggregated_rows = [row for sublist in gathered if sublist is not None for row in sublist]

            is_rank_zero = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
            if is_rank_zero and isinstance(self.logger, pl.loggers.WandbLogger):
                columns = ['input_ids', 'gold_output_ids', 'model_output_ids', 'input_str', 'gold_output_str', 'model_output_str', 'input_im', 'gold_output_im', 'model_output_im']
                table = wandb.Table(columns=columns)
                for row in aggregated_rows:
                    table.add_data(row['input_ids'], row['gold_output_ids'], row['model_output_ids'], row['input_str'], row['gold_output_str'], row['model_output_str'], row['input_im'], row['gold_output_im'], row['model_output_im'])

                table_name = f"val_generations/epoch{self.current_epoch}_batch{batch_idx}"
                self.logger.experiment.log({table_name: table}, commit=False)

            # result = self.model.generate_without_stack(batch=batch, rxn_matrix=self.rxn_matrix, fpindex=self.fpindex)
            # images_gen = draw_generation_results(result)
            # images_ref = draw_batch(batch)
            # if self.logger is not None:
            #     tb_logger = self.logger.experiment
            #     for i, (image_gen, image_ref) in enumerate(zip(images_gen, images_ref)):
            #         tb_logger.add_images(
            #             f"val/{i}_generate",
            #             np.array(image_gen) / 255,
            #             self.current_epoch,
            #             dataformats="HWC",
            #         )
            #         tb_logger.add_images(
            #             f"val/{i}_reference",
            #             np.array(image_ref) / 255,
            #             self.current_epoch,
            #             dataformats="HWC",
            #         )

        return loss

    # def configure_gradient_clipping(
    #     self,
    #     optimizer,
    #     gradient_clip_val: Optional[Union[int, float]] = None,
    #     gradient_clip_algorithm: Optional[str] = None,
    # ):
    #     assert gradient_clip_algorithm in ('norm', None), gradient_clip_algorithm
    #     self.model.clip_grad_norm_(gradient_clip_val)
