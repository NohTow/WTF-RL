#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional



import datasets
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np

from multiprocessing import Pool
from nltk.translate.bleu_score import SmoothingFunction

from datasets import load_dataset

import torch
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import LambdaLR
from PIL import Image
from io import BytesIO
import base64
from torchvision.io import ImageReadMode, read_image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

import re
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
# import json
import io
import string

import evaluate
import transformers
from filelock import FileLock
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    AutoFeatureExtractor,
    AutoProcessor,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
    BlipForConditionalGeneration,
    # BlipTokenizer,
    Trainer,
    PreTrainedModel,
    BlipConfig,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.22.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/summarization/requirements.txt")

logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)

# A list of all multilingual tokenizer which require lang attribute.
MULTILINGUAL_TOKENIZERS = [MBartTokenizer, MBartTokenizerFast, MBart50Tokenizer, MBart50TokenizerFast]


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    generator_name_or_path: str = field(
        metadata={"help": "Path to pretrained generator or model identifier from huggingface.co/models"}
    )
    discriminator_name_or_path: str = field(
        metadata={"help": "Path to pretrained discriminator or model identifier from huggingface.co/models"}
    )

    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
                "the model's position embeddings."
            )
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    data_dir: Optional[str] = field(default=None, metadata={"help": "The data directory containing input files."})
    image_column: Optional[str] = field(
        default="image_path",
        metadata={"help": "The name of the column in the datasets containing the full image file paths."},
    )
    caption_column: Optional[str] = field(
        default="caption",
        metadata={"help": "The name of the column in the datasets containing the image captions."},
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file (a jsonlines file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file (a jsonlines file)."},
    )

    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )

    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )

    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=512,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."



dataset_name_mapping = {
    "image_caption_dataset.py": ("image_path", "caption"),
}


def get_constant_schedule_with_warmup(optimizer: Optimizer, num_warmup_steps: int, last_epoch: int = -1):
    """
    Create a schedule with a constant learning rate preceded by a warmup period during which the learning rate
    increases linearly between 0 and the initial lr set in the optimizer.
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1.0, num_warmup_steps))
        return 1.0

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)

def scale_grad_value_(parameters, scale_value: float) -> None:
    r"""Scale gradient of an iterable of parameters with specified value.

    Gradients are modified in-place.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        scale_value (float or int): value to scale the gradients.
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    scale_value = float(scale_value)
    for p in filter(lambda p: p.grad is not None, parameters):
        p.grad.mul_(scale_value)
class SelfBleu():
    def __init__(self, test_text='', sample_size=1000, gram=5):
        super().__init__()
        self.name = 'Self-Bleu'
        self.test_data = test_text
        self.gram = gram
        self.sample_size = sample_size
        self.reference = None
        self.is_first = True

    def get_name(self):
        return self.name

    def get_score(self, is_fast=True, ignore=False):
        if ignore:
            return 0
        if self.is_first:
            self.get_reference()
            self.is_first = False
        if is_fast:
            return self.get_bleu_fast()
        return self.get_bleu_parallel()

    def get_reference(self):
        if self.reference is None:
            reference = list()
            for text in self.test_data:
                text = nltk.word_tokenize(text)
                reference.append(text)
            self.reference = reference
            return reference
        else:
            return self.reference

    def get_bleu(self):
        ngram = self.gram
        bleu = list()
        reference = self.get_reference()
        weight = tuple((1. / ngram for _ in range(ngram)))
        for i, hypothesis in enumerate(sef.test_data):
            hypothesis = nltk.word_tokenize(hypothesis)
            bleu.append(nltk.translate.bleu_score.sentence_bleu(reference, hypothesis, weight,
                                                                smoothing_function=SmoothingFunction().method1))
            print(nltk.translate.bleu_score.sentence_bleu(reference, hypothesis, weight,
                                                                smoothing_function=SmoothingFunction().method1))
        return sum(bleu) / len(bleu)

    def calc_bleu(self, reference, hypothesis, weight):
        return nltk.translate.bleu_score.sentence_bleu(reference, hypothesis, weight,
                                                       smoothing_function=SmoothingFunction().method1)

    def get_bleu_fast(self):
        reference = self.get_reference()
        # random.shuffle(reference)
        reference = reference[0:self.sample_size]
        return self.get_bleu_parallel(reference=reference)

    def get_bleu_parallel(self, reference=None):
        ngram = self.gram
        if reference is None:
            reference = self.get_reference()
        weight = tuple((1. / ngram for _ in range(ngram)))
        pool = Pool(os.cpu_count())
        result = list()
        sentence_num = len(reference)
        for index in range(sentence_num):
            hypothesis = reference[index]
            other = reference[:index] + reference[index+1:]
            result.append(pool.apply_async(self.calc_bleu, args=(other, hypothesis, weight)))
        score = 0.0
        cnt = 0
        for i in result:
            score += i.get()
            cnt += 1
        pool.close()
        pool.join()
        return score / cnt


# Model definition
class GCN(PreTrainedModel):
    def __init__(self, config, generator_name_or_path, discriminator_name_or_path, cache_dir, learning_rate, custom_args):
        super(GCN, self).__init__(config)
        self.generator = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    
        self.discriminator = AutoModel.from_pretrained(
        "openai/clip-vit-base-patch32", cache_dir=".cache"
        )
        self.MLP = torch.nn.Sequential(
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 2),
        )
        
        self.MLP.load_state_dict(torch.load("pretrained_disc_scst_repro_27999"))
        self.MLP_optimizer = Adam([
                {'params': self.MLP.parameters()},
        ], lr=7e-5)
        self.tokenizer_BLIP = AutoTokenizer.from_pretrained("Salesforce/blip-image-captioning-large")
        self.tokenizer_CLIP = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32", cache_dir="/srv/tempdd/achaffin/.cache")
        self.BCELoss = torch.nn.BCELoss()
        self.optimizer_gen = (Adam(list(self.generator.parameters()), lr=learning_rate))
        self.scheduler_gen = get_constant_schedule_with_warmup(self.optimizer_gen, 100)
        self.baseline = custom_args.baseline
        self.disc_weight = custom_args.disc_weight

    

# We use torchvision for faster image pre-processing. The transforms are implemented as nn.Module,
# so we jit it to be faster.
class TransformWithAugmentation(torch.nn.Module):
    def __init__(self, image_size, mean, std, n_views=2):
        super().__init__()
        self.n_views = n_views
        self.preprocess = torch.nn.Sequential(
            transforms.Resize((resolution, resolution), interpolation=Image.BICUBIC),
            transforms.Normalize(mean=mean, std=std)
        )
        s=1
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        self.transforms = torch.nn.Sequential(
            transforms.RandomResizedCrop(size=image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply(torch.nn.ModuleList([color_jitter]), p=0.8),
            transforms.RandomGrayscale(p=0.2),
        )
    

    def forward(self, x):
        with torch.no_grad():
            x = self.preprocess(x)
            x = [self.transforms(x) for i in range(self.n_views)]
        return x

class Transform_BLIP(torch.nn.Module):
    def __init__(self, resolution, mean, std):
        super().__init__()
        self.preprocess = torch.nn.Sequential(
            transforms.Resize((resolution, resolution), interpolation=Image.BICUBIC),
            transforms.Normalize(mean=mean, std=std)
        )
        

    def forward(self, x):
        with torch.no_grad():
            x = self.preprocess(x)
        return x

class Transform_CLIP(torch.nn.Module):
    def __init__(self, image_size, mean, std):
        super().__init__()
        self.preprocess = torch.nn.Sequential(
            # transforms.ToTensor(),
            transforms.Resize([image_size], interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean, std),
        )
        

    def forward(self, x):
        with torch.no_grad():
            x = self.preprocess(x)
        return x

def collate_fn_train(examples):
    input_ids = torch.tensor([example["input_ids"] for example in examples], dtype=torch.long)
    pixel_values_BLIP = torch.stack([torch.tensor(example["pixel_values_BLIP"], dtype=torch.float) for example in examples])
    image_clip_embeds = torch.tensor([example["image_clip_embeds"] for example in examples], dtype=torch.float)
    neighbours_image_embeddings = torch.tensor([embeddings for example in examples for embeddings in example["embeddings_neighbours"]], dtype=torch.float)
    decoder_input_ids = torch.tensor([example["decoder_input_ids"] for example in examples], dtype=torch.long)
    attention_mask = torch.tensor([example["attention_mask"] for example in examples], dtype=torch.long)
    row_id = torch.tensor([example["row_id"] for example in examples], dtype=torch.float)
    return {
        "input_ids": input_ids,
        "pixel_values_BLIP": pixel_values_BLIP,
        "image_clip_embeds": image_clip_embeds,
        "neighbours_image_embeddings": neighbours_image_embeddings,
        "decoder_input_ids": decoder_input_ids,
        "attention_mask": attention_mask,
        "row_id": row_id,
    }

def collate_fn(examples):
    input_ids = torch.tensor([example["input_ids"] for example in examples], dtype=torch.long)
    pixel_values_BLIP = torch.stack([torch.tensor(example["pixel_values_BLIP"], dtype=torch.float) for example in examples])
    image_clip_embeds = torch.tensor([example["image_clip_embeds"] for example in examples], dtype=torch.float)
    decoder_input_ids = torch.tensor([example["decoder_input_ids"] for example in examples], dtype=torch.long)
    attention_mask = torch.tensor([example["attention_mask"] for example in examples], dtype=torch.long)
    row_id = torch.tensor([example["row_id"] for example in examples], dtype=torch.float)
    return {
        "input_ids": input_ids,
        "pixel_values_BLIP": pixel_values_BLIP,
        "image_clip_embeds": image_clip_embeds,
        "decoder_input_ids": decoder_input_ids,
        "attention_mask": attention_mask,
        "row_id": row_id,
    }

transtab = str.maketrans({key: None for key in string.punctuation})
class TransformTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, is_eval=False):
        # Disable BatchNorm & Dropout
        # for module in model.discriminator.modules():
        #     if isinstance(module, torch.nn.BatchNorm2d) or isinstance(module, torch.nn.Dropout):
        #         if hasattr(module, 'weight'):
        #             module.weight.requires_grad_(False)
        #         if hasattr(module, 'bias'):
        #             module.bias.requires_grad_(False)
        #         module.eval()
        # print(model)
        # for module in model.generator.modules():
        #     if isinstance(module, torch.nn.Dropout):
        #         if hasattr(module, 'weight'):
        #             module.weight.requires_grad_(False)
        #         if hasattr(module, 'bias'):
        #             module.bias.requires_grad_(False)
        #         module.eval()
        if(is_eval):
            labels = torch.clone(inputs["decoder_input_ids"])
            output_sequences = model.generator.generate(pixel_values=inputs["pixel_values_BLIP"], input_ids=inputs["input_ids"], num_beams=3, max_length=20)
            print(["{}".format(sequence.replace(".", "")) for sequence in model.tokenizer_BLIP.batch_decode(output_sequences, skip_special_tokens=True)])
            tokenized_captions = model.tokenizer_CLIP(["{}".format(sequence.replace(".", "")) for sequence in model.tokenizer_CLIP.batch_decode(output_sequences, skip_special_tokens=True)], padding="longest", truncation=True, return_tensors="pt").to(model.device)
            text_outputs = model.discriminator.text_model(
                input_ids=tokenized_captions["input_ids"],
                attention_mask=tokenized_captions["attention_mask"],
                return_dict=model.config.return_dict
            )
            text_embeds = text_outputs[1]
            text_embeds = model.discriminator.text_projection(text_embeds)

            # normalized features
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
            outputs_clip = {}
            outputs_clip["text_embeds"] = text_embeds
            outputs_clip["image_embeds"] = inputs["image_clip_embeds"]
            return (torch.tensor(0.0, requires_grad=False), (output_sequences, labels), outputs_clip)
  
        image_embeds = inputs["image_clip_embeds"]
    
        # Generate captions
        with torch.no_grad():
            # Beam search
            output_policy = model.generator.generate(pixel_values=inputs["pixel_values_BLIP"], num_beams=5, no_repeat_ngram_size=3, return_dict_in_generate=True) # model.generator.generate(pixel_values=inputs
            
            # Greedy search
            output_baseline = model.generator.generate(pixel_values=inputs["pixel_values_BLIP"], do_sample=False, return_dict_in_generate=True)
   
        
        # Computing logits of generated captions with a forward pass (using scores of HF is a mess, especially for BS)
        gt_logits = model.generator(pixel_values=inputs["pixel_values_BLIP"], input_ids=inputs["decoder_input_ids"], return_dict=True)["decoder_logits"]
        shift_logits = gt_logits[..., :-1, :].contiguous()
        shift_labels = inputs["decoder_input_ids"][..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=1)
        logprob_gt = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        logprob_gt = - logprob_gt.view(shift_logits.size(0), shift_logits.size(1))

        policy_logits = model.generator(pixel_values=inputs["pixel_values_BLIP"], input_ids=output_policy["sequences"], return_dict=True)["decoder_logits"]
        shift_logits = policy_logits[..., :-1, :].contiguous()
        shift_labels = output_policy["sequences"][..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=1)
        logprob_policy = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        logprob_policy = - logprob_policy.view(shift_logits.size(0), shift_logits.size(1))

        
        # Decoding sequences to feed it to CLIP
        discriminator_inputs_text =  ["{}".format(sequence.replace(".", "").strip()) for sequence in model.tokenizer_BLIP.batch_decode(output_policy["sequences"], skip_special_tokens=True)] + ["{}".format(sequence.replace(".", "").strip()) for sequence in model.tokenizer_BLIP.batch_decode(inputs["decoder_input_ids"], skip_special_tokens=True)] + ["{}".format(sequence.replace(".", "").strip()) for sequence in model.tokenizer_BLIP.batch_decode(output_baseline["sequences"], skip_special_tokens=True)]
      
  
        
        print("Policy text {}".format(discriminator_inputs_text[:int(len(discriminator_inputs_text)/3)]))
        print("GT text {}".format(["{}".format(discriminator_inputs_text[int(len(discriminator_inputs_text)/3):2*int(len(discriminator_inputs_text)/3)])]))
        print("Baseline text {}".format(["{}".format(discriminator_inputs_text[2*int(len(discriminator_inputs_text)/3):])]))
        print(inputs["row_id"])
        tokenized_captions = model.tokenizer_CLIP(discriminator_inputs_text, padding="longest", truncation=True, return_tensors="pt").to(model.device)
        # Computing CLIP embeddings
        text_outputs = model.discriminator.text_model(
            input_ids=tokenized_captions["input_ids"],
            attention_mask=tokenized_captions["attention_mask"],
            # position_ids=inputs["position_ids"],
            return_dict=model.config.return_dict
        )
        text_embeds = text_outputs[1]
        text_embeds = model.discriminator.text_projection(text_embeds)

        # normalized features
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        # Getting scores from discriminator and training it
        disc_logits = model.MLP(torch.cat((image_embeds.repeat((2, 1)).detach(), text_embeds[:2*text_embeds.shape[0]//3].detach()), dim=1))
        disc_scores = torch.nn.Softmax(dim=1)(disc_logits)
        labels_disc = torch.zeros((disc_logits.shape[0], ), device=model.device)
        labels_disc[(labels_disc.shape[0] // 2):] = 1
        loss_disc = model.BCELoss(disc_scores[:, 1], labels_disc)

        self.log({"MLP_acc": (torch.sum(torch.argmax(disc_scores, dim=1) == labels_disc) / len(disc_scores)).item()})
        loss_disc.backward()
        self.model.MLP_optimizer.step()
        # Which neighbours to keep as baselines. Too many/hard lead to too negative rewards and require to add a constant to the reward to not crash
        neighbours_to_keep = [3,4]
        logits = torch.matmul(text_embeds, torch.cat((image_embeds, inputs["neighbours_image_embeddings"][torch.tensor([neighbor_to_keep + i*5 for i in range(inputs["input_ids"].shape[0]) for neighbor_to_keep in neighbours_to_keep], dtype=torch.long)]), dim=0).t()) * model.discriminator.logit_scale.exp()

        # Computing mask to get rewards using the big similarity matrix
        # Diagonal for BS/GT and zeros for GS (only GS as the baseline)
        # loss_mask = loss_mask.repeat(2, 1)
        # loss_mask_txt = torch.cat((loss_mask, torch.zeros((inputs["input_ids"].shape[0], inputs["input_ids"].shape[0]), device=model.device)))
        # But since we consider everything else as the baseline (not excluding BS/GT), everything is at zero and we substract identity in the final calculation
        loss_mask = torch.diag(torch.ones((inputs["input_ids"].shape[0],), device=model.device))
        loss_mask_img = torch.zeros((inputs["input_ids"].shape[0]*3, inputs["input_ids"].shape[0]*(len(neighbours_to_keep)+1)), device=model.device)
        loss_mask_txt =  torch.zeros((inputs["input_ids"].shape[0]*3, inputs["input_ids"].shape[0]*(len(neighbours_to_keep)+1)), device=model.device)

        exp_logits = logits.exp()
        # Substract identity to exclude itself from LSE
        log_prob_text = logits - ((exp_logits * (loss_mask_txt == 0)).sum(0) - exp_logits).log()
        log_prob_img = logits.T - ((exp_logits.t() * (loss_mask_img == 0).t()).sum(0) - exp_logits.t()).log()
   
      
        # Prevent underflow by clipping reward to log(2**-23) in case sum_exp have been dominated by the largest value and so the substraction return 0, hence log return -inf and the reward is thus inf
        log_prob_text = torch.clamp(log_prob_text, max=16)
        log_prob_img = torch.clamp(log_prob_img, max=16)
        # log_prob_text = torch.clamp(log_prob_text, min=0, max=16)
        # log_prob_img = torch.clamp(log_prob_img, min=0, max=16)

        log_prob_text /= model.discriminator.logit_scale.exp()
        log_prob_img /= model.discriminator.logit_scale.exp()
        
        
        # Unidirectional reward
        # rewards_policy = torch.diag(log_prob_text[:(logits.shape[0]//3)])
        # Bidirectional reward
        rewards_policy = (torch.diag(log_prob_text[:(logits.shape[0]//3)]) + torch.diag(log_prob_img[:, :(logits.shape[0]//3)])) / 2
        rewards_gt = (torch.diag(log_prob_text[(logits.shape[0]//3):2*(logits.shape[0]//3)]) + torch.diag(log_prob_img[:, (logits.shape[0]//3):2*(logits.shape[0]//3)])) / 2

        disc_scores_policy = disc_scores[:disc_scores.shape[0]//2, 1]
        disc_scores_gt = disc_scores[disc_scores.shape[0]//2:, 1]
        
        # Rewards are CLIP + discriminator
        rewards_gt = (0.94 * rewards_gt + 0.06 * disc_scores_gt)
        rewards_policy = (0.94 * rewards_policy + 0.06 * disc_scores_policy) # + 0.0125 unidirectionnal rewards are more
        self.log({"mean_reward_policy": torch.mean(rewards_policy).item()})
        self.log({"mean_reward_disc_gt": torch.mean(disc_scores_gt).item()})
        self.log({"mean_reward_disc_policy": torch.mean(disc_scores_policy).item()})
         
        self.log({"mean_total_reward_policy": torch.mean(rewards_policy).item()})
        self.log({"mean_total_reward_gt": torch.mean(rewards_gt).item()})
        # RL loss
        policy_loss = - torch.mean((rewards_policy).detach() * torch.mean(logprob_policy, dim=1))
        # WTF loss
        policy_loss = policy_loss - torch.mean((rewards_policy).detach() * torch.mean(logprob_policy, dim=1))
        self.log({"policy_loss": (policy_loss).item()})
        
        policy_loss.backward()
        # Gradient accumulation
        # if(((self.state.global_step + 1) % 1) == 0):

        #     self.model.optimizer_gen.step()
        #     self.model.scheduler_gen.step()
        #     self.model.optimizer_gen.zero_grad()
        
        return torch.tensor(0.0, requires_grad=True)
 

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            (loss, outputs_blip, outputs_clip) = self.compute_loss(model, inputs, return_outputs=True, is_eval=True)
        return (loss, outputs_blip, (outputs_clip["text_embeds"], outputs_clip["image_embeds"]))
    def get_train_dataloader(self):
        train_dataset = self.train_dataset
        train_sampler = self._get_train_sampler()
        return torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=collate_fn_train,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
        

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    parser.add_argument("--baseline", type=str, default="greedy", help="What to use as baseline for SCST (greedy, gt, or sample)", choices=["greedy", "gt", "sample"])
    parser.add_argument("--disc_weight", type=float, default=0, help="Weight of the discriminator for the weighted average")
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, custom_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    # send_example_telemetry("run_summarization", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    
    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files this script will use the first column for the full texts and the second column for the
    # summaries (unless you specify column names for this with the `text_column` and `summary_column` arguments).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    base_path = "VisualNews/"
    tokenizer_blip = AutoTokenizer.from_pretrained("Salesforce/blip-image-captioning-large")
    tokenizer_clip = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32", cache_dir="/srv/tempdd/achaffin/.cache")
    config = BlipConfig.from_pretrained("Salesforce/blip-image-captioning-large")
    model = GCN(config, model_args.generator_name_or_path, model_args.discriminator_name_or_path, cache_dir=model_args.cache_dir, learning_rate=training_args.learning_rate, custom_args=custom_args)
    model.cuda()
    config_clip = model.discriminator.config
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        "openai/clip-vit-base-patch32",
        cache_dir=model_args.cache_dir,
    )


   

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = raw_datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    # Get the column names for input/target.
    dataset_columns = dataset_name_mapping.get(data_args.dataset_name, None)
    if data_args.caption_column is None:
        caption_column = dataset_columns[2] if dataset_columns is not None else column_names[0]
    else:
        caption_column = data_args.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"--text_column' value '{data_args.text_column}' needs to be one of: {', '.join(column_names)}"
            )
    if data_args.image_column is None:
        image_column = dataset_columns[1] if dataset_columns is not None else column_names[0]
    else:
        image_column = data_args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"--image_column' value '{data_args.image_column}' needs to be one of: {', '.join(column_names)}"
            )
    
    mean, std = [0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]
    # mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    resolution = 384
    image_transformations_blip = Transform_BLIP(
        resolution, mean, std
    )
    image_transformations_blip = torch.jit.script(image_transformations_blip)

    image_transformations_clip = Transform_CLIP(
        model.discriminator.config.vision_config.image_size, feature_extractor.image_mean, feature_extractor.image_std
    )
    image_transformations_clip = torch.jit.script(image_transformations_clip)

    convert_tensor = transforms.ToTensor()
    
    def preprocess_function_train(examples):
        prompt_generator = "A photography of"
        prompts_generator = tokenizer_blip(prompt_generator)
        prompts_generator = [prompts_generator.input_ids] * len(examples[caption_column])
        examples["input_ids"] = prompts_generator

        captions = [caption.split("&&")[0].strip() for caption in examples[caption_column]]

        text_inputs = tokenizer_blip(captions, padding="max_length", max_length=77, truncation=True)
        examples["decoder_input_ids"] = text_inputs.input_ids
        examples["attention_mask"] = text_inputs.attention_mask
        images = [Image.open(BytesIO(base64.b64decode(img_data))).convert('RGB') for img_data in examples[image_column]]
        examples["pixel_values_BLIP"] = processor(images, return_tensors="pt")["pixel_values"]
        examples["image_clip_embeds"] = [embedding for embedding in examples["img_embeds"]]
        nearest_neighbors = [neighbours for neighbours in examples["nearest_neighbors"]]
        examples["embeddings_neighbours"] = [train_img_embeds[nearest_neighbor] for nearest_neighbor in nearest_neighbors] 
       
        del examples['predicted_object_labels']
        # del examples['img']
        # del examples[caption_column]
        return examples
    def preprocess_function(examples):
        prompt_generator = "A photography of"
        prompts_generator = tokenizer_blip(prompt_generator)
        prompts_generator = [prompts_generator.input_ids] * len(examples[caption_column])
        examples["input_ids"] = prompts_generator

        captions = [caption.split("&&")[0].strip() for caption in examples[caption_column]]
    
        text_inputs = tokenizer_blip(captions, padding="max_length", max_length=77, truncation=True)
        examples["decoder_input_ids"] = text_inputs.input_ids
        examples["attention_mask"] = text_inputs.attention_mask
        images = [Image.open(BytesIO(base64.b64decode(img_data))).convert('RGB') for img_data in examples[image_column]]
        examples["pixel_values_BLIP"] = processor(images, return_tensors="pt")["pixel_values"]
        with torch.no_grad():
            vision_outputs = model.discriminator.vision_model(
                pixel_values=torch.stack([image_transformations_clip(convert_tensor(image)) for image in images]).to(model.device),
                return_dict=model.config.return_dict
            )
            image_embeds = vision_outputs[1]
            image_embeds = model.discriminator.visual_projection(image_embeds)

            # normalized features
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)


        examples["image_clip_embeds"] = image_embeds.cpu().numpy()
        del examples['predicted_object_labels']
        return examples
    processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    def transform_images(examples):
        images = [convert_tensor(Image.open(BytesIO(base64.b64decode(img_data))).convert('RGB')) for img_data in examples[image_column]]
        examples["pixel_values_CLIP"] = [image_transformations_clip(image) for image in images]
        examples["pixel_values_BLIP"] = processor(images, return_tensors="pt")["pixel_values"]
        return examples

    def filter_corrupt_images(examples):
        """remove problematic images"""
        valid_images = []
        for img_data in examples[image_column]:
            try:
                # Image.open(visual_news_path + "/origin" + image_file[1:])
                Image.open(BytesIO(base64.b64decode(img_data))).convert('RGB')
                valid_images.append(True)
            except Exception:
                print("can't read")
                valid_images.append(False)
        return valid_images

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        train_dataset = train_dataset.add_column("row_id", np.arange(len(train_dataset)))
        # Loading pre-computed embeddings
        train_img_embeds = np.load("embeddings/image_embeddings_train.npy")
        train_text_embeds = np.load("embeddings/caption_embeddings_train.npy")
        train_dataset_embeddings = datasets.Dataset.from_dict({"caption_embeds": train_text_embeds, "img_embeds": train_img_embeds, "nearest_neighbors": np.load("embeddings/nn_train.npy")})
        train_dataset = datasets.concatenate_datasets([train_dataset, train_dataset_embeddings], axis=1)
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.filter(
                filter_corrupt_images, batched=True, num_proc=data_args.preprocessing_num_workers
            )   
            train_dataset = train_dataset.map(
                preprocess_function_train,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                # remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )
            # train_dataset.set_transform(transform_images)

    if training_args.do_eval:
        # max_target_length = data_args.val_max_target_length
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        eval_dataset = eval_dataset.add_column("row_id", np.arange(len(eval_dataset)))
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.filter(
                filter_corrupt_images, batched=True, num_proc=data_args.preprocessing_num_workers
            )
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                # remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )

    if training_args.do_predict:
        # max_target_length = data_args.val_max_target_length
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )
            # eval_dataset.set_transform(transform_images)

    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id

    def compute_metrics(eval_preds):
        (preds, labels), (text_embeds, image_embeds) = eval_preds
        preds = np.where(preds != -100, preds, tokenizer_ofa.pad_token_id)
        # Decoding generated captions
        decoded_preds = tokenizer_ofa.batch_decode(preds, skip_special_tokens=True)
        transtab = str.maketrans({key: None for key in string.punctuation})
        decoded_preds = [elem.translate(transtab).strip() for elem in decoded_preds]

        # Written quality metrics
        references = [elem["caption"].split("&&") for elem in eval_dataset]
        coco = COCO("./datasets/caption_data/test_caption_coco_format.json")
        output_res = [{"image_id": elem["image_id"], "caption": decoded_preds[i]} for i, elem in enumerate(eval_dataset)]
        cocoRes = coco.loadRes(output_res)
        cocoEval = COCOEvalCap(coco, cocoRes)
        cocoEval.evaluate()
        result = cocoEval.eval

        # Retrieval results
        logits_per_text = np.matmul(text_embeds, np.transpose(image_embeds))

        indices = np.argsort(-logits_per_text)
        ranks = indices.argsort()
        ranks = np.diag(ranks).astype(int)
        ranks = ranks + 1 
        mrr_img = np.average(np.reciprocal(ranks.astype(np.float32)))
        for k in {1, 5, 10}:
            result[f"R@{k}_img_CLIP"] = np.average(ranks <= k)
        
        indices = np.argsort(np.transpose(-logits_per_text))
        ranks = indices.argsort()
        ranks = np.diag(ranks).astype(int)
        ranks = ranks + 1
        print(ranks)
        mrr_text = np.average(np.reciprocal(ranks.astype(np.float32)))
        for k in {1, 5, 10}:
            result[f"R@{k}_text_CLIP"] = np.average(ranks <= k)
        
        result["mrr_text"] = mrr_text
        result["mrr_img"] = mrr_img
        
        SBleu = SelfBleu(decoded_preds, sample_size=1000)
        result["SelfBLEU"] = SBleu.get_bleu_fast()
        
        
        return result
    


    trainer = TransformTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        data_collator=collate_fn,
        compute_metrics=compute_metrics
    )
    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    # max_length = (
    #     training_args.generation_max_length
    #     if training_args.generation_max_length is not None
    #     else data_args.val_max_target_length
    # )
    max_length = 256
    num_beams = 3
    # num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(max_length=max_length, num_beams=num_beams, metric_key_prefix="eval")
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        predict_results = trainer.predict(
            predict_dataset, metric_key_prefix="predict", max_length=max_length, num_beams=num_beams
        )
        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                predictions = tokenizer.batch_decode(
                    predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                predictions = [pred.strip() for pred in predictions]
                output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
                with open(output_prediction_file, "w") as writer:
                    writer.write("\n".join(predictions))

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "summarization"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if data_args.lang is not None:
        kwargs["language"] = data_args.lang

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()