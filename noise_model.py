from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from sentence_transformers import InputExample
from torch.utils.data import DataLoader
from sentence_transformers import losses
from torch import nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from sentence_transformers import SentenceTransformer
import json
import os, random
from scipy.stats import spearmanr
import numpy as np
import torch
import torch.nn as nn
import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from transformers import pipeline, BitsAndBytesConfig
import argparse
from datasets import load_dataset, Dataset
from peft import (
    get_peft_config, get_peft_model, get_peft_model_state_dict,
    set_peft_model_state_dict, LoraConfig, PeftType, PrefixTuningConfig, PromptEncoderConfig,
    prepare_model_for_kbit_training,
)
import json, os
from tqdm import tqdm
# from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import transformers
import sys
import gc
from sys import getsizeof
import gc
from peft import prepare_model_for_kbit_training
from transformers import GPTQConfig
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from auto_gptq import exllama_set_max_input_length
import numpy as np
import re
from sentence_transformers import SentenceTransformer
import pickle
from datasets import load_dataset


def add_noise(embeddings, noise_factor_1, noise_factor_2, noise_factor_3, noise_factor_4, gaussian_noise_sigma):
    w_n = embeddings / embeddings.abs().max().item()
    w_n[w_n > 0.75] = w_n[w_n > 0.75] + torch.randn_like(w_n[w_n > 0.75]) * noise_factor_1 * gaussian_noise_sigma
    mask = (w_n <= 0.75) * (w_n >= 0.5)
    w_n[mask] = w_n[mask] + torch.randn_like(w_n[mask]) * noise_factor_2 * gaussian_noise_sigma
    mask = (w_n <= 0.5) * (w_n >= 0.25)
    w_n[mask] = w_n[mask] + torch.randn_like(w_n[mask]) * noise_factor_3 * gaussian_noise_sigma
    mask = (w_n <= 0.25)
    w_n[mask] = w_n[mask] + torch.randn_like(w_n[mask]) * noise_factor_4 * gaussian_noise_sigma

    return w_n