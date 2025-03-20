
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import evaluate
import numpy as np
from datasets import ClassLabel, load_dataset

import transformers
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    HfArgumentParser,
    PretrainedConfig,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from datasets import load_dataset

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/token-classification/requirements.txt")


# # Global variables
# HF_HOME = "./.cache"
# TRANSFORMERS_CACHE = "./.cache/transformers"
# if not os.path.exists(HF_HOME):
#     os.makedirs(HF_HOME)

# if not os.path.exists(TRANSFORMERS_CACHE):
#     os.makedirs(TRANSFORMERS_CACHE)                     
# os.environ["HF_HOME"] = HF_HOME
# os.environ["TRANSFORMERS_CACHE"] = TRANSFORMERS_CACHE

print("done!")