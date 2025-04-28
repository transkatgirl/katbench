import numpy as np
from aenum import extend_enum

from lighteval.metrics.metrics import Metrics, SampleLevelMetric
from lighteval.metrics.utils.metric_utils import MetricCategory, MetricUseCase
from lighteval.tasks.default_prompts import LETTER_INDICES
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc

def prompt_fn(line, task_name: str = None):
    # must subset string to prevent OOM errors
    return Doc(task_name=task_name, query=line["text"][:32768], gold_index=None, choices=None)

TASKS_TABLE = [
    LightevalTaskConfig(
        name="pile_10k",
        suite=["community"],
        prompt_function=prompt_fn,
        hf_repo="NeelNanda/pile-10k",
        hf_subset="default",
        hf_avail_splits=["train"],
        evaluation_splits=["train"],
        few_shots_split=None,
        few_shots_select=None,
        generation_size=-1,
        stop_sequence=["\n"],
        metric=[Metrics.word_perplexity, Metrics.bits_per_byte],
        trust_dataset=True
    )
]