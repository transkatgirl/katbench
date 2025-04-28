import numpy as np
from aenum import extend_enum

from lighteval.metrics.metrics import Metrics, SampleLevelMetric
from lighteval.metrics.utils.metric_utils import MetricCategory, MetricUseCase
from lighteval.tasks.default_prompts import LETTER_INDICES
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc

def prompt_fn(line, task_name: str = None):
    # must subset string to prevent OOM errors
    return Doc(task_name=task_name, query=line["text"][:4096], gold_index=None, choices=None)

language_datasets = {
    "tinystories": {
        "hf_repo": "roneneldan/TinyStories",
        "hf_subsets": [],
        "hf_avail_splits": ["train", "validation"],
        "evaluation_splits": ["validation"],
    },
    "pile": {
        "hf_repo": "lighteval/pile_helm",
        "hf_subsets": ["arxiv", "bibliotik", "commoncrawl", "dm_mathematics", "enron", "europarl", "freelaw", "github", "gutenberg", "hackernews", "nih_exporter", "opensubtitles", "openwebtext2", "pubmed_abstracts", "pubmed_central", "stackexchange", "uspto", "wikipedia", "youtubesubtitles"],
        "hf_avail_splits": ["test"],
        "evaluation_splits": ["test"],
    },
}

language_tasks = []

for key, value in language_datasets.items():
    if len(value["hf_subsets"]) == 0:
        language_tasks.append(LightevalTaskConfig(
            name=key,
            suite=["community"],
            prompt_function=prompt_fn,
            hf_repo=value["hf_repo"],
            hf_subset="default",
            hf_avail_splits=value["hf_avail_splits"],
            evaluation_splits=value["evaluation_splits"],
            few_shots_split=None,
            few_shots_select=None,
            generation_size=-1,
            stop_sequence=["\n"],
            metric=[Metrics.word_perplexity, Metrics.bits_per_byte],
            trust_dataset=True
        ))
    else:
        for subset in value["hf_subsets"]:
            language_tasks.append(LightevalTaskConfig(
                name=key+":"+subset,
                suite=["community"],
                prompt_function=prompt_fn,
                hf_repo=value["hf_repo"],
                hf_subset=subset,
                hf_avail_splits=value["hf_avail_splits"],
                evaluation_splits=value["evaluation_splits"],
                few_shots_split=None,
                few_shots_select=None,
                generation_size=-1,
                stop_sequence=["\n"],
                metric=[Metrics.word_perplexity, Metrics.bits_per_byte],
                trust_dataset=True
            ))

TASKS_TABLE = language_tasks