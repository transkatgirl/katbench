context_length = 4096
language_datasets = {
    "pile": {
        "hf_repo": "lighteval/pile_helm",
        "hf_subsets": ["arxiv", "bibliotik", "commoncrawl", "dm-mathematics", "enron", "europarl", "freelaw", "github", "gutenberg", "hackernews", "nih-exporter", "opensubtitles", "openwebtext2", "pubmed-abstracts", "pubmed-central", "stackexchange", "uspto", "wikipedia", "youtubesubtitles"],
        "evaluation_splits": ["test"],
    },
    "pile:ubuntu-irc-broken": {
        "hf_repo": "timaeus/pile-ubuntu_irc-broken",
        "hf_subsets": [],
        "evaluation_splits": ["train"],
    },
    "fineweb:bbc": {
        "hf_repo": "permutans/fineweb-bbc-news",
        "hf_subsets": ["sample-10BT"],
        "evaluation_splits": ["train"],
    },
    "twitteraae": {
        "hf_repo": "lighteval/TwitterAAE",
        "hf_subsets": ["aa", "white"],
        "evaluation_splits": ["test"],
    },
    "culturax:eng": {
        "hf_repo": "yiyic/culturaX_eng",
        "hf_subsets": [],
        "evaluation_splits": ["test"],
    },
    "culturax:deu": {
        "hf_repo": "yiyic/culturaX_deu",
        "hf_subsets": [],
        "evaluation_splits": ["test"],
    },
    "culturax:esp": {
        "hf_repo": "yiyic/culturaX_esp",
        "hf_subsets": [],
        "evaluation_splits": ["test"],
    },
    "culturax:fra": {
        "hf_repo": "yiyic/culturaX_fra",
        "hf_subsets": [],
        "evaluation_splits": ["test"],
    },
    "culturax:ja": {
        "hf_repo": "yiyic/CulturaX_ja_test",
        "hf_subsets": [],
        "evaluation_splits": ["train"],
    },
    "culturax:yi": {
        "hf_repo": "yiyic/CulturaX_yi_test",
        "hf_subsets": [],
        "evaluation_splits": ["train"],
    },
    "culturax:he": {
        "hf_repo": "yiyic/CulturaX_he_test",
        "hf_subsets": [],
        "evaluation_splits": ["train"],
    },
    "culturax:am": {
        "hf_repo": "yiyic/CulturaX_am_test",
        "hf_subsets": [],
        "evaluation_splits": ["train"],
    },
    "culturax:mt": {
        "hf_repo": "yiyic/CulturaX_mt_test",
        "hf_subsets": [],
        "evaluation_splits": ["train"],
    },
    "culturax:hi": {
        "hf_repo": "yiyic/CulturaX_hi_test",
        "hf_subsets": [],
        "evaluation_splits": ["train"],
    },
    "culturax:ur": {
        "hf_repo": "yiyic/CulturaX_ur_test",
        "hf_subsets": [],
        "evaluation_splits": ["train"],
    },
    "culturax:gu": {
        "hf_repo": "yiyic/CulturaX_gu_test",
        "hf_subsets": [],
        "evaluation_splits": ["train"],
    },
    "culturax:si": {
        "hf_repo": "yiyic/CulturaX_si_test",
        "hf_subsets": [],
        "evaluation_splits": ["train"],
    },
    "culturax:pa": {
        "hf_repo": "yiyic/CulturaX_pa_test",
        "hf_subsets": [],
        "evaluation_splits": ["train"],
    },
    "culturax:tr": {
        "hf_repo": "yiyic/CulturaX_tr_test",
        "hf_subsets": [],
        "evaluation_splits": ["train"],
    },
    "culturax:kk": {
        "hf_repo": "yiyic/CulturaX_kk_test",
        "hf_subsets": [],
        "evaluation_splits": ["train"],
    },
    "culturax:ko": {
        "hf_repo": "yiyic/CulturaX_ko_test",
        "hf_subsets": [],
        "evaluation_splits": ["train"],
    },
    "culturax:mn": {
        "hf_repo": "yiyic/CulturaX_mn_test",
        "hf_subsets": [],
        "evaluation_splits": ["train"],
    },
    "culturax:mhr": {
        "hf_repo": "yiyic/CulturaX_mhr_test",
        "hf_subsets": [],
        "evaluation_splits": ["train"],
    },
    "culturax:fi": {
        "hf_repo": "yiyic/CulturaX_fi_test",
        "hf_subsets": [],
        "evaluation_splits": ["train"],
    },
    "culturax:ar": {
        "hf_repo": "yiyic/CulturaX_ar_test",
        "hf_subsets": [],
        "evaluation_splits": ["train"],
    },
    "culturax:hu": {
        "hf_repo": "yiyic/CulturaX_hu_test",
        "hf_subsets": [],
        "evaluation_splits": ["train"],
    },
    "culturax:zh": {
        "hf_repo": "yiyic/CulturaX_zh_test",
        "hf_subsets": [],
        "evaluation_splits": ["train"],
    },
    "the-stack": {
        "hf_repo": "bigcode/the-stack-smol-xs",
        "hf_subsets": ["ada", "agda", "alloy", "antlr", "applescript", "assembly", "augeas", "awk", "batchfile", "bison", "bluespec", "c", "c++", "c-sharp", "clojure", "cmake", "coffeescript", "common-lisp", "css", "cuda", "dart", "dockerfile", "elixir", "elm", "emacs-lisp","erlang", "f-sharp", "fortran", "glsl", "go", "groovy", "haskell","html", "idris", "isabelle", "java", "java-server-pages", "javascript", "julia", "kotlin", "lean", "literate-agda", "literate-coffeescript", "literate-haskell", "lua", "makefile", "maple", "markdown", "mathematica", "matlab", "ocaml", "pascal", "perl", "php", "powershell", "prolog", "protocol-buffer", "python", "r", "racket", "restructuredtext", "rmarkdown", "ruby", "rust", "sas", "scala", "scheme", "shell", "smalltalk", "solidity", "sparql", "sql", "stan", "standard-ml", "stata", "systemverilog", "tcl", "tcsh", "tex", "thrift", "typescript", "verilog", "vhdl", "visual-basic", "xslt", "yacc", "zig"],
        "evaluation_splits": ["train"],
    },
    "reddit-posts:popular-mix": {
        "hf_repo": "smwilliams/reddit_large_posts",
        "hf_subsets": [],
        "evaluation_splits": ["train"],
    },
    "reddit-posts:nsfw-stories": {
        "hf_repo": "acheong08/nsfw_reddit",
        "hf_subsets": [],
        "evaluation_splits": ["train"],
    },
    "erotic-books": {
        "hf_repo": "AlekseyKorshuk/erotic-books",
        "hf_subsets": [],
        "evaluation_splits": ["train"],
    },
    "1k_stories": {
        "hf_repo": "FareedKhan/1k_stories_100_genre",
        "hf_subsets": [],
        "evaluation_splits": ["train"],
    },
}

# TODO: Create mini (~10k rows) versions of Fal7acy/4chan-archive, HuggingFaceGECLM/REDDIT_submissions, HuggingFaceGECLM/REDDIT_comments, lemonilia/Elliquiy-Role-Playing-Forums_2023-04, recursal/Fanatic-Fandom, jkkummerfeld/irc_disentangle, enryu43/twitter100m_tweets, lighteval/TwitterAAE, Helsinki-NLP/open_subtitles, ontocord/CulturaY, EleutherAI/wikitext_document_level, milistu/AMAZON-Products-2023, vincha77/filtered_yelp_restaurant_reviews, breadlicker45/discord_data, lmsys/lmsys-chat-1m, open-web-math/open-web-math, mlfoundations/dclm-baseline-1.0-parquet

import numpy as np
from aenum import extend_enum

from lighteval.metrics.metrics import Metrics, SampleLevelMetric
from lighteval.metrics.utils.metric_utils import MetricCategory, MetricUseCase
from lighteval.tasks.default_prompts import LETTER_INDICES
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc

def prompt_fn(line, task_name: str = None):
    if "title" in line and line["title"] and "text" in line and line["text"]:
        return Doc(task_name=task_name, query=(line["title"]+"\n\n"+line["text"])[:context_length], gold_index=None, choices=None)
    elif "text" in line and line["text"]:
        return Doc(task_name=task_name, query=line["text"][:context_length], gold_index=None, choices=None)
    elif "content" in line and line["content"]:
        return Doc(task_name=task_name, query=line["content"][:context_length], gold_index=None, choices=None)
    elif "tweet" in line and line["tweet"]:
        return Doc(task_name=task_name, query=line["tweet"][:context_length], gold_index=None, choices=None)
    elif "input" in line and line["input"] and "output" in line and line["output"]:
        return Doc(task_name=task_name, query=(line["input"]+"\n---\n\n"+line["output"])[:context_length], gold_index=None, choices=None)
    elif "instruction" in line and line["instruction"] and "output" in line and line["output"] and not "input" in line:
        return Doc(task_name=task_name, query=(line["instruction"]+"\n\n"+line["output"])[:context_length], gold_index=None, choices=None)
    elif "title" in line and line["title"] and "story" in line and line["story"]:
        return Doc(task_name=task_name, query=(line["title"]+"\n\n"+line["story"])[:context_length], gold_index=None, choices=None)
    elif "story" in line and line["story"]:
        return Doc(task_name=task_name, query=line["story"][:context_length], gold_index=None, choices=None)
    else:
        print("Warning: dropping line from " + task_name)
        print(line)
        return None

language_tasks = []

for key, value in language_datasets.items():
    if len(value["hf_subsets"]) == 1:
        language_tasks.append(LightevalTaskConfig(
            name=key,
            suite=["community"],
            prompt_function=prompt_fn,
            hf_repo=value["hf_repo"],
            hf_subset=value["hf_subsets"][0],
            hf_avail_splits=value["evaluation_splits"],
            evaluation_splits=value["evaluation_splits"],
            few_shots_split="",
            few_shots_select=None,
            generation_size=-1,
            stop_sequence=["\n"],
            metric=[Metrics.word_perplexity, Metrics.byte_perplexity, Metrics.bits_per_byte],
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
                hf_avail_splits=value["evaluation_splits"],
                evaluation_splits=value["evaluation_splits"],
                few_shots_split="",
                few_shots_select=None,
                generation_size=-1,
                stop_sequence=["\n"],
                metric=[Metrics.word_perplexity, Metrics.byte_perplexity, Metrics.bits_per_byte],
                trust_dataset=True
            ))
    if len(value["hf_subsets"]) == 0:
        language_tasks.append(LightevalTaskConfig(
            name=key,
            suite=["community"],
            prompt_function=prompt_fn,
            hf_repo=value["hf_repo"],
            hf_subset="default",
            hf_avail_splits=value["evaluation_splits"],
            evaluation_splits=value["evaluation_splits"],
            few_shots_split="",
            few_shots_select=None,
            generation_size=-1,
            stop_sequence=["\n"],
            metric=[Metrics.word_perplexity, Metrics.byte_perplexity, Metrics.bits_per_byte],
            trust_dataset=True
        ))

TASKS_TABLE = language_tasks

import os
if not os.path.exists("katbench.txt"):
    with open("katbench.txt", "w") as f:
        for task in TASKS_TABLE:
            f.write(task.suite[0] + "|" + task.name + "|0|0\n")