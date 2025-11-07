# Code Evaulation Guide

This document provides a code evaluation program for the VibeThinker-1.5B model.

## Evaulation Process

### 1. Clone the Required Project

```shell
git clone git@github.com:LiveCodeBench/LiveCodeBench.git
cd LiveCodeBench
```
### 2. Install Dependencies

require python version: python3.12

```shell
pip install -e . -i https://mirrors.aliyun.com/pypi/simple/
pip install datasets==3.6.0 -i https://mirrors.aliyun.com/pypi/simple/
```

### 3. Download LiveCodeBench Dataset for eval

optional:  export HF_ENDPOINT=https://hf-mirror.com

```python
from datasets import load_dataset

load_dataset("livecodebench/code_generation_lite", split="test", version_tag="release_v6", trust_remote_code=True)
```


### 4. Customize Chat Template

1. **add model enum**

edit lcb_runner/lm_styles.py, add enum for LMStyles:
```python
class LMStyle(Enum):
    ...
    VibeThinker = "VibeThinker"
```

2. **add chat template**
edit the lcb_runner/prompts/code_generation.py   
in the format_prompt_generation function, add this snippet code:   
```python

def format_prompt_generation(
    question: CodeGenerationProblem, LanguageModelStyle: LMStyle
) -> str:
    ...

    if LanguageModelStyle == LMStyle.VibeThinker:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            "<put local path of vibe thinker 1.5B here>", padding_side="left", use_fast=False
        )
        prompt = f"{PromptConstants.SYSTEM_MESSAGE_GENERIC}\n\n"
        prompt += f"{get_generic_question_template_answer(question)}"
        chat_messages = [
            {
                "role": "user",
                "content": prompt,
            },
        ]
        prompt = tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True,
            truncation=False,
            padding=False,
        )
        return prompt
```

3. **add LanguageModel** 

   edit lcb_runner/lm_styles.py,  in LanguageModelList, add this model:

```python
LanguageModelList: list[LanguageModel] = [
    ...

    LanguageModel(
        model_name="VibeThinker/VibeThinker-1.5B",
        model_repr="VibeThinker-1.5B",
        model_style=LMStyle.VibeThinker,
        release_date=datetime(2025, 11, 10),
        link="https://huggingface.co/WeiboAI/VibeThinker-1.5B",
    ),
]

```


### 5. Run the Evaluation

1. LiveCodeBench v6

```shell
# LiveCodeBench v6 (2025.02.01 - 2025.05.01 for release v6, 131 problems total):
N_ROLLOUT=8
python -m lcb_runner.runner.main \
    --model VibeThinker/VibeThinker-1.5B \
    # --local_model_path <local model path for VibeThinker-1.5B>
    --scenario codegeneration \
    --evaluate \
    --release_version release_v6 \
    --temperature 0.6 \
    --n $N_ROLLOUT \
    --codegen_n $N_ROLLOUT \
    --max_tokens 40960 \
    --start_date 2025-02-01 \
    --tensor_parallel_size 4 \
    --num_process_evaluate 180
```

2. LiveCodeBench v5

```shell
# LiveCodeBench v5 (2024.08.01 - 2025.02.01 for release v5, 279 problems total):
N_ROLLOUT=8
python -m lcb_runner.runner.main \
    --model VibeThinker/VibeThinker-1.5B \
    # --local_model_path <local model path for VibeThinker-1.5B>
    --scenario codegeneration \
    --evaluate \
    --release_version release_v5 \
    --temperature 0.6  \
    --n $N_ROLLOUT \
    --codegen_n $N_ROLLOUT \
    --max_tokens 40960 \
    --start_date 2024-08-01 \
    --tensor_parallel_size 4 \
    --num_process_evaluate 180
```


# Acknowledgements

This evaluation program is built upon the [LiveCodeBench/LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench) project and . Thanks to the original authors for their contributions.