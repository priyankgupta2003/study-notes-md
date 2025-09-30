
- **[AI2 Reasoning Challenge](https://allenai.org/data/arc)** (ARC): The dataset is exclusively comprised of natural, grade-school science questions designed for human tests.
- **[HumanEval](https://paperswithcode.com/dataset/humaneval)**: It is used to measure program synthesis from docstrings. It includes 164 original programming problems assessing language comprehension, algorithms, and math, some resembling software interview questions.
- **[HellaSwag](https://paperswithcode.com/dataset/hellaswag)**: A challenge to measure commonsense inference, and shows it remains difficult for state-of-the-art models. While humans achieve >95% accuracy on trivial questions, the models struggle with <48% accuracy.
- **[Measuring Massive Multitask Language Understanding](https://paperswithcode.com/dataset/mmlu)** **(**MMLU): An evaluation metric for text models' multitask accuracy, covering 57 tasks, including math, US history, computer science, law, etc. High accuracy requires extensive knowledge of the world and problem-solving ability.
- **[TruthfulQA](https://paperswithcode.com/dataset/truthfulqa)**: A truthfulness benchmark designed to assess the accuracy of language models in generating answers to questions. It consists of 817 questions across 38 categories, encompassing topics such as health, law, finance, and politics.


## **Language Model Evaluation Harness**

[EleutherAI](https://www.eleuther.ai/) has released a benchmarking script capable of evaluating any language model, whether it is proprietary and accessible via API or an open-source model. It is customized for generative tasks, utilizing publicly available prompts to enable fair comparisons among the papers.

Reproducibility stands as a vital aspect of any evaluation process! Especially in generative models, there are numerous parameters available during inference, each offering varying levels of randomness. They employ a task versioning feature that guarantees comparability of results even after the tasks undergo updates.

`https://github.com/EleutherAI/lm-evaluation-harness`


## **InstructEval**

![[Pasted image 20250929225855.png]]


**1. Problem-Solving Evaluation**

It consists of the following test to evaluate the model’s ability on **World** **Knowledge** using [Massive Multitask Language Understanding](https://paperswithcode.com/dataset/mmlu) (MMLU), **Complex Instructions** using [BIG-Bench Hard](https://paperswithcode.com/dataset/big-bench) (BBH), **Comprehension and Arithmetic** using [Discrete Reasoning Over Paragraphs](https://paperswithcode.com/dataset/drop) (DROP), **Programming** using [HumanEval](https://paperswithcode.com/dataset/humaneval), and lastly **Causality** using [Counterfactual Reasoning Assessment](https://arxiv.org/abs/2112.11941) (CRASS). These automated evaluations assess the model's performance across various tasks.

**2. Writing Evaluation**

This category will evaluate the model based on the following subjective metrics: Informative, Professional, Argumentative, and Creative. They used the GPT-4 model to evaluate the output of different models by presenting a rubric and asking the model to score the outputs on the [Likert scale](https://en.wikipedia.org/wiki/Likert_scale) between 1 and 5.

**3. Alignment to Human Values**

Finally, a crucial aspect of instruction-tuned models is their alignment with human values. We anticipate these models will uphold values such as helpfulness, honesty, and harmlessness. The leaderboard will evaluate the model by presenting pairs of dialogues and asking it to choose the appropriate one.


# Domain Specific LLMs

## **BloombergGPT**

[BloombergGPT](https://www.bloomberg.com/company/press/bloomberggpt-50-billion-parameter-llm-tuned-finance/) is a proprietary domain-specific 50B LLM trained for the financial domain.'

Its **training dataset** is called “FinPile” and is made of many English financial documents derived from diverse sources, encompassing financial news, corporate filings, press releases, and even social media taken from Bloomberg archives (thus, it’s a proprietary dataset). Data ranges from company filings to market-relevant news from March 2007 to July 2022.

The model is **based on the BLOOM model.** It’s a decoder-only transformer with 70 layers of decoder blocks, multi-head self-attention, layer normalization, and feed-forward networks equipped with the GELU non-linear function. The model is based on the Chinchilla scaling laws.


## **The FinGPT Project**

The [FinGPT](https://github.com/AI4Finance-Foundation/FinGPT) project aims to bring the power of LLMs into the world of finance. It aims to do so in two ways:

1. Providing open finance datasets.
2. Finetuning open-source LLMs on finance datasets for several use cases.

Many datasets collected by FinGPT are specifically for financial sentiment analysis.

- [Financial Phrasebank](https://huggingface.co/datasets/financial_phrasebank): It contains 4840 sentences from English financial news, categorized by financial sentiment (written by agreement between 5-8 annotators).
- [Financial Opinion Mining and Question Answering (FIQA)](https://huggingface.co/datasets/pauri32/fiqa-2018)**:** Consists of 17k sentences from microblog headlines and financial news, classified with financial sentiment.
- [Twitter Financial Dataset (sentiment)](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment): about 10k tweets with financial sentiment.

## Med-PaLM for the Medical Domain

[Med-PaLM](https://sites.research.google/med-palm/) is a finetuned version of PaLM (by Google) specifically for the medical domain.

