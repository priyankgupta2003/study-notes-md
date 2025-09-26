
# Evaluating LLM Performance

## Introduction

In this lesson, we will explore two crucial aspects of language model evaluation: objective functions and evaluation metrics.

Objective functions, also known as loss functions, play a vital role in guiding the learning process during model training. On the other hand, evaluation metrics provide interpretable measures of the model's capabilities and are used to assess its performance on various tasks.

We will dive into the perplexity evaluation metric, commonly used for LLMs, and explore several benchmarking frameworks, such as GLUE, SuperGLUE, BIG-bench, HELM, and FLASK, that help comprehensively evaluate language models across diverse scenarios.

## Objective Functions and Evaluation Metrics

Objective functions and evaluation metrics are essential components in machine learning models.

The **objective function**, also known as the **loss function**, is a mathematical formula used during the training phase. It gives a loss score to the model in function of the model parameters. During training, the learning algorithm computes gradients of the loss function and updates the model parameters to minimize it. As a consequence, to guarantee (smooth) learning, the loss function needs to be differentiable and have an excellent smooth form.

The objective function typically used for LLMs is the **cross-entropy loss**. In the case of causal language modeling, the model predicts the next token from a fixed list of tokens, essentially making it a classification problem.

On the other hand, **evaluation metrics** are used to assess the model's performance in an interpretable way for people. Unlike the objective function, evaluation metrics are not directly used during training. As a consequence, evaluation metrics don’t need to be differentiable, as we won’t have to compute gradients for them. Standard evaluation metrics include accuracy, precision, recall, F1-score, and mean squared error.

Typical evaluation metrics for LLMs can be:

- **Intrinsic metrics**, i.e., metrics strictly related to the training objective. A popular example is the **perplexity** metric.
- **Extrinsic metrics** are metrics that aim to assess performance on several downstream tasks and are not strictly related to the training objective. The GLUE, SuperGLUE, BIG-bench, HELM, and FLASK benchmarks are popular examples.

## The Perplexity Evaluation Metric

Perplexity is an evaluation metric used to assess the performance of LLMs. It measures how well a language model predicts a given sample or sequence of words, such as a sentence. The lower the perplexity value, the better the language model is at predicting the sample.

LLMs are designed to model the probability distributions of words within sentences. They can generate sentences resembling human writing and assess the sentences' quality. Perplexity is a measure that quantifies the uncertainty or "perplexity" a model experiences when assigning probabilities to sequences of words.

The first step in computing perplexity is to calculate the probability of a sentence by multiplying the probabilities of individual words according to the language model. Longer sentences tend to have lower probabilities due to the multiplication of factors smaller than one. To make comparisons between sentences with different lengths possible, perplexity normalizes the probability by dividing it by the number of words in the sentence and taking the geometric mean.

### Perplexity Example

Consider an example where a language model is trained to predict the subsequent word in a sentence: "A red fox." For a competent LLM, the predicted word probabilities could be as follows, step by step.

> P(“a red fox.”) =
> 
> = P(“a”) * P(“red” | “a”) * P(“fox” | “a red”) * P(“.” | “a red fox”) =
> 
> = 0.4 * 0.27 * 0.55 * 0.79 =
> 
> = 0.0469

It would be nice to compare the probabilities assigned to different sentences to see which sentences are better predicted by the language model. However, since the probability of a sentence is obtained from a product of probabilities, the longer the sentence, the lower its probability (since it’s a product of factors with values smaller than one). We should find a way of measuring these sentence probabilities without the influence of the sentence length.

This can be done by normalizing the sentence probability by the number of words in the sentence. Since the probability of a sentence is obtained by multiplying many factors, we can average them using the [geometric mean](https://en.wikipedia.org/wiki/Geometric_mean).

Let’s call _Pnorm(W)_ the normalized probability of the sentence _W_. Let _n_ be the number of words in _W_. Then, applying the geometric mean:

> Pnorm(W) = P(W) ^ (1 / n)

Using our specific sentence, “_a red fox._”:

> Pnorm(“a red fox.”) = P(“a red ”) ^ (1 / 4) = 0.465

Great! This number can now be used to compare the probabilities of sentences with different lengths. The higher this number is over a well-written sentence, the better the language model.

So, what does this have to do with perplexity? Well, perplexity is just the reciprocal of this number.

Let’s call _PP(W)_ the perplexity computed over the sentence _W_. Then:

> PP(W) = 1 / Pnorm(W) =
> 
> = 1 / (P(W) ^ (1 / n))
> 
> = (1 / P(W)) ^ (1 / n)

Let’s compute it with `numpy`:

Copy

```python
import numpy as np

probabilities = np.array([0.4, 0.27, 0.55, 0.79])
sentence_probability = probabilities.prod()
sentence_probability_normalized = sentence_probability ** (1 / len(probabilities))
perplexity = 1 / sentence_probability_normalized
print(perplexity) # 2.1485556947850033
```

Suppose we further train the LLM, and the probabilities of the next best word become higher. How would the final perplexity be, higher or lower?

Copy

```python
probabilities = np.array([0.7, 0.5, 0.6, 0.9])
sentence_probability = probabilities.prod()
sentence_probability_normalized = sentence_probability ** (1 / len(probabilities))
perplexity = 1 / sentence_probability_normalized
print(perplexity) # 1.516647134682679 -> lower
```

## The GLUE Benchmark

The [GLUE](https://gluebenchmark.com/) (General Language Understanding Evaluation) benchmark comprises nine diverse English sentence understanding tasks categorized into three groups.

- The first group, Single-Sentence Tasks, evaluates the model's ability to determine grammatical correctness (CoLA) and sentiment polarity (SST-2) of individual sentences.
- The second group, Similarity, and Paraphrase Tasks, focuses on assessing the model's capacity to identify paraphrases in sentence pairs (MRPC and QQP) and determine the similarity score between sentences (STS-B).
- The third group, Inference Tasks, challenges the model to handle sentence entailment and relationships. This includes recognizing textual entailment (RTE), answering questions based on sentence information (QNLI), and resolving pronoun references (WNLI).

The final GLUE score is obtained by averaging performance across all nine tasks. By providing a unified evaluation platform, GLUE facilitates a deeper understanding of the strengths and weaknesses of various NLP models.

## The **SuperGLUE Benchmark**

The [SuperGLUE](https://super.gluebenchmark.com/) benchmark builds upon the GLUE benchmark but introduces more complex tasks to push the boundaries of current NLP approaches. The key features of SuperGLUE are:

1. Tasks: SuperGLUE consists of eight diverse language understanding tasks. These tasks include Boolean question answering, textual entailment, coreference resolution, reading comprehension with commonsense reasoning, and word sense disambiguation.
2. Difficulty: The benchmark retains the two hardest tasks from GLUE and adds new tasks based on the challenges faced by current NLP models, ensuring greater complexity and relevance to real-world language understanding scenarios.
3. Human Baselines: Human performance estimates are included for each task, providing a benchmark for evaluating the performance of NLP models against human-level understanding.
4. Evaluation: NLP models are evaluated on these tasks, and their performance is measured using a single-number overall score obtained by averaging the scores of all individual tasks.

## The BIG-Bench Benchmark

[BIG-bench](https://github.com/google/BIG-bench) is a large-scale and diverse benchmark designed to evaluate the capabilities of large language models. It consists of 204 or more language tasks that cover a wide range of topics and languages. These are challenging and not entirely solvable by current models.

The benchmark supports two types of tasks: JSON-based and programmatic tasks. JSON tasks involve comparing output and target pairs to evaluate performance, while programmatic tasks use Python to measure text generation and conditional log probabilities.

The tasks include writing code, common-sense reasoning, playing games, linguistics, and more.

The researchers found that aggregate performance improves with model size but still falls short of human performance. Model predictions become better calibrated with increased scale, and sparsity offers benefits.

This benchmark is considered a "living benchmark," accepting new task submissions for continuous peer review. The code for BIG-bench is open-source on [GitHub](https://github.com/google/BIG-bench), and the research paper is available on [arXiv](https://arxiv.org/abs/2206.04615).

## The HELM Benchmark

The [HELM](https://crfm.stanford.edu/2022/11/17/helm.html) (Holistic Evaluation of Language Models) benchmark addresses the lack of a unified standard for comparing language models and aims to assess them in their totality. The benchmark has three main components:

1. Broad Coverage and Recognition of Incompleteness: HELM evaluates language models over a diverse set of scenarios, considering different tasks, domains, languages, and user-facing applications. It acknowledges that not all scenarios can be covered but explicitly identifies major scenarios and missing metrics to highlight improvement areas.
2. Multi-Metric Measurement: HELM evaluates language models based on multiple criteria, unlike previous benchmarks that often focus on a single metric like accuracy. It measures 7 metrics: accuracy, calibration, robustness, fairness, bias, toxicity, and efficiency. This multi-metric approach ensures that non-accuracy desiderata are not overlooked.
3. Standardization: HELM aims to standardize the evaluation process for different language models. It specifies an adaptation procedure using few-shot prompting, making it easier to compare models effectively. By evaluating 30 models from various providers, HELM improves the overall landscape of language model evaluation and encourages a more transparent and reliable infrastructure for language technologies.

## The FLASK Benchmark

The [FLASK](https://arxiv.org/abs/2307.10928) (Fine-grained Language Model Evaluation based on Alignment Skill Sets) benchmark is an evaluation protocol for LLMs. It breaks down the evaluation process into 12 specific instance-wise skill sets, each representing a crucial aspect of a model's capabilities.

These skill sets comprise logical correctness, logical efficiency, factuality, commonsense understanding, comprehension, insightfulness, completeness, metacognition, readability, conciseness, and harmlessness.

By breaking down the evaluation into these specific skill sets, FLASK allows for a precise and comprehensive assessment of a model's performance across various tasks, domains, and difficulty levels. This approach provides a more detailed and nuanced understanding of a language model's strengths and weaknesses, enabling researchers and developers to improve the models in targeted ways and address specific challenges in natural language processing.

[![Assessing skills across diverse tasks for a range of LLMs, image credit:](https://images.spr.so/cdn-cgi/imagedelivery/j42No7y-dcokJuNgXeA0ig/0d4f3ce6-b3fd-4177-a499-dc2e758759ff/flask/w=1200,quality=90,fit=scale-down)](https://assessing%20skills%20across%20diverse%20tasks%20for%20a%20range%20of%20llms,%20image%20credit:%20https//arxiv.org/pdf/2307.10928.pdf)

## Conclusion

In this lesson, we explored the essential concepts of evaluating LLM performance through objective functions and evaluation metrics. The objective or loss function plays a critical role during model training. It guides the learning algorithm to minimize the loss score by updating model parameters. For LLMs, the common objective function is the cross-entropy loss.

On the other hand, evaluation metrics are used to assess the model's performance more interpretably, though they are not directly used during training. Perplexity is one such intrinsic metric used to measure how well an LLM predicts a given sample or sequence of words.

Additionally, the lesson introduced several popular extrinsic evaluation benchmarks, such as GLUE, SuperGLUE, BIG-bench, HELM, and FLASK, which evaluate language models on diverse tasks and scenarios, covering aspects like accuracy, fairness, robustness, and more.

By understanding these concepts and using appropriate evaluation metrics and benchmarks, researchers and developers can gain valuable insights into language models' strengths and weaknesses, leading to improving these technologies.