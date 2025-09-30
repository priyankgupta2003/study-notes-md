
### Few-Shot (In-Context) Learning

Few-shot learning (also called In-Context learning) enables the LLMs to learn from the examples provided to them. For instance, it is possible to show a couple of examples of JSON-formatted responses to receive the model’s output in JSON format. It means that **the models can learn from examples and follow directions without changing weights or repeating the training process.**

LLMs are able to answer questions using external knowledge bases through in-context learning. Let’s think about how we could create a Q&A chatbot leveraging an LLM. The LLM has a cut-off training date, so it can’t access the information or events after that date. Also, they tend to hallucinate, which refers to generating non-factual responses based on their limited knowledge. As a solution, it is possible to provide additional context to the LLM through the Internet (e.g., Google search) or retrieve it from a database and include it in the prompt so that the model can leverage it to generate the correct response.

### Fine-Tuning
The fine-tuning method proves valuable when adapting the model to a more complex use case. This technique can improve model understanding by providing more examples and adjusting weights based on errors, for tasks like classification or summarization.

The fine-tuning approach is an excellent option for creating a model with task-specific knowledge and building on top of the available powerful LLMs.

### Training
If fine-tuning is not effective, consider training a model from scratch with domain-specific data. However, this requires significant resources, such as cost, dataset availability, and expertise.


### LLMOps

LLMOps is essentially a set of tools and best practices designed to manage the GenAI lifecycle, from development and deployment to maintenance.


## Steps Involved in LLMOps and Differences with MLOps

1. Selection of a Foundation Model
	Foundation models are pre-trained LLMs that can be adapted for various downstream tasks.
	This differs from standard MLOps, where a model is typically trained from scratch with a smaller architectures or on different data, especially for tabular classification and regression tasks

2. Adaptation to Downstream Tasks
	After selecting a foundation model, it can be customized for specific tasks through techniques such as prompt engineering, fine-tuning the model using LoRA, SFT or other methods.
```
fine-tuning can be utilized to enhance the model's performance on a specific task, requiring a high-quality dataset for it (thus, involving a data collection step). In the case of fine-tuning, there are different approaches such as fine-tuning the model, fine-tuning the instructions, or using [soft prompts](https://learnprompting.org/docs/trainable/soft_prompting). There are challenges with fine-tuning due to the large size of the model. Additionally, deploying the newly finetuned model on a new infrastructure can be difficult. To solve this problem, today, there are finetuning techniques that improve only a small subset of additional parameters to add to the existing foundational model, such as [LoRA](https://arxiv.org/abs/2106.09685). Using LoRA, it’s possible to keep the same foundation model always deployed on the infrastructure while adding the additional finetuned parameters when needed. Recently, popular proprietary models like GPT3.5 and PaLM can now be finetuned easily directly on the company platform.
```

3. Evaluations
	Evaluations for LLMs are complex because they generate free text, and its harder to t devise a metrics that can be computed via code for evaluating the free text. 
	For example, to evaluate the quality of an answer given by an LLM assistant whose job is to summarize YouTube videos, for which you don’t have reference summaries written by humans.
	Currently, organizations often resort to A/B testing to assess the effectiveness of their models, checking whether the user’s satisfaction is the same or better after the change in production.

4. Deployment and Monitoring
	- concern about LLMOps is the latency of the model
	- As the model is an autoregressive models, it takes time output a complete paragraph. This is in contrast with the most popular applications of LLMs, which want them as assistants, which, therefore, should be able to output text at a throughput similar to a user's reading speed.
	- W&B Prompts is one of tools for LLMOps.
		- W&B Prompts offers a comprehensive set of features that allow developers to visualize and inspect the execution flow of LLMs, analyze the inputs and outputs, view intermediate results, and securely manage prompts and LLM chain configurations.
		- A key component of W&B Prompts is [Trace](https://github.com/wandb/wandb), a tool that tracks and visualizes the inputs, outputs, execution flow, and model architecture of LLM chains.
		- It provides a Trace Table for an overview of the inputs and outputs of a chain, a Trace Timeline that displays the execution flow of the chain color-coded according to component types, and a Model Architecture view that provides details about the structure of the chain and the parameters used to initialize each component.
	

## Training Process

1. Dataset - curating a comprehensive database containing relevant information

```
Splitting the dataset into training and validation sets is a standard process. The training set is utilized during the training process to optimize the model's parameters. On the other hand, the validation set is used to assess the model's performance and ensure it is not overfitting by evaluating its generalization ability.
```


2. Model -The transformer has been the dominant network architecture for natural language processing tasks in recent years. It is powered by the attention mechanism, which enables the models to accurately identify the relationship between words.

3. Training - 
The first generation of foundational models like BERT were trained with Masked Language Modeling (MLM) learning objectives. This is achieved by randomly masking words from the corpus and configuring the model to predict the masked word. By employing this objective, the model gains the ability to consider the contextual information preceding and following the masked word, enabling it to make informed decisions. This objective may not be the most suitable choice for generative tasks, as ideally, the model should not have access to future words while predicting the current word.

The GPT family models used the Autoregressive learning objective. This algorithm ensures that the model consistently attempts to predict the next word without accessing the future content within the corpus. The training process will be iterative, which feeds back the generated tokens to the model to predict the next word. Masked attention ensures that, at each time step, the model is prevented from seeing future words.

To train or finetune models, you have the option to either implement the training loop using libraries such as PyTorch or utilize the `Trainer` class provided by Huggingface. The latter option enables us to easily configure different hyperparameters, log, save checkpoints, and evaluate the model.

### Popular Datasets

1. Falcon RefinedWeb
2. The Pile
3. Red Pajama Data
4. Stack Overflow Posts


## Train an LLM in the Cloud
```
An Amazon SageMaker notebook instance is a fully managed machine learning (ML) compute instance powered by Amazon Elastic Compute Cloud (Amazon EC2). This instance runs the Jupyter Notebook application, allowing you to create and manage Jupyter notebooks for data preprocessing, ML model training, and model deployment.
```

Parameters of the LLM
https://huggingface.co/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.TrainingArguments

```python
from transformers import Trainer, TrainingArguments

args = TrainingArguments(
    output_dir="GPT2-scratch-openwebtext",
    evaluation_strategy="steps",
    save_strategy="steps",
    eval_steps=500,
    save_steps=500,
    num_train_epochs=2,
    logging_steps=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=1,
    weight_decay=0.1,
    warmup_steps=100,
    lr_scheduler_type="cosine",
    learning_rate=5e-4,
    bf16=True,
    ddp_find_unused_parameters=False,
    run_name="GPT2-scratch-openwebtext",
    report_to="wandb"
)
```


# Going at Scale with LLM Training

## The Zero Redundancy Optimizer (ZeRO)

It has made it possible to train these models with lower hardware requirements.
ZeRO is a parallelized optimizer that drastically reduces the resources required for model and data parallelism while significantly increasing the number of parameters that can be trained.
ZeRO is designed to make the most of data parallelism's computational and memory resources, reducing the memory and compute requirements of each device (GPU) used for model training. It achieves this by distributing the various model training states (weights, gradients, and optimizer states) across the available devices (GPUs and CPUs) in the distributed training hardware.
As long as the aggregated device memory is large enough to share the model states, ZeRO-powered data parallelism can accommodate models of any size.

## The Stages of ZeRO

1. **Stage 1 - Optimizer State Partitioning :** **Shards optimizer states across data parallel workers/GPUs.** This results in a 4x memory reduction, with the same communication volume as data parallelism. For example, this stage can be used to train a 1.5 billion parameter GPT-2 model on eight V100 GPUs.
2. **Stage 2 - Gradient Partitioning :** **Shards optimizer states and gradients** across data parallel workers/GPUs. This leads to an 8x memory reduction, with the same communication volume as data parallelism. For example, this stage can be used to train a 10 billion parameter GPT-2 model on 32 V100 GPUs.
3. -**Stage 3 - Parameter Partitioning**: **Shards optimizer states, gradients, and model parameters across data parallel workers/GPUs.** This results in a linear memory reduction with the data parallelism degree. ZeRO can train a trillion-parameter model on about 512 NVIDIA GPUs with all three stages.
4. **Stage 3 Extra - Offloading to CPU and NVMe memory**: In addition to these stages, ZeRO-3 includes the infinity offload engine to form [ZeRO-Infinity,](https://arxiv.org/abs/2104.07857) which can offload to both CPU and [NVMe memory](https://www.pogolinux.com/blog/why-leverage-nvme-ssds-on-premise-artificial-intelligence-machine-learning/) for significant memory savings. This technique allows you to train even larger models that wouldn't fit into GPU memory. It offloads optimizer states, gradients, and parameters to the CPU, allowing you to train models with billions of parameters on a single GPU.

## DeepSpeed

DeepSpeed is a high-performance library for accelerating distributed deep learning training. It incorporates ZeRO and other state-of-the-art training techniques, such as distributed training, mixed precision, and checkpointing, through lightweight APIs compatible with PyTorch.

1. **Scale**: DeepSpeed's ZeRO stage one provides system support to run models up to 100 billion parameters, which is 10 times larger than the current state-of-the-art large models.
2. **Speed**: DeepSpeed combines ZeRO-powered data parallelism with model parallelism to achieve up to five times higher throughput over the state-of-the-art across various hardware.
3. **Cost**: The improved throughput translates to significantly reduced training costs. For instance, to train a model with 20 billion parameters, DeepSpeed requires three times fewer resources.
4. **Usability**: Only a few lines of code changes are needed to enable a PyTorch model to use DeepSpeed and ZeRO. DeepSpeed does not require a code redesign or model refactoring, and it does not put limitations on model dimensions, batch size, or any other training parameters.


## Accelerate and DeepSpeed ZeRO

The [Hugging Face Accelerate](https://huggingface.co/docs/accelerate/index) library allows you to leverage DeepSpeed's ZeRO features by making very few code changes. By using Accelerate and DeepSpeed ZeRO, we can significantly increase the maximum batch size that our hardware can handle without running into OOM errors.


## Logbook of Training Runs

Despite these libraries, there are still unexpected obstacles in the training runs. This is because there may be instabilities during training that are hard to recover from, such as spikes in the loss function.

For example, here’s a [logbook](https://github.com/huggingface/m4-logs/blob/master/memos/README.md) of the training of reproduction of [Flamingo (by Google Deepmind)](https://www.deepmind.com/blog/tackling-multiple-tasks-with-a-single-visual-language-model), an 80B parameters vision and language model, done by Hugging Face. In the following image, the second chart shows the loss function of the final model as the training progresses. Some of these spikes rapidly recovered to the original loss level, and some others diverged and never recovered.

To stabilize and continue the training, the authors usually applied a rollback, i.e., a re-start from a checkpoint a few hundred steps prior to the spike/divergence, sometimes with a decrease in the learning rate (shown in the first chart of the image).

Other times, it may be possible for the model to be stuck in a local optimum, thus requiring other rollbacks. Sometimes, memory errors may require a manual inspection