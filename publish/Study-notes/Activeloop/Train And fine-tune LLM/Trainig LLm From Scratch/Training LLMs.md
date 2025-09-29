
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

- [](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Trainer.model)**model** ([PreTrainedModel](https://huggingface.co/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel) or `torch.nn.Module`, _optional_) — The model to train, evaluate or use for predictions. If not provided, a `model_init` must be passed.
    
    [Trainer](https://huggingface.co/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer) is optimized to work with the [PreTrainedModel](https://huggingface.co/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel) provided by the library. You can still use your own models defined as `torch.nn.Module` as long as they work the same way as the 🤗 Transformers models.
    
- [](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Trainer.args)**args** ([TrainingArguments](https://huggingface.co/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.TrainingArguments), _optional_) — The arguments to tweak for training. Will default to a basic instance of [TrainingArguments](https://huggingface.co/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.TrainingArguments) with the `output_dir` set to a directory named _tmp_trainer_ in the current directory if not provided.
- [](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Trainer.data_collator)**data_collator** (`DataCollator`, _optional_) — The function to use to form a batch from a list of elements of `train_dataset` or `eval_dataset`. Will default to [default_data_collator()](https://huggingface.co/docs/transformers/v4.56.2/en/main_classes/data_collator#transformers.default_data_collator) if no `processing_class` is provided, an instance of [DataCollatorWithPadding](https://huggingface.co/docs/transformers/v4.56.2/en/main_classes/data_collator#transformers.DataCollatorWithPadding) otherwise if the processing_class is a feature extractor or tokenizer.
- [](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Trainer.train_dataset)**train_dataset** (Union[`torch.utils.data.Dataset`, `torch.utils.data.IterableDataset`, `datasets.Dataset`], _optional_) — The dataset to use for training. If it is a [Dataset](https://huggingface.co/docs/datasets/v4.1.0/en/package_reference/main_classes#datasets.Dataset), columns not accepted by the `model.forward()` method are automatically removed.
    
    Note that if it’s a `torch.utils.data.IterableDataset` with some randomization and you are training in a distributed fashion, your iterable dataset should either use a internal attribute `generator` that is a `torch.Generator` for the randomization that must be identical on all processes (and the Trainer will manually set the seed of this `generator` at each epoch) or have a `set_epoch()` method that internally sets the seed of the RNGs used.
    
- [](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Trainer.eval_dataset)**eval_dataset** (Union[`torch.utils.data.Dataset`, dict[str, `torch.utils.data.Dataset`, `datasets.Dataset`]), _optional_) — The dataset to use for evaluation. If it is a [Dataset](https://huggingface.co/docs/datasets/v4.1.0/en/package_reference/main_classes#datasets.Dataset), columns not accepted by the `model.forward()` method are automatically removed. If it is a dictionary, it will evaluate on each dataset prepending the dictionary key to the metric name.
- [](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Trainer.processing_class)**processing_class** (`PreTrainedTokenizerBase` or `BaseImageProcessor` or `FeatureExtractionMixin` or `ProcessorMixin`, _optional_) — Processing class used to process the data. If provided, will be used to automatically process the inputs for the model, and it will be saved along the model to make it easier to rerun an interrupted training or reuse the fine-tuned model. This supersedes the `tokenizer` argument, which is now deprecated.
- [](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Trainer.model_init)**model_init** (`Callable[[], PreTrainedModel]`, _optional_) — A function that instantiates the model to be used. If provided, each call to [train()](https://huggingface.co/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer.train) will start from a new instance of the model as given by this function.
    
    The function may have zero argument, or a single one containing the optuna/Ray Tune/SigOpt trial object, to be able to choose different architectures according to hyper parameters (such as layer count, sizes of inner layers, dropout probabilities etc).
    
- [](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Trainer.compute_loss_func)**compute_loss_func** (`Callable`, _optional_) — A function that accepts the raw model outputs, labels, and the number of items in the entire accumulated batch (batch_size * gradient_accumulation_steps) and returns the loss. For example, see the default [loss function](https://github.com/huggingface/transformers/blob/052e652d6d53c2b26ffde87e039b723949a53493/src/transformers/trainer.py#L3618) used by [Trainer](https://huggingface.co/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer).
- [](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Trainer.compute_metrics)**compute_metrics** (`Callable[[EvalPrediction], Dict]`, _optional_) — The function that will be used to compute metrics at evaluation. Must take a [EvalPrediction](https://huggingface.co/docs/transformers/v4.56.2/en/internal/trainer_utils#transformers.EvalPrediction) and return a dictionary string to metric values. _Note_ When passing TrainingArgs with `batch_eval_metrics` set to `True`, your compute_metrics function must take a boolean `compute_result` argument. This will be triggered after the last eval batch to signal that the function needs to calculate and return the global summary statistics rather than accumulating the batch-level statistics
- [](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Trainer.callbacks)**callbacks** (List of [TrainerCallback](https://huggingface.co/docs/transformers/v4.56.2/en/main_classes/callback#transformers.TrainerCallback), _optional_) — A list of callbacks to customize the training loop. Will add those to the list of default callbacks detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).
    
    If you want to remove one of the default callbacks used, use the [Trainer.remove_callback()](https://huggingface.co/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer.remove_callback) method.
    
- [](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Trainer.optimizers)**optimizers** (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, _optional_, defaults to `(None, None)`) — A tuple containing the optimizer and the scheduler to use. Will default to an instance of `AdamW` on your model and a scheduler given by [get_linear_schedule_with_warmup()](https://huggingface.co/docs/transformers/v4.56.2/en/main_classes/optimizer_schedules#transformers.get_linear_schedule_with_warmup) controlled by `args`.
- [](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Trainer.optimizer_cls_and_kwargs)**optimizer_cls_and_kwargs** (`tuple[Type[torch.optim.Optimizer], dict[str, Any]]`, _optional_) — A tuple containing the optimizer class and keyword arguments to use. Overrides `optim` and `optim_args` in `args`. Incompatible with the `optimizers` argument.
    
    Unlike `optimizers`, this argument avoids the need to place model parameters on the correct devices before initializing the Trainer.
    
- [](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Trainer.preprocess_logits_for_metrics)**preprocess_logits_for_metrics** (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`, _optional_) — A function that preprocess the logits right before caching them at each evaluation step. Must take two tensors, the logits and the labels, and return the logits once processed as desired. The modifications made by this function will be reflected in the predictions received by `compute_metrics`.