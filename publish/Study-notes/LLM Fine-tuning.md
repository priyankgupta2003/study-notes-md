- **Full Finetuning:** This method involves adjusting all the parameters in the pretrained LLM models to adapt to a specific task. While effective, it is resource-intensive and requires extensive computational power, therefore it’s rarely used.

- **Low-Rank Adaptation (LoRA):** LoRA is a technique that aims to adapt LLMs to specific tasks and datasets while simultaneously reducing computational resources and costs. By applying low-rank approximations to the downstream layers of LLMs, LoRA significantly reduces the number of parameters to be trained, thereby lowering the GPU memory requirements and training costs. 

- **Supervised Finetuning (SFT):** SFT involves doing standard supervised finetuning with a pretrained LLM on a small amount of demonstration data. This method is less resource-intensive than full finetuning but still requires significant computational power.

- **Reinforcement Learning from Human Feedback (RLHF):** RLHF is a training methodology where models are trained to follow human feedback over multiple iterations. This method can be more effective than SFT, as it allows for continuous improvement based on human feedback. We’ll also see some alternatives to RLHF, such as Direct Preference Optimization (DPO), and Reinforcement Learning from AI Feedback (RLAIF).



# LORA

- This technique tackles the issues related to the fine-tuning process, such as extensive memory demands and computational inefficiency

- LoRA introduces a compact set of parameters, referred to as **low-rank matrices**, to store the necessary changes in the model instead of altering all parameters.

	- **Maintaining Pretrained Weights**: LoRA adopts a unique strategy by preserving the pretrained weights of the model. This approach reduces the risk of catastrophic forgetting, ensuring the model maintains the valuable knowledge it gained during pretraining.

	- **Efficient Rank-Decomposition**: LoRA incorporates rank-decomposition weight matrices, known as update matrices, to the existing weights. These update matrices have significantly fewer parameters than the original model, making them highly memory-efficient. By training only these newly added weights, LoRA achieves a faster training process with reduced memory demands. These LoRA matrices are typically integrated into the attention layers of the original model.


## **Open-source Resources for LoRA**

- [**PEFT Library**](https://github.com/huggingface/peft): Parameter-efficient fine-tuning (PEFT) methods facilitate efficient adaptation of pre-trained language models to various downstream applications without fine-tuning all the model's parameters. By fine-tuning only a portion of the model's parameters, PEFT methods like LoRA, Prefix Tuning, and P-Tuning, including QLoRA, significantly reduce computational and storage costs.

- [**Lit-GPT**](https://github.com/Lightning-AI/lit-gpt)**:** Lit-GPT from LightningAI is an open-source resource designed to simplify the fine-tuning process, making it easier to apply LoRA's techniques without manually altering the core model architecture. Models available for this purpose include [Vicuna](https://lmsys.org/blog/2023-03-30-vicuna/), [Pythia](https://www.eleuther.ai/papers-blog/pythia-a-suite-for-analyzing-large-language-modelsacross-training-and-scaling), and [Falcon](https://falconllm.tii.ae/). Specific configurations can be applied to different weight matrices, and precision settings can be adjusted to manage memory consumption.

## QLoRA: An Efficient Variant of LoRA

- to save memory without sacrificing performance.
- involves backpropagating gradients through a frozen, 4-bit quantized pretrained language model into Low-Rank Adapters. This approach significantly reduces memory usage, enabling the fine-tuning of even larger models on consumer-grade GPUs
- uses a new data type known as 4-bit NormalFloat (NF4), which is optimal for normally distributed weights. It also employs double quantization to reduce the average memory footprint by quantizing the quantization constants and paged optimizers to manage memory spikes.

Steps to fine-tune

1. load the dataset
2. load the pre-trained tokenizer
3. define the formatting function called `prepare_sample_text`
4. call `ConstantLenghtDataset` function  using the combination of a tokenizer, deep lake dataset object, and formatting function.
```
The additional arguments, such as `infinite=True` ensure that the iterator will restart when all data points have been used, but there are still training steps remaining. Alongside `seq_length`, which determines the maximum sequence length, it must be completed according to the model's configuration
```

5. Initialize the model
```
The LoRA approach is employed for fine-tuning, which involves introducing new parameters to the network while keeping the base model unchanged during the tuning process. This approach has proven to be highly efficient, enabling fine-tuning of the model by training less than 1% of the total parameters. (For more details, refer to the following [post](https://medium.com/@nlpiation/pre-trained-transformers-gpt-3-2-but-1000x-smaller-cafe4269a96c).)

With the TRL library, we can seamlessly add additional parameters to the model by defining a number of configurations. The variable `r` represents the dimension of matrices, where lower values lead to fewer trainable parameters. `lora_alpha` serves as the scaling factor, while `bias` determines which bias parameters the model should train, with options of `none`, `all`, and `lora_only`. The remaining parameters are self-explanatory.
```

6. Configure the TrainingArguments
```
employ the argument `bf16=True` in order to minimize memory usage during the model's fine-tuning process
```

7. Use "weight and biases" for monitoring the training
8. use "SFTTrainer" class to tie all the components together
```
SFTTrainer accepts the model, training arguments, training dataset, and LoRA method configurations to construct the trainer object. The `packing` argument indicates that we used the `ConstantLengthDataset` class earlier to pack samples together.
```


```python
from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=lora_config,
    packing=True,
)
```

9. merge the base model with the trained LoRA layers to create a standalone model.
	1.  loading the base model
```python
from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
  "facebook/opt-1.3b", return_dict=True, torch_dtype=torch.bfloat16
)
```
2. merge the base model with the LoRA layers from the checkpoint specified using the `.from_pretrained()` method
```python
from peft import PeftModel

# Load the Lora model
model = PeftModel.from_pretrained(model, "./OPT-fine_tuned-LIMA-CPU/<desired_checkpoint>/")
model.eval()
```

3. we can use the PEFT model’s `.merge_and_unload()` method to combine the base model and LoRA layers as a standalone object.
4. save the weights using the `.save_pretrained()` method


To Inference the model, we can utilize Huggingface's `.generate()` method to interact with models effortlessly.