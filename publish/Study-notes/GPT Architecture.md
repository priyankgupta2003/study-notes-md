
- The **encoder-only** category focused on extracting meaningful representations from input data. An example model of this category is [BERT](https://arxiv.org/abs/1810.04805).
- The **encoder-decoder** category enabled sequence-to-sequence tasks such as translation and summarization or training multimodal models like caption generators. An example model of this category is [BART](https://arxiv.org/abs/1910.13461).
- The **decoder-only** category specializes in generating outputs based on given instructions, as we have in Large Language Models. An example model of this category is [GPT](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf).

### Self-Attention Mechanism

At the core of the Transformer model lies the self-attention mechanism, which calculates a weighted sum of the embeddings of all words in a sentence for each word. These weights are determined based on some learned “attention” scores between words. The terms with higher relevance to one another will receive higher “attention” weights.

Based on the inputs, this is implemented using Query, Key, and Value vectors. Here is a brief description of each vector.

- **Query Vector**: It represents the word or token for which the attention weights are being calculated. The Query vector determines which parts of the input sequence should receive more attention. Multiplying word embeddings with the Query vector is like asking, **"What should I pay attention to?"**
- **Key Vector**: It represents the set of words or tokens in the input sequence that are compared with the Query. The Key vector helps identify the relevant or essential information in the input sequence. Multiplying word embeddings with the Key vector is like asking, **"What is important to consider?"**
- **Value Vector**: It contains the input sequence's associated information or features for each word or token. The Value vector provides the actual data that will be weighted and combined based on the attention weights calculated between the Query and Key. The Value vector answers the question, **"What information do we have?"**

The self-attention mechanism enabled the models to highlight the important parts of the content for the task. **It is helpful in encoder-only or decoder-only models to create a powerful representation of the input. The text can be transformed into embeddings for encoder-only scenarios, whereas the text is generated for decoder-only models.**

The effectiveness of the attention mechanism significantly increases when applied in a multi-head setting. In this configuration, multiple attention components process the same information, with each head learning to focus on distinct aspects of the text, such as verbs, nouns, numbers, and more, throughout the training process.

## **Transformer Architectures**


1. **Encoder- Decoder Architecture**

![[Pasted image 20250916220040.png]]

The encoder-decoder, also known as the **full transformer architecture**, comprises multiple stacked encoder components connected to several stacked decoder components through a cross-attention mechanism.

It is notably **well-suited for sequence-to-sequence (i.e., handling text as both input and output) tasks** such as translation or summarization, mainly when designing models with multi-modality, like image captioning with the image as input and the corresponding caption as the expected output. Cross-attention will help the decoder focus on the most important part of the content during the generation process.
eg: BART pre-trained model.

2. **Encoder Only Architecture**
![[Pasted image 20250916220429.png]]

the encoder-only models are formed by stacking multiple encoder components. As the encoder output cannot be connected to another decoder, its output can be directly used as a text-to-vector method, for instance, to measure similarity.
Alternatively, it can be combined with a classification head (feedforward layer) on top to facilitate label prediction.

The primary distinction in the encoder-only architecture lies in the absence of the Masked Self-Attention layer. As a result, the encoder can handle the entire input simultaneously. This differs from decoders, where future tokens need to be masked during training to prevent “cheating” when generating new tokens. Due to this property, they are ideally suited for creating representations from a document while retaining complete information.

``` 
The BERT paper (or an improved variant like RoBERTa) introduced a widely recognized pre-trained model that significantly improved the state-of-the-art scores on numerous NLP tasks. The model undergoes pre-training with two learning objectives:

1. Masked Language Modeling: masking random tokens from the input and attempting to predict them.
2. Next Sentence Prediction: Present sentences in pairs and assess the likelihood of the second sentence in the subsequent sequence of the first sentence.
```

3. **Decoder-Only**

![[Pasted image 20250916224703.png]]

The decoder-only networks continue to serve as the foundation for most large language models today, with slight variations in some instances. Because of the implementation of masked self-attention, their primary use case revolves around the next-token-prediction task, which sparked the concept of prompting.

scaling up the decoder-only models can significantly enhance the network's language understanding and generalization capabilities. As a result, they can excel at a diverse range of tasks simply by using different prompts.
Large pre-trained models like GPT-4 and LLaMA 2 exhibit the ability to perform tasks such as classification, summarization, translation, etc., by leveraging the appropriate prompt.

The large language models, such as those in the GPT family, undergo pre-training using the Causal Language Modeling objective. This means the model aims to predict the next word, while the attention mechanism can only attend to previous tokens on the left. This implies that the model can solely rely on the previous context to predict the next token and is unable to peek at future tokens, preventing any form of cheating.


### **GPT Architecture**
The GPT family comprises decoder-only models, wherein each block in the stack is comprised of a self-attention mechanism and a position-wise fully connected feed-forward network.

The **self-attention mechanism**, also known as scaled dot-product attention, allows the model to weigh the importance of each word in the input when generating the next word in the sequence. It computes a weighted sum of all the words in the sequence, where the weights are determined by the attention scores.

The critical aspect to focus on is the addition of “masking” to the self-attention that prevents the model from attending to certain positions/words.

![[Pasted image 20250921181806.png]]
> Illustrating which tokens are attended to by masked self-attention at a particular timestamp. (Image taken from [NLPiation](https://medium.com/mlearning-ai/what-are-the-differences-in-pre-trained-transformer-base-models-like-bert-distilbert-xlnet-gpt-4b3ea30ef3d7)) As you see in the figure, we pass the whole sequence to the model, but the model at timestep 5 tries to predict the next token by only looking at the previously generated tokens, masking the future tokens. This prevents the model from “cheating” by predicting tokens leveraging future tokens.

```
import numpy as np

def self_attention(query, key, value, mask=None):
    # Compute attention scores
    scores = np.dot(query, key.T)
    
    if mask is not None:
        # Apply mask by setting masked positions to a large negative value
        scores = scores + mask * -1e9
    
    # Apply softmax to obtain attention weights
    attention_weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
    
    # Compute weighted sum of value vectors
    output = np.dot(attention_weights, value)
    
    return output
```

The first step is to compute a Query, Key, and Value vector for each word in the input sequence using separate learned linear transformations of the input vector. It is a simple feedforward linear layer that the model learns during training.

Then, we can **calculate the attention scores by taking the dot product of its Query vector with the Key vector of every other word**. Currently, the application of masking is feasible by setting the scores in specific locations to a large negative number. This effectively informs the model that those words are unimportant and should be disregarded during attention. To get the attention weights, apply the SoftMax function to the attention scores to convert them into probabilities. This gives the weights of the input words and effectively turns the significant negative scores to zero. Lastly, multiply each Value vector by its corresponding weight and sum them up. This produces the output of the masked self-attention mechanism for the word.

The provided code snippet illustrates the process of a single self-attention head, but in reality, each layer contains multiple heads, which could range from 16 to 32 heads, depending on the architecture. These heads operate simultaneously to enhance the model's performance.

