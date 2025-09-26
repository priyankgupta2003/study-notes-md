
### Decoding Methods

Decoding methods are fundamental strategies used by LLMs to generate text

At each decoding step, the LLM gives a score to each of its vocabulary tokens. A high score is related to a high probability of that token being the next token, according to the patterns learned by the model during training.

```
Higher the score, higher the probability of that token being the next token
```

### Greedy Search

Greedy Search is the simplest of all the decoding methods.

With Greedy Search, the model selects the token with the highest probability as its next output token. While this method is computationally efficient, it can often result in repetitive or less optimal responses due to its focus on immediate reward rather than long-term outcomes.

### Sampling

Sampling introduces randomness into the text generation process, where the model randomly selects the next word based on its probability distribution. This method allows for more diverse and varied output but can sometimes produce less coherent or logical text.

### Beam Search

This method selects the top N (with N being a parameter) candidate subsequent tokens with the highest probabilities at each step, but only up to a certain number of steps. In the end, the model generates the sequence of tokens (i.e., the beam) with the highest joint probability.

advantage: 
- reduces the search space and produces more consistent results.
cons:
- this method might be slower and lead to suboptimal outputs as it can miss high-probability words hidden behind a low-probability word.

### Top-K Sampling

This model narrow downs the sampling pool to the top K of most probable words. 

This method provides a balance between diversity and relevance by limiting the sampling space, thus offering more control over the generated text.

### Top-p (Nucleus Sampling)

It selects the words from the smallest possible set of tokens whose cumulative probability exceeds a certain threshold P.

pros:
- This method offers fine-grained control and avoids the inclusion of rare or low-probability tokens. 
cons
- The dynamically determined shortlist sizes can sometimes be a limitation.


## Parameters That Influence Text Generation

### Temperature
The temperature influences the randomness or determinism of the generated text.

lower values -> more deterministic and focused
higher value -> increases the randomness, more diverse output

It controls the randomness of predictions by scaling the logits before applying softmax during the text generation process.

