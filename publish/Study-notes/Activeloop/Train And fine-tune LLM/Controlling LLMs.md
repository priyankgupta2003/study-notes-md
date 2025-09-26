
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

1. **Logits** : When a language model makes a prediction, it generates a vector of logits, one for the next possible token. These logits represent the raw, unnormalized prediction scores for each token.
2. **Softmax**: The softmax function is applied to these logits to convert them into probabilities. The softmax function also ensures that these probabilities sum up to 1.
3. **Temperature**: The temperature parameter is used to control the randomness of the model's output. It does this by dividing the logits by the temperature value before the softmax step.
	- **High Temperature (e.g., > 1)**: The logits are scaled down, which makes the softmax output more uniform. This means the model is more likely to pick less likely words, resulting in more diverse and "creative" outputs, but potentially with more mistakes or nonsensical phrases.
	- **Low Temperature (e.g., < 1)**: The logits are scaled up, which makes the softmax output more peaked. This means the model is more likely to pick the most likely word. The output will be more focused and conservative, sticking closer to the most probable outputs but potentially less diverse.
	- **Temperature = 1**: The logits are not scaled, preserving the original probabilities. This is a kind of "neutral" setting.

## Stop Sequences

Stop sequences are specific sets of character sequences that halt the text generation process once they appear in the output. They offer a way to guide the length and structure of the generated text, providing a form of control over the output.

### Frequency and Presence Penalties

Frequency and presence penalties are used to discourage or encourage the repetition of certain words in the generated text. A frequency penalty reduces the likelihood of the model repeating tokens that have appeared frequently, while a presence penalty discourages the model from repeating any token that has already appeared in the generated text.


