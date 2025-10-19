
![[Pasted image 20251014170449.png]]



### LCEL

The **LangChain Expression Language** (LCEL) takes a [declarative](https://en.wikipedia.org/wiki/Declarative_programming) approach to building new [Runnables](https://python.langchain.com/docs/concepts/runnables/) from existing Runnables.

This means that you describe what _should_ happen, rather than _how_ it should happen, allowing LangChain to optimize the run-time execution of the chains.

We often refer to a `Runnable` created using LCEL as a "chain". It's important to remember that a "chain" is `Runnable` and it implements the full [Runnable Interface](https://python.langchain.com/docs/concepts/runnables/).


## Benefits of LCEL

LangChain optimizes the run-time execution of chains built with LCEL in a number of ways:

- **Optimized parallel execution**: Run Runnables in parallel using [RunnableParallel](https://python.langchain.com/docs/concepts/lcel/#runnableparallel) or run multiple inputs through a given chain in parallel using the [Runnable Batch API](https://python.langchain.com/docs/concepts/runnables/#optimized-parallel-execution-batch). Parallel execution can significantly reduce the latency as processing can be done in parallel instead of sequentially.
- **Guaranteed Async support**: Any chain built with LCEL can be run asynchronously using the [Runnable Async API](https://python.langchain.com/docs/concepts/runnables/#asynchronous-support). This can be useful when running chains in a server environment where you want to handle large number of requests concurrently.
- **Simplify streaming**: LCEL chains can be streamed, allowing for incremental output as the chain is executed. LangChain can optimize the streaming of the output to minimize the time-to-first-token(time elapsed until the first chunk of output from a [chat model](https://python.langchain.com/docs/concepts/chat_models/) or [llm](https://python.langchain.com/docs/concepts/text_llms/) comes out).

