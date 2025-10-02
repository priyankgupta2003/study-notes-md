
## **Architecture**

![[Pasted image 20250916165607.png]]

![[Pasted image 20250916165724.png]]


```
# 1. Character-based Splitting
# Splits text into chunks based on a specified number of characters.
# Useful for consistent chunk sizes regardless of content structure.
print("\n--- Using Character-based Splitting ---")
char_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
char_docs = char_splitter.split_documents(documents)
create_vector_store(char_docs, "chroma_db_char")

# 2. Sentence-based Splitting
# Splits text into chunks based on sentences, ensuring chunks end at sentence boundaries.
# Ideal for maintaining semantic coherence within chunks.
print("\n--- Using Sentence-based Splitting ---")
sent_splitter = SentenceTransformersTokenTextSplitter(chunk_size=1000)
sent_docs = sent_splitter.split_documents(documents)
create_vector_store(sent_docs, "chroma_db_sent")

# 3. Token-based Splitting
# Splits text into chunks based on tokens (words or subwords), using tokenizers like GPT-2.
# Useful for transformer models with strict token limits.
print("\n--- Using Token-based Splitting ---")
token_splitter = TokenTextSplitter(chunk_overlap=0, chunk_size=512)
token_docs = token_splitter.split_documents(documents)
create_vector_store(token_docs, "chroma_db_token")

# 4. Recursive Character-based Splitting
# Attempts to split text at natural boundaries (sentences, paragraphs) within character limit.
# Balances between maintaining coherence and adhering to character limits.
print("\n--- Using Recursive Character-based Splitting ---")
rec_char_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=100)
rec_char_docs = rec_char_splitter.split_documents(documents)
create_vector_store(rec_char_docs, "chroma_db_rec_char")

# 5. Custom Splitting
# Allows creating custom splitting logic based on specific requirements.
# Useful for documents with unique structure that standard splitters can't handle.
print("\n--- Using Custom Splitting ---")

```


```
# 1. OpenAI Embeddings
# Uses OpenAI's embedding models.
# Useful for general-purpose embeddings with high accuracy.
# Note: The cost of using OpenAI embeddings will depend on your OpenAI API usage and pricing plan.
# Pricing: https://openai.com/api/pricing/
print("\n--- Using OpenAI Embeddings ---")
openai_embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
create_vector_store(docs, openai_embeddings, "chroma_db_openai")

# 2. Hugging Face Transformers
# Uses models from the Hugging Face library.
# Ideal for leveraging a wide variety of models for different tasks.
# Note: Running Hugging Face models locally on your machine incurs no direct cost other than using your computational resources.
# Note: Find other models at https://huggingface.co/models?other=embeddings
print("\n--- Using Hugging Face Transformers ---")
huggingface_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)
create_vector_store(docs, huggingface_embeddings, "chroma_db_huggingface")
```

```
# 1. Similarity Search
# This method retrieves documents based on vector similarity.
# It finds the most similar documents to the query vector based on cosine similarity.
# Use this when you want to retrieve the top k most similar documents.
print("\n--- Using Similarity Search ---")
query_vector_store("chroma_db_with_metadata", query,
                   embeddings, "similarity", {"k": 3})

# 2. Max Marginal Relevance (MMR)
# This method balances between selecting documents that are relevant to the query and diverse among themselves.
# 'fetch_k' specifies the number of documents to initially fetch based on similarity.
# 'lambda_mult' controls the diversity of the results: 1 for minimum diversity, 0 for maximum.
# Use this when you want to avoid redundancy and retrieve diverse yet relevant documents.
# Note: Relevance measures how closely documents match the query.
# Note: Diversity ensures that the retrieved documents are not too similar to each other,
#       providing a broader range of information.
print("\n--- Using Max Marginal Relevance (MMR) ---")
query_vector_store(
    "chroma_db_with_metadata",
    query,
    embeddings,
    "mmr",
    {"k": 3, "fetch_k": 20, "lambda_mult": 0.5},
)

# 3. Similarity Score Threshold
# This method retrieves documents that exceed a certain similarity score threshold.
# 'score_threshold' sets the minimum similarity score a document must have to be considered relevant.
# Use this when you want to ensure that only highly relevant documents are retrieved, filtering out less relevant ones.
print("\n--- Using Similarity Score Threshold ---")
query_vector_store(
    "chroma_db_with_metadata",
    query,
    embeddings,
    "similarity_score_threshold",
    {"k": 3, "score_threshold": 0.1},
)
```


```
def continual_chat():
    print("Start chatting with the AI! Type 'exit' to end the conversation.")
    chat_history = []  # Collect chat history here (a sequence of messages)
    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break
        # Process the user's query through the retrieval chain
        result = rag_chain.invoke({"input": query, "chat_history": chat_history})
        # Display the AI's response
        print(f"AI: {result['answer']}")
        # Update the chat history
        chat_history.append(HumanMessage(content=query))
        chat_history.append(SystemMessage(content=result["answer"]))
```

