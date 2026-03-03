# mem0 Chat API Integration Guide

## Overview

mem0's Chat API (`mem0.chat.completions.create()`) provides agentic answering with automatic memory retrieval. This guide explains how to integrate it with custom VLLM endpoints for your PersonalMemoryQA system.

## Key Capabilities

### ✅ Custom VLLM Endpoint Support
**Yes!** mem0 uses `litellm` which supports OpenAI-compatible endpoints (including VLLM):

```python
response = mem0.chat.completions.create(
    model="openai/Qwen/Qwen2.5-3B-Instruct",  # litellm format
    messages=messages,
    user_id="user_123",
    base_url="http://127.0.0.1:8000/v1",  # Your VLLM endpoint
    api_key="EMPTY",  # VLLM doesn't need real keys
    temperature=0.7,
    max_tokens=512
)
```

### ✅ Single-Round QA Support
**Yes!** Just pass a single message:

```python
messages = [
    {"role": "user", "content": "When did I visit the Golden Gate Bridge?"}
]
```

The "last 6 messages" retrieval logic still applies, but for single-round QA it just uses your one question.

## How It Works

### 1. Memory Retrieval Process

```python
def _fetch_relevant_memories(self, messages, user_id, agent_id, run_id, filters, limit):
    # Uses last 6 messages (for single-round QA, that's just your question)
    message_input = [f"{message['role']}: {message['content']}" for message in messages][-6:]
    return self.mem0_client.search(
        query="\n".join(message_input),
        user_id=user_id,
        filters=filters,
        limit=limit  # Number of memories to retrieve
    )
```

### 2. Prompt Augmentation

The retrieved memories are automatically injected into the final user message:

```
System: You are an expert at answering questions based on the provided memories...

User: - Relevant Memories/Facts: {retrieved_memories}
      - Entities: {entities}
      - User Question: {your_question}
```

### 3. Async Memory Storage

When you send a message, mem0 **asynchronously** stores it as a new memory in the background (does NOT block the response).

## Integration Options for PersonalMemoryQA

### Option 1: Pure Chat API Mode (Recommended)

Use mem0 Chat API for **both indexing and answering**:

```python
# During indexing: Add memories with --no-mem0-infer
mem0_client.add(memory_text, user_id="user", metadata={"source": "email"})

# During QA: Use Chat API
response = mem0.chat.completions.create(
    model="openai/Qwen/Qwen2.5-3B-Instruct",
    messages=[{"role": "user", "content": question}],
    user_id="user",
    base_url="http://127.0.0.1:8000/v1",
    api_key="EMPTY",
    limit=10  # Top-k memories to retrieve
)

answer = response.choices[0].message.content
```

**Pros:**
- Simple integration
- Automatic memory retrieval
- Automatic prompt augmentation
- Single API call

**Cons:**
- Less control over prompt format
- Uses last 6 messages for retrieval (may not be optimal)
- Auto-stores user questions as memories (may not be desired)

### Option 2: Hybrid Mode (More Control)

Use mem0 for **retrieval only**, then call your own VLLM answerer:

```python
# Retrieve relevant memories
memories = mem0_client.search(
    query=question,
    user_id="user",
    limit=10
)

# Build custom prompt
memory_context = "\n".join([m["memory"] for m in memories["results"]])
custom_prompt = f"""Based on the following memories, answer the question.

Memories:
{memory_context}

Question: {question}
"""

# Call your existing VLLM answerer
answer = your_vllm_answerer(custom_prompt)
```

**Pros:**
- Full control over prompt format
- No auto-storage of questions
- Can customize retrieval query
- Compatible with existing evaluation pipeline

**Cons:**
- More code
- Manual prompt engineering
- Doesn't use mem0's MEMORY_ANSWER_PROMPT

### Option 3: Extended Chat API Mode (Experimental)

Subclass `Completions` to customize behavior:

```python
from mem0.proxy.main import Completions, Chat, Mem0 as BaseMem0

class CustomCompletions(Completions):
    def _fetch_relevant_memories(self, messages, user_id, agent_id, run_id, filters, limit):
        # Custom retrieval logic
        # Use ALL messages, not just last 6
        message_input = [f"{message['role']}: {message['content']}" for message in messages]
        return self.mem0_client.search(query="\n".join(message_input), ...)
    
    def _async_add_to_memory(self, messages, user_id, agent_id, run_id, metadata, filters):
        # Disable auto-storage of questions
        pass

class CustomMem0(BaseMem0):
    def __init__(self, config=None):
        super().__init__(config=config)
        self.chat.completions = CustomCompletions(self.mem0_client)
```

## litellm Model Format Reference

### VLLM (OpenAI-compatible API)
```python
model="openai/<model_name>"
base_url="http://127.0.0.1:8000/v1"
api_key="EMPTY"
```

### Other Providers
```python
# OpenAI (default)
model="gpt-4"
api_key="sk-..."

# Anthropic
model="claude-3-opus-20240229"
api_key="sk-ant-..."

# Azure OpenAI
model="azure/<deployment_name>"
base_url="https://<resource>.openai.azure.com"
api_key="..."
api_version="2024-02-01"
```

See: https://docs.litellm.ai/docs/providers/

## Retrieval vs Answering Configuration

**Important distinction:**

### mem0 Config (`Memory.from_config()`)
Used for **memory operations** (add, search, extraction):
```python
config = {
    "llm": {  # For memory EXTRACTION during add()
        "provider": "openai",
        "config": {"model": "gpt-4"}
    },
    "embedder": {  # For vector embedding
        "provider": "huggingface",
        "config": {"model": "all-MiniLM-L6-v2"}
    }
}
```

### Chat API Parameters (`chat.completions.create()`)
Used for **answering**:
```python
mem0.chat.completions.create(
    model="openai/Qwen/Qwen2.5-3B-Instruct",  # Answering LLM
    base_url="http://127.0.0.1:8000/v1",
    api_key="EMPTY"
)
```

These are **completely separate**:
- Config LLM: Used during indexing for fact extraction
- Chat API model: Used during QA for answering

## Example Integration into mem0_baseline.py

```python
def answer_with_chat_api(
    self,
    question: str,
    user_id: str = "default_user",
    limit: int = 10,
    model: str = "openai/Qwen/Qwen2.5-3B-Instruct",
    base_url: str = "http://127.0.0.1:8000/v1",
    temperature: float = 0.7,
    max_tokens: int = 512
) -> str:
    """
    Answer question using mem0's Chat API with custom VLLM endpoint.
    
    Args:
        question: User question
        user_id: User ID for memory scope
        limit: Number of memories to retrieve
        model: Model name in litellm format
        base_url: VLLM endpoint URL
        temperature: Sampling temperature
        max_tokens: Max response length
        
    Returns:
        Generated answer
    """
    from mem0 import Mem0
    
    # Initialize mem0 with existing config
    mem0 = Mem0(config=self.config)
    
    # Single-round QA
    messages = [{"role": "user", "content": question}]
    
    response = mem0.chat.completions.create(
        model=model,
        messages=messages,
        user_id=user_id,
        base_url=base_url,
        api_key="EMPTY",
        temperature=temperature,
        max_tokens=max_tokens,
        limit=limit
    )
    
    return response.choices[0].message.content
```

## Testing

Run the test script:
```bash
# Make sure VLLM server is running with Qwen2.5-3B-Instruct
python scripts/QA_Agent/mem0/test_mem0_chat_api.py
```

## Recommendations

1. **For PersonalMemoryQA baseline comparison:**
   - Use **Option 2 (Hybrid Mode)** for fair comparison
   - This separates retrieval from answering
   - Allows consistent evaluation with other baselines

2. **For production/demo:**
   - Use **Option 1 (Pure Chat API)** for simplicity
   - Easier to maintain and explain
   - Good user experience with conversation support

3. **For research/experiments:**
   - Use **Option 3 (Extended)** to customize behavior
   - Study impact of retrieval strategy (last-6 vs all vs custom)
   - Compare prompt formats

## Limitations

1. **Last 6 messages for retrieval:**
   - Default behavior uses last 6 messages
   - For single-round QA, this is fine (only 1 message)
   - For multi-turn, may miss context from earlier turns
   - Can be customized via subclassing

2. **Auto-storage of questions:**
   - Every user question is stored as a memory
   - May pollute memory store with metadata-only questions
   - Can be disabled via subclassing

3. **Function calling requirement:**
   - Chat API checks `litellm.supports_function_calling(model)`
   - Most modern models support this (GPT-4, Claude, Qwen, etc.)
   - If your model doesn't, you'll get an error

## Next Steps

1. Run test script to verify VLLM endpoint compatibility
2. Choose integration option based on your needs
3. Implement in `mem0_baseline.py`
4. Compare with existing answerer implementations
5. Run evaluation on PersonalMemoryQA dataset
