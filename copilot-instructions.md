# Copilot Instructions: Mira-LLEmery Architecture

## Overview
Mira-LLEmery is a FastAPI server that mimics Ollama's API while using a local GGUF model (Dolphin Mistral 24B) via LlamaCPP with Memary agent for context/memory augmentation. The system is designed to prevent multi-turn hallucinations and provide OpenAI-compatible endpoints.

## Core Architecture
First and foremost use uv for everything.  Check the Makefile a lot is auotmated.

### 1. Model Stack
- **Model**: Dolphin Mistral 24B (GGUF format via LlamaCPP)
- **GPU**: Full offload with RTX 4090 (all layers on GPU)
- **Memory System**: Memary agent for conversation history and entity tracking
- **API**: FastAPI with OpenAI-compatible endpoints

### 2. Key Components

#### LLM Configuration (`main.py:210-226`)
```python
llm = LlamaCPP(
    model_path=model_path,
    temperature=0.1,  # Low for factual responses
    model_kwargs={
        "n_gpu_layers": -1,  # Full GPU offload
        "top_k": 40,        # Limit vocabulary (40-50 recommended)
        "top_p": 0.9,       # Nucleus sampling (0.7-0.9 recommended)
        "repeat_penalty": 1.1,
        "max_tokens": 200,  # Strict limit (100-300 range)
        "stop": ["[/INST]", "\nUser:", "\nuser:", "\nAssistant:", "\nassistant:", "\nHuman:", "\nhuman:"],
        "frequency_penalty": 0.1,
        "presence_penalty": 0.1,
        "chat_format": "mistral-instruct",
    }
)
```

#### Memary Agent Integration (`main.py:305-355`)
- **Purpose**: Provides conversation history, entity tracking, and user/system personas
- **Key Fix**: Removes problematic user persona injection that caused multi-turn hallucinations
- **Memory Files**: 
  - `past_chat.json` - conversation history
  - `memory_stream.json` - processed memories
  - `entity_knowledge_store.json` - entity relationships
  - `system_persona.txt` - system prompt/persona
  - `user_persona.txt` - dynamic user context

## Critical Anti-Hallucination Mechanisms

### 1. Multi-Turn Detection & Prevention
**Problem**: LLM generates simulated conversations instead of single responses
**Solution**: Multi-layered prevention system

#### Stream-Level Detection (`main.py:650-670`)
```python
# Case-insensitive patterns for role indicators with punctuation
multi_turn_patterns = [
    r'\nuser:', r'\nassistant:', r'\nUser:', r'\nAssistant:',
    r'\nhuman:', r'\nHuman:', r'\nai:', r'\nAI:',
    r'\[INST\]', r'\[/INST\]',
    r'User said:', r'Assistant:', r'Response:'
]

# Case-sensitive patterns for standalone role words (avoid "AI" false positives)
end_role_patterns = [r'\s+user\s*$', r'\s+assistant\s*$', r'\s+ai\s*$']
```
- Monitors streaming output token-by-token
- Immediately stops stream when multi-turn patterns detected
- **ENHANCED**: Separate case-sensitive detection for end patterns
- **FIXED**: Prevents false positives with legitimate content like "AI"
- Cleans response before saving to memory

#### Response Cleaning (`main.py:130-175`)
- Removes conversation transcripts from LLM output
- Enhanced patterns for Mistral-specific markers
- **NEW**: Emotion marker detection and extraction ([neutral], [angry], etc.)
- **NEW**: Catches standalone role words ("user", "assistant", etc.) at end of responses
- **FIXED**: Case-sensitive matching for role words to avoid false positives (e.g., "AI" in "dive into AI")
- **NEW**: Handles cases where role words appear without colons or newlines
- Logs all cleaning actions with `repr()` for debugging

#### Broken Response Detection & Recovery (`main.py:165-210`)
- **`is_broken_response()`**: Detects various broken response patterns:
  - Responses too short (< 5 characters)
  - Just role words ("user", "assistant", etc.)
  - Multi-turn hallucinations starting with role patterns
  - Generic "role: content" patterns at response start
- **`generate_recovery_response()`**: Provides graceful recovery messages:
  - "I'm sorry, I lost my thought there. Could you repeat your question?"
  - "I apologize for the confusion. Could you please rephrase that?"
  - Random selection from multiple recovery options
- **Integration**: Applied in streaming, non-streaming, and bypass handlers
- **Memory Protection**: Broken responses are not saved to memory

#### System Prompt Enhancement (`main.py:490-510`)
- Adds comprehensive anti-multi-turn instructions
- Enforces single response requirement
- Applied to both API-provided and default system prompts

### 2. Mistral-Specific Optimizations
Based on Mistral best practices guide:

- **Temperature**: 0.1 (vs default 0.7) for deterministic output
- **Top-k**: 40 for focused vocabulary
- **Top-p**: 0.9 for nucleus sampling
- **Max tokens**: 200 (strict limit prevents rambling)
- **Stop sequences**: Comprehensive list prevents self-dialogue
- **Chat format**: `mistral-instruct` template

### 3. Context Management
**Memary Agent Fixes**:
- Removes user persona as "user" message (confused LLM about speaker)
- Moves user persona to system context instead
- Maintains conversation history without role confusion
- **NEW**: Configurable history message limit (MAX_HISTORY_MESSAGES = 30)
  - Research shows LLM degradation after ~30 messages in context
  - Only affects messages sent to LLM, full history preserved in storage
  - All conversations saved indefinitely for memory/search purposes
  - Configurable via MAX_HISTORY_MESSAGES constant in main.py

## API Endpoints

### Primary Endpoints
1. **`POST /v1/chat/completions`** - Main chat endpoint
   - Supports streaming (`stream: true`) and non-streaming
   - Memory augmentation via Memary agent
   - Bypass mode (`bypass_memory: true`) for direct LLM testing
   - OpenAI-compatible request/response format

2. **`POST /v1/completions`** - Text completion
3. **`POST /v1/embeddings`** - Text embeddings
4. **`GET /v1/models`** - Model listing
5. **`POST /regenerate-persona`** - Dynamic persona generation

### Testing & Debug Endpoints
- **`GET /health`** - Health check
- **`GET /metrics`** - Prometheus metrics
- **`POST /props`** - Properties update (dummy)

## Memory & Persona System

### Dynamic Persona Generation (`main.py:230-295`)
- Analyzes conversation history and memory stream
- Generates concise user persona using LLM
- Updates `user_persona.txt` automatically
- Triggered at startup and via `/regenerate-persona`

### Memory Flow
1. User message → Memary agent processes → Adds context
2. System persona + User persona + Limited history → LLM context
3. LLM response → Cleaned → Saved to memory
4. Memory persisted to JSON files (full history preserved)
5. **Context Limiting**: Only last N messages sent to LLM (configurable via MAX_HISTORY_MESSAGES)
6. **Full Storage**: All conversations saved indefinitely for memory/search

## Logging & Debugging

### Rich Console Logging
- **Color-coded by role**: System (blue), User (green), Assistant (magenta)
- **Full content display**: No truncation for debugging
- **Multi-turn detection alerts**: Red warnings with context snippets
- **Memory context visibility**: Shows exact prompts sent to LLM
- **Raw LLM logging**: `log_raw_llm_response()` shows unparsed model output
- **Broken response alerts**: Red alerts when recovery responses are triggered

### Log Categories
- `[CHAT REQUEST]` - Incoming API requests
- `[LLM REQUEST]` - Prompts sent to model
- `[LLM RESPONSE]` - Model outputs (cleaned)
- `[RAW LLM]` - Raw model output before cleaning
- `[MEMORY CONTEXT]` - Memary agent context
- `[RESPONSE CLEANING]` - Multi-turn cleaning actions
- `[RECOVERY]` - Broken response detection and recovery
- `[MEMORY SAVE/SKIP]` - Memory saving decisions
- `[MEMARY FIX]` - Context manipulation fixes

## File Structure

### Core Files
- `main.py` - FastAPI server, LLM setup, API endpoints, broken response detection
- `pyproject.toml` - Dependencies (llama-index, llama-cpp-python GPU build)
- `Makefile` - Server control, memory clearing, testing commands
- `test.sh` - Comprehensive testing script (chat, recovery, emotion, streaming)
- `manage_memory.py` - Memory management script (clear, backup, restore, status)

### Data Directory (`data/`)
- `past_chat.json` - Conversation history
- `memory_stream.json` - Processed memories  
- `entity_knowledge_store.json` - Entity relationships
- `system_persona.txt` - System prompt
- `user_persona.txt` - Dynamic user context

### Memary Agent (`src/memary/`)
- `agent/base_agent.py` - Core memory logic
- `agent/chat_agent.py` - Chat-specific agent

## Common Issues & Solutions

### Multi-Turn Hallucinations
**Symptom**: LLM generates "User: ... Assistant: ..." exchanges or very short responses like "user"
**Causes**: 
1. User persona injected as "user" message
2. Context format confuses model about speaker
3. Insufficient stop sequences
4. **NEW**: Corrupted conversation context from broken responses
5. **NEW**: Model generating broken multi-turn patterns internally
6. **NEW**: Too many history messages in context (>30 messages causes degradation)

**Solutions**:
1. Move user persona to system context
2. Enhanced stop sequences in model config
3. Aggressive streaming detection and cleaning
4. **NEW**: Minimum length checks before pattern detection
5. **NEW**: Skip saving broken responses to memory
6. **NEW**: Recovery commands (`make restart`, `./manage_memory.py clear-recent`)
7. **NEW**: Broken response detection with graceful recovery messages
8. **NEW**: Raw LLM logging for troubleshooting broken outputs
9. **NEW**: Configurable history limit (MAX_HISTORY_MESSAGES = 30)
10. Clear system prompt instructions

### Broken Response Recovery
**Symptom**: LLM outputs start with role indicators or are very short/nonsensical
**Detection**: `is_broken_response()` function catches various broken patterns
**Recovery**: System automatically responds with helpful messages like:
- "I'm sorry, I lost my thought there. Could you repeat your question?"
- "I apologize for the confusion. Could you please rephrase that?"
**Prevention**: Broken responses are not saved to memory to prevent contamination

#### Real-World Example (Working Correctly)
```
🛑 STOPPING STREAM: Multi-turn pattern detected
INFO:__main__:[RESPONSE CLEANING] Original: '\nuser: Hi\nassistant'
INFO:__main__:[RESPONSE CLEANING] Cleaned: 'user: Hi'
🔧 BROKEN RESPONSE DETECTED - Generating recovery response
INFO:__main__:[RECOVERY] Generated recovery response: I apologize for the confusion. Could you please rephrase that?
```
**What happened**: LLM generated broken multi-turn output → System detected it → Stopped stream → Generated recovery message → Saved recovery to memory instead of broken response. **This is the correct behavior!**

### Memory Context Issues
**Symptom**: Responses don't reflect conversation history
**Debug**: Check `[MEMORY CONTEXT]` logs for proper context construction

### GPU/Performance Issues
**Check**: Model loading logs should show all layers on GPU
**Config**: Ensure `n_gpu_layers: -1` for full offload

## Development Workflow

### Interpreting System Logs
**When you see "🔧 BROKEN RESPONSE DETECTED"** - This is GOOD! It means:
1. The system caught a broken LLM response before it reached the user
2. A graceful recovery message was generated instead
3. Memory was protected from contamination
4. The user gets a helpful "please rephrase" message

**Key Log Indicators**:
- `🛑 STOPPING STREAM` - Stream detection working
- `[RESPONSE CLEANING]` - Multi-turn cleaning active
- `🔧 BROKEN RESPONSE DETECTED` - Recovery system engaged
- `[RECOVERY] Generated recovery response` - Graceful fallback provided
- `[MEMORY SAVE]` - Recovery message saved instead of broken response

### Testing Multi-Turn Prevention
```bash
# Test with bypass mode (no memory)
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello"}], "bypass_memory": true, "stream": true}'

# Test with memory context
./test.sh chat "Hello"

# Test emotion detection  
./test.sh emotion

# Test system recovery
./test.sh recovery

# Clear memory and restart
make clear-memory && make restart
```

### Memory Management
```bash
# Clear all memory
./manage_memory.py clear-all

# Clear just recent problematic entries (recovery)
./manage_memory.py clear-recent

# Check memory status and file sizes
./manage_memory.py status

# Create backup before major changes
./manage_memory.py backup

# Restore from backup
./manage_memory.py restore backup_name

# Emergency restart when system is broken
make restart

# Test if system needs recovery
./test.sh recovery

# Regenerate persona from history
curl -X POST http://localhost:8000/regenerate-persona
```

### Testing Commands

Use the dedicated test script for all testing functionality:

```bash
# Basic testing
./test.sh help                    # Show all test commands
./test.sh status                  # Check server status  
./test.sh chat "Your question"    # Test chat functionality
./test.sh bypass "Question"       # Test without memory context
./test.sh recovery                # Test if system needs recovery

# Specific functionality tests
./test.sh emotion                 # Test emotion detection
./test.sh health                  # Health endpoint
./test.sh models                  # Models endpoint
./test.sh stream "Tell a story"   # Test streaming responses
./test.sh content "Question"      # Get just response content
```

## Key Principles

1. **Single Response Only**: Never allow multi-turn generation
2. **Memory Clarity**: User persona goes in system context, not as user message
3. **Aggressive Prevention**: Multiple layers of multi-turn detection
4. **Graceful Recovery**: Automatic detection and recovery from broken responses
5. **Full Logging**: Log complete prompts, raw outputs, and cleaned responses for debugging
6. **Mistral Optimization**: Use model-specific best practices
7. **GPU Efficiency**: Full offload for maximum performance
8. **Emotion Awareness**: Extract and return emotion markers from responses
9. **Recovery Mechanisms**: Built-in tools for handling broken conversation states
10. **Memory Protection**: Never save broken responses that could contaminate context

## Future Improvements

1. **Context Length Management**: Implement sliding window for long conversations
2. **Persona Refinement**: Periodic persona updates based on interaction patterns
3. **Response Quality**: Fine-tune sampling parameters based on response quality metrics
4. **Memory Optimization**: Compress old memories while preserving key information
5. **Multi-Model Support**: Abstract LLM interface for different model backends
6. **Enhanced Recovery**: More sophisticated broken response pattern detection
7. **Response Quality Metrics**: Implement scoring system for response quality
8. **Adaptive Prompting**: Dynamic system prompt adjustment based on response quality
