import json
import time
import logging
import re
from pathlib import Path
from typing import List
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
from llama_index.core import Settings
from llama_index.llms.llama_cpp import LlamaCPP
from memary.agent.chat_agent import ChatAgent
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel

# Configure Rich console
console = Console()

# Configuration Constants
MAX_HISTORY_MESSAGES = 30  # Maximum chat history messages for LLM context (research shows degradation after ~30)

# Configure logging with Rich
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, show_path=False)]
)
logger = logging.getLogger(__name__)

def log_llm_request(messages: List, context: str = ""):
    """Log LLM request with rich formatting"""
    console.print(Panel.fit(
        f"[bold yellow]🔄 LLM REQUEST ({context})[/bold yellow]\n" +
        f"[dim]Messages count: {len(messages)}[/dim]",
        border_style="yellow"
    ))
    
    for i, msg in enumerate(messages):
        if hasattr(msg, 'role') and hasattr(msg, 'content'):
            role = msg.role
            content = msg.content
        elif isinstance(msg, dict):
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
        else:
            role = 'unknown'
            content = str(msg)
            
        # Color code by role
        if role == 'system':
            color = "blue"
            icon = "⚙️"
        elif role == 'user':
            color = "green"
            icon = "👤"
        elif role == 'assistant':
            color = "magenta"
            icon = "🤖"
        else:
            color = "white"
            icon = "❓"
            
        # Show full content without truncation
        content_display = content.replace('\n', '\\n')
        
        console.print(f"  {icon} [{color}]{role}[/{color}]: {content_display}")

def log_llm_response(response_text: str, context: str = ""):
    """Log LLM response with rich formatting"""
    console.print(Panel.fit(
        f"[bold cyan]✅ LLM RESPONSE ({context})[/bold cyan]\n" +
        f"[dim]Length: {len(response_text)} chars[/dim]",
        border_style="cyan"
    ))
    
    # Show full response without truncation
    response_display = response_text.replace('\n', '\\n')
    console.print(f"  🗨️  [cyan]{response_display}[/cyan]")
    
    # Check for multi-turn patterns
    multi_turn_patterns = [
        r'\nuser:', r'\nassistant:', r'\nUser:', r'\nAssistant:',
        r'\nhuman:', r'\nHuman:', r'\nai:', r'\nAI:',
    ]
    
    # Check case-insensitive patterns first
    for pattern in multi_turn_patterns:
        if re.search(pattern, response_text, re.IGNORECASE):
            console.print(f"  ⚠️  [bold red]MULTI-TURN DETECTED: Found pattern '{pattern.strip()}'[/bold red]")
            # Show where it starts
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                start_pos = match.start()
                context_snippet = response_text[max(0, start_pos-50):start_pos+100]
                console.print(f"      [red]Context: ...{context_snippet}...[/red]")
            break
    
    # Check case-sensitive end patterns (to avoid false positives with "AI" etc.)
    end_role_patterns = [
        r'\s+user\s*$',           # "user" at the end
        r'\s+assistant\s*$',      # "assistant" at the end
        r'\s+human\s*$',          # "human" at the end
        r'\s+ai\s*$',             # "ai" (lowercase only) at the end
    ]
    
    for pattern in end_role_patterns:
        if re.search(pattern, response_text):  # Case-sensitive
            console.print(f"  ⚠️  [bold red]END ROLE DETECTED: Found pattern '{pattern.strip()}'[/bold red]")
            match = re.search(pattern, response_text)
            if match:
                start_pos = match.start()
                context_snippet = response_text[max(0, start_pos-50):start_pos+100]
                console.print(f"      [red]Context: ...{context_snippet}...[/red]")
            break

def log_raw_llm_response(response_text: str, context: str = ""):
    """Log raw LLM response without any processing for debugging"""
    console.print(Panel.fit(
        f"[bold yellow]🔍 RAW LLM RESPONSE ({context})[/bold yellow]\n" +
        f"[dim]Length: {len(response_text)} chars[/dim]\n" +
        f"[dim]Repr: {repr(response_text[:100])}{'...' if len(response_text) > 100 else ''}[/dim]",
        border_style="yellow"
    ))
    
    # Show raw response with all special characters visible
    console.print(f"  🔍 [yellow]RAW: {repr(response_text)}[/yellow]")
    
    # Also show human-readable version but mark it clearly
    response_display = response_text.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
    console.print(f"  📝 [yellow]DISPLAY: {response_display}[/yellow]")

def log_memory_context(messages: List, context: str = ""):
    """Log memory context with rich formatting"""
    console.print(Panel.fit(
        f"[bold magenta]🧠 MEMORY CONTEXT ({context})[/bold magenta]\n" +
        f"[dim]Messages count: {len(messages)}[/dim]",
        border_style="magenta"
    ))
    
    for i, msg in enumerate(messages):
        if isinstance(msg, dict):
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
        else:
            role = 'unknown'
            content = str(msg)
            
        # Color code by role
        if role == 'system':
            color = "blue"
            icon = "⚙️"
        elif role == 'user':
            color = "green"
            icon = "👤"
        elif role == 'assistant':
            color = "magenta"
            icon = "🤖"
        else:
            color = "white"
            icon = "❓"
            
        # Show full content without truncation
        content_display = content.replace('\n', '\\n')
        
        console.print(f"  [{i}] {icon} [{color}]{role}[/{color}]: {content_display}")

def is_broken_response(response_text: str) -> bool:
    """
    Detect if the LLM response is a broken multi-turn hallucination.
    Returns True if the response appears to be broken and needs recovery.
    """
    response = response_text.strip()
    
    # Check if response is too short (less than 5 characters)
    if len(response) < 5:
        return True
    
    # Check if response is just a role word
    if response.lower() in ['user', 'assistant', 'human', 'ai', 'system']:
        return True
    
    # Check for incomplete responses that end abruptly
    if response.endswith((' I ', ' I', ' and', ' but', ' the', ' to', ' for', ' with', ' in', ' on', ' at')):
        return True
        
    # Check if response starts with role patterns (multi-turn hallucination)
    broken_patterns = [
        r'^user:',
        r'^assistant:',
        r'^human:',
        r'^ai:',
        r'^system:',
        r'^\w+:\s*you',  # "user: you're just", etc.
        r'^\w+:\s*\w+.*$',  # Generic "role: content" at start
        r'^(user|assistant|human|ai|system)\s*$',  # Just role names
        r'^\w+:\s*$',  # Role with colon but no content
        r'^[A-Za-z]+:\s*[A-Za-z]+\s*$',  # "role: single_word"
    ]
    
    for pattern in broken_patterns:
        if re.match(pattern, response, re.IGNORECASE):
            return True
    
    return False

def generate_recovery_response() -> str:
    """Generate a recovery response when the LLM produces broken output."""
    recovery_responses = [
        "I'm sorry, I lost my thought there. Could you repeat your question?",
        "I apologize for the confusion. Could you please rephrase that?",
        "Sorry, I seemed to have gotten distracted. What were you asking?",
        "I'm sorry, something went wrong with my response. Could you try asking again?",
        "Apologies, I lost track of what I was saying. Could you repeat that?",
    ]
    
    import random
    return random.choice(recovery_responses)

def extract_emotions(response_text: str) -> dict:
    """Extract emotion markers from response text and return emotion data."""
    emotion_pattern = r'\[(?:neutral|anger|sadness|joy|fear|surprise|disgust|happy|sad|angry|excited|calm|worried|confused)\]'
    emotions_found = re.findall(emotion_pattern, response_text, re.IGNORECASE)
    
    emotion_data = {
        "emotions_detected": [],
        "primary_emotion": None
    }
    
    if emotions_found:
        # Clean up emotion markers (remove brackets and convert to lowercase)
        clean_emotions = [emotion.strip('[]').lower() for emotion in emotions_found]
        emotion_data["emotions_detected"] = clean_emotions
        
        # Set primary emotion (first detected emotion)
        if clean_emotions:
            emotion_data["primary_emotion"] = clean_emotions[0]
            
        logger.info(f"[EMOTION EXTRACTION] Detected: {clean_emotions}")
    
    return emotion_data

def clean_llm_response(response_text: str) -> str:
    """
    Clean LLM response to prevent multi-turn conversation hallucinations.
    Removes simulated user/assistant exchanges that the LLM sometimes generates.
    Enhanced with Mistral-specific patterns and more comprehensive cleaning.
    Also handles emotion markers from the client.
    """
    cleaned = response_text.strip()
    
    # First, extract and log emotion markers if present
    emotion_pattern = r'\[(?:neutral|anger|sadness|joy|fear|surprise|disgust|happy|sad|angry|excited|calm|worried|confused)\]'
    emotions_found = re.findall(emotion_pattern, cleaned, re.IGNORECASE)
    
    if emotions_found:
        logger.info(f"[EMOTION DETECTION] Found emotion markers: {emotions_found}")
        # You can choose to either:
        # 1. Remove emotion markers (clean output)
        # 2. Keep them (preserve emotion context)
        # 3. Process them for emotion tracking
        
        # For now, let's remove them from the final output but log them
        cleaned = re.sub(emotion_pattern, '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\s+', ' ', cleaned)  # Clean up extra spaces
        cleaned = cleaned.strip()
    
    # Pattern to detect simulated multi-turn conversations
    # Enhanced patterns based on Mistral best practices
    # Made more specific to avoid false positives with legitimate content
    patterns = [
        r'\nuser:\s*.*',  # Remove everything after "user:"
        r'\nassistant:\s*.*',  # Remove everything after "assistant:"
        r'\nUser:\s*.*',  # Capitalized versions
        r'\nAssistant:\s*.*',
        r'\nhuman:\s*.*',  # Alternative formats
        r'\nHuman:\s*.*',
        r'\nai:\s*.*',
        r'\nAI:\s*.*',
        r'\[INST\].*',  # Mistral instruction markers
        r'\[/INST\].*',
        r'\nUser said:\s*.*',  # Common conversation patterns
        r'\nAssistant:\s*.*',
        r'\nResponse:\s*.*',
        # More specific role patterns to avoid false positives
        r'\n(?:user|assistant|human|ai|system):\s+.*',  # Only match actual role words with colons
    ]
    
    for pattern in patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove specific role words that appear at the end without colons
    # This catches cases like "...response content here user" or "...response assistant"
    # Be very specific to avoid false positives with legitimate content
    end_role_patterns = [
        r'\s+user\s*$',           # "user" at the end
        r'\s+assistant\s*$',      # "assistant" at the end  
        r'\s+human\s*$',          # "human" at the end
        r'\s+ai\s*$',             # "ai" (lowercase only) at the end
        r'\s+response\s*$',       # "response" at the end
    ]
    
    # Apply these patterns case-sensitively to avoid matching "AI" in legitimate content
    for pattern in end_role_patterns:
        cleaned = re.sub(pattern, '', cleaned)  # No IGNORECASE flag
    
    # Also remove any trailing conversation-like patterns at the end
    # Sometimes the LLM will end with incomplete multi-turn format
    # Made more specific to avoid false positives
    end_patterns = [
        r'\n(?:user|assistant|human|ai|system):\s*$',  # Only match actual role words ending with colons
        r'\n(?:user|assistant|human|ai|system):\s*\w+.*$',  # Lines with actual role words: content at the end
    ]
    
    for pattern in end_patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
    
    # Clean up extra whitespace
    cleaned = cleaned.strip()
    
    # Log if we made changes
    if cleaned != response_text.strip():
        logger.info("[RESPONSE CLEANING] Removed simulated multi-turn content")
        logger.info(f"[RESPONSE CLEANING] Original length: {len(response_text)}, Cleaned length: {len(cleaned)}")
        logger.info(f"[RESPONSE CLEANING] Original: {repr(response_text)}")
        logger.info(f"[RESPONSE CLEANING] Cleaned: {repr(cleaned)}")
    
    return cleaned

# Setup dirs and files if missing
data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

files = {
    "memory_stream.json": [],
    "entity_knowledge_store.json": {},
    "past_chat.json": [],
    "system_persona.txt": "You are a helpful AI companion that learns from interactions.",
    "user_persona.txt": "The user is someone who enjoys building AI tools and chatting casually."
}

for filename, content in files.items():
    path = data_dir / filename
    if not path.exists():
        if isinstance(content, (list, dict)):
            with open(path, "w") as f:
                json.dump(content, f)
        else:
            with open(path, "w") as f:
                f.write(content)

# Load your GGUF model (swap path)
model_path = "/home/nick/.cache/ask_llm/models/bartowski/cognitivecomputations_Dolphin-Mistral-24B-Venice-Edition-GGUF/cognitivecomputations_Dolphin-Mistral-24B-Venice-Edition-Q4_K_M.gguf"  # e.g., llama-3-8b.gguf from HF

print(f"Loading model from: {model_path}")
print("Initializing LlamaCPP with GPU support...")

llm = LlamaCPP(
    model_path=model_path,
    temperature=0.1,  # Much lower temperature for more focused responses
    model_kwargs={
        "n_gpu_layers": -1,  # GPU offload
        "top_k": 40,  # Limit vocabulary consideration
        "top_p": 0.9,  # Nucleus sampling
        "repeat_penalty": 1.1,  # Penalize repetition
        "max_tokens": 500,  # Increased token limit for better responses (was 200)
        "stop": ["[/INST]", "\nUser:", "\nuser:", "\nAssistant:", "\nassistant:", "\nHuman:", "\nhuman:", "\n\nUser:", "\n\nuser:", "\n\nAssistant:", "\n\nassistant:", "user:", "assistant:", "User:", "Assistant:"],  # Comprehensive stop sequences
        "frequency_penalty": 0.1,  # Penalize frequent tokens (if supported)
        "presence_penalty": 0.1,   # Penalize repeated topics (if supported)
        "chat_format": "mistral-instruct",  # Use Mistral's chat template format
    },
    verbose=True
)

print(f"LlamaCPP model loaded. GPU layers: {llm.model_kwargs.get('n_gpu_layers', 'unknown')}")
logger.info(f"Model loaded with GPU acceleration: {llm.model_kwargs.get('n_gpu_layers', 'unknown')} layers")
Settings.llm = llm

def generate_dynamic_persona():
    """Generate user persona dynamically based on conversation history."""
    logger.info("Generating dynamic user persona from conversation history...")
    
    # Read conversation history
    past_chat_file = data_dir / "past_chat.json"
    try:
        with open(past_chat_file, "r") as f:
            chat_history = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        chat_history = []
    
    # Read memory stream for additional context
    memory_stream_file = data_dir / "memory_stream.json"
    try:
        with open(memory_stream_file, "r") as f:
            memory_stream = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        memory_stream = []
    
    # If no conversation history exists, use default persona
    if not chat_history and not memory_stream:
        logger.info("No conversation history found, using default persona")
        return
    
    # Prepare context for persona generation
    history_text = ""
    if chat_history:
        # Get recent conversations (last 20 messages to avoid token limits)
        recent_chat = chat_history[-20:] if len(chat_history) > 20 else chat_history
        history_text += "Recent conversations:\n"
        for msg in recent_chat:
            role = msg.get("role", "unknown")
            content = msg.get("content", "").strip()
            if content:
                history_text += f"{role}: {content}\n"
    
    if memory_stream:
        # Get recent memories (last 10 entries)
        recent_memories = memory_stream[-10:] if len(memory_stream) > 10 else memory_stream
        if recent_memories:
            history_text += "\nRecent memories:\n"
            for memory in recent_memories:
                if isinstance(memory, dict) and "content" in memory:
                    history_text += f"- {memory['content']}\n"
                elif isinstance(memory, str):
                    history_text += f"- {memory}\n"
    
    if not history_text.strip():
        logger.info("No meaningful conversation history found, keeping default persona")
        return
    
    # Create prompt for persona generation
    persona_prompt = f"""Based on the following conversation history and memories, write a concise 1-2 sentence persona description for the user. Focus on their interests, communication style, and key characteristics that would be helpful for an AI assistant to know.

{history_text}

User persona (keep it brief and focused):"""
    
    try:
        logger.info("Requesting LLM to generate user persona...")
        logger.info(f"[PERSONA GENERATION] prompt: {persona_prompt[:300]}{'...' if len(persona_prompt) > 300 else ''}")
        
        response = llm.complete(persona_prompt, max_tokens=150, temperature=0.3)
        generated_persona = str(response.text).strip()
        
        # Clean up the response (remove any prefix like "User persona:" if present)
        if ":" in generated_persona and len(generated_persona.split(":")) > 1:
            generated_persona = generated_persona.split(":", 1)[1].strip()
        
        logger.info(f"[PERSONA GENERATION] generated: {generated_persona}")
        
        # Save the generated persona
        user_persona_file = data_dir / "user_persona.txt"
        with open(user_persona_file, "w") as f:
            f.write(generated_persona)
        
        logger.info(f"Dynamic user persona generated and saved: {generated_persona}")
        
    except Exception as e:
        logger.error(f"Failed to generate dynamic persona: {e}")
        logger.info("Using existing user persona file")

# Generate dynamic persona before initializing agent
generate_dynamic_persona()

# Init Memary agent
logger.info("Initializing Memary agent...")
chat_agent = ChatAgent(
    agent_name="CompanionBot",
    memory_stream_json=str(data_dir / "memory_stream.json"),
    entity_knowledge_store_json=str(data_dir / "entity_knowledge_store.json"),
    system_persona_txt=str(data_dir / "system_persona.txt"),
    user_persona_txt=str(data_dir / "user_persona.txt"),
    past_chat_json=str(data_dir / "past_chat.json")
)

# FIX: Remove the problematic user persona injection from contexts
# The Memary agent incorrectly injects user persona as a "user" message which confuses the LLM
# Remove the second context which is the user persona if it exists
if (len(chat_agent.message.contexts) >= 2 and 
    chat_agent.message.contexts[1].role == "user" and 
    chat_agent.message.contexts[1].content == chat_agent.message.user_persona):
    
    logger.info("[MEMARY FIX] Removing problematic user persona injection from contexts")
    logger.info(f"[MEMARY FIX] Removed user persona: {chat_agent.message.contexts[1].content[:100]}...")
    chat_agent.message.contexts.pop(1)  # Remove the user persona "user" message
    
    # Also remove from llm_message
    if (len(chat_agent.message.llm_message["messages"]) >= 2 and 
        chat_agent.message.llm_message["messages"][1].role == "user" and 
        chat_agent.message.llm_message["messages"][1].content == chat_agent.message.user_persona):
        chat_agent.message.llm_message["messages"].pop(1)
        
    # Instead, append user persona to system persona to make it part of system context
    if chat_agent.message.user_persona.strip():
        original_system = chat_agent.message.system_persona
        enhanced_system = f"{original_system}\n\nUser Context: {chat_agent.message.user_persona}"
        chat_agent.message.system_persona = enhanced_system
        
        # Update the system context
        if chat_agent.message.contexts and chat_agent.message.contexts[0].role == "system":
            chat_agent.message.contexts[0].content = enhanced_system
        if chat_agent.message.llm_message["messages"] and chat_agent.message.llm_message["messages"][0].role == "system":
            chat_agent.message.llm_message["messages"][0].content = enhanced_system
            
        logger.info("[MEMARY FIX] Moved user persona to system context instead")

logger.info("Memary agent initialized successfully")

app = FastAPI()

@app.get("/health")
def health():
    return "OK"

@app.get("/metrics")
def metrics():
    # Dummy Prometheus-style metrics; expand as needed
    return Response(content="llm_requests_total 0\nllm_tokens_total 0\n", media_type="text/plain")

@app.post("/props")
async def props(request: Request):
    # Dummy: accepts props update, but does nothing
    props_data = await request.json()
    logger.info(f"[PROPS] Received properties update: {list(props_data.keys()) if props_data else 'empty'}")
    return {"message": "Properties updated (dummy)"}

@app.post("/tokenize")
async def tokenize(request: Request):
    data = await request.json()
    content = data.get("content", "")
    tokens = llm._model.tokenize(content.encode("utf-8"))
    return {"tokens": tokens}

@app.post("/detokenize")
async def detokenize(request: Request):
    data = await request.json()
    tokens = data.get("tokens", [])
    content = llm._model.detokenize(tokens).decode("utf-8")
    return {"content": content}

@app.get("/v1/models")
def models():
    return {
        "object": "list",
        "data": [{"id": "local-llama", "object": "model", "created": int(time.time()), "owned_by": "local"}]
    }

@app.post("/v1/completions")
async def completions(request: Request):
    data = await request.json()
    prompt = data.get("prompt", "")
    max_tokens = data.get("max_tokens", 128)
    temperature = data.get("temperature", 0.7)
    
    logger.info(f"[COMPLETION REQUEST] prompt: {prompt[:200]}{'...' if len(prompt) > 200 else ''}")
    logger.info(f"[COMPLETION REQUEST] max_tokens: {max_tokens}, temperature: {temperature}")
    
    response = llm.complete(prompt, max_tokens=max_tokens, temperature=temperature)
    response_text = str(response.text)
    
    logger.info(f"[COMPLETION RESPONSE] output: {response_text[:200]}{'...' if len(response_text) > 200 else ''}")
    
    return {
        "id": "cmpl-local",
        "object": "text_completion",
        "created": int(time.time()),
        "model": "local-llama",
        "choices": [{"text": response_text, "index": 0, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": len(prompt.split()), "completion_tokens": len(response_text.split()), "total_tokens": 0}
    }

@app.post("/v1/chat/completions")
async def chat_completion(request: Request):
    data = await request.json()
    messages = data.get("messages", [])
    stream = data.get("stream", False)
    
    # Add bypass mode for testing without memory
    bypass_memory = data.get("bypass_memory", False)
    
    if not messages:
        return {"error": "No messages provided"}
    
    # Extract system messages from API call
    api_system_messages = [msg["content"] for msg in messages if msg["role"] == "system"]
    
    # Take last user message (agent handles context/memory)
    user_message = next((msg["content"] for msg in reversed(messages) if msg["role"] == "user"), "")
    if not user_message:
        return {"error": "No user message found"}
    
    logger.info(f"[CHAT REQUEST] user_message: {user_message[:200]}{'...' if len(user_message) > 200 else ''}")
    logger.info(f"[CHAT REQUEST] api_system_messages: {api_system_messages}")
    logger.info(f"[CHAT REQUEST] stream: {stream}, total_messages: {len(messages)}, bypass_memory: {bypass_memory}")
    
    # Temporarily override system persona if API provides system message
    original_system_persona = None
    original_system_context = None
    if api_system_messages:
        # Combine API system messages and add comprehensive anti-multi-turn instruction
        combined_system = " ".join(api_system_messages)
        combined_system += "\n\nCRITICAL: Only provide a single, complete response. NEVER generate conversation transcripts, user messages, or continue with multi-turn exchanges. Stop immediately after your response.\n\nEMOTION HANDLING: If you detect emotional context from the user, you may include emotion markers like [neutral], [happy], [sad], [angry], etc. in your response to indicate the appropriate emotional tone. Supported emotions: neutral, anger, sadness, joy, fear, surprise, disgust, happy, sad, angry, excited, calm, worried, confused."
        
        original_system_persona = chat_agent.message.system_persona
        chat_agent.message.system_persona = combined_system
        
        # Update the first context which is the system message
        if chat_agent.message.contexts and chat_agent.message.contexts[0].role == "system":
            original_system_context = chat_agent.message.contexts[0].content
            chat_agent.message.contexts[0].content = combined_system
        
        # Also update the llm_message system context
        if chat_agent.message.llm_message["messages"] and chat_agent.message.llm_message["messages"][0].role == "system":
            chat_agent.message.llm_message["messages"][0].content = combined_system
            
        logger.info("[CHAT SYSTEM OVERRIDE] Temporarily overriding system persona with API message")
    else:
        # Add comprehensive anti-multi-turn instruction to default system persona
        current_system = chat_agent.message.system_persona
        if "NEVER generate conversation transcripts" not in current_system:
            enhanced_system = current_system + "\n\nCRITICAL: Only provide a single, complete response. NEVER generate conversation transcripts, user messages, or continue with multi-turn exchanges. Stop immediately after your response.\n\nEMOTION HANDLING: If you detect emotional context from the user, you may include emotion markers like [neutral], [happy], [sad], [angry], etc. in your response to indicate the appropriate emotional tone. Supported emotions: neutral, anger, sadness, joy, fear, surprise, disgust, happy, sad, angry, excited, calm, worried, confused."
            
            original_system_persona = chat_agent.message.system_persona
            chat_agent.message.system_persona = enhanced_system
            
            # Update contexts
            if chat_agent.message.contexts and chat_agent.message.contexts[0].role == "system":
                original_system_context = chat_agent.message.contexts[0].content
                chat_agent.message.contexts[0].content = enhanced_system
            
            if chat_agent.message.llm_message["messages"] and chat_agent.message.llm_message["messages"][0].role == "system":
                chat_agent.message.llm_message["messages"][0].content = enhanced_system
                
            logger.info("[CHAT SYSTEM ENHANCEMENT] Added anti-multi-turn instruction to system prompt")
    
    try:
        if bypass_memory:
            # Bypass memory mode - use direct LLM without Memary agent
            logger.info("[CHAT BYPASS] Processing without Memary memory context...")
            
            from llama_index.core.llms import ChatMessage, MessageRole
            
            # Build simple context without memory
            simple_messages = []
            
            # Add system message
            if api_system_messages:
                combined_system = " ".join(api_system_messages)
                simple_messages.append(ChatMessage(role=MessageRole.SYSTEM, content=combined_system))
            
            # Add just the user message
            simple_messages.append(ChatMessage(role=MessageRole.USER, content=user_message))
            
            # Log what we're sending to the LLM (bypass mode)
            log_llm_request(simple_messages, "BYPASS MODE")
            
            if stream:
                def generate_bypass_stream():
                    logger.info("[CHAT BYPASS STREAMING] Starting stream response...")
                    response_stream = Settings.llm.stream_chat(simple_messages)
                    full_response = ""
                    
                    # First collect the complete response without sending chunks
                    for token in response_stream:
                        chunk_content = str(token.delta)
                        full_response += chunk_content
                    
                    logger.info(f"[CHAT BYPASS STREAMING RESPONSE] full_output: {full_response[:200]}{'...' if len(full_response) > 200 else ''}")
                    
                    # Log the complete response from LLM
                    # log_raw_llm_response(full_response, "BYPASS STREAMING")
                    log_llm_response(full_response, "BYPASS STREAMING")
                    
                    # Check if response is broken and decide what to send
                    if is_broken_response(full_response):
                        console.print("[bold red]🔧 BROKEN BYPASS RESPONSE - Generating recovery[/bold red]")
                        recovery_response = generate_recovery_response()
                        logger.info(f"[RECOVERY] Bypass generated recovery: {recovery_response}")
                        
                        # Send ONLY the recovery response as chunks
                        words = recovery_response.split()
                        for i, word in enumerate(words):
                            word_chunk = word + (" " if i < len(words) - 1 else "")
                            chunk = {
                                "id": "chatcmpl-bypass",
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": "local-llama",
                                "choices": [{
                                    "index": 0,
                                    "delta": {"content": word_chunk},
                                    "finish_reason": None
                                }]
                            }
                            yield f"data: {json.dumps(chunk)}\n\n"
                        emotion_data = {"emotions_detected": [], "primary_emotion": None}
                    else:
                        # Send the good response as chunks
                        words = full_response.split()
                        for i, word in enumerate(words):
                            word_chunk = word + (" " if i < len(words) - 1 else "")
                            chunk = {
                                "id": "chatcmpl-bypass",
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": "local-llama",
                                "choices": [{
                                    "index": 0,
                                    "delta": {"content": word_chunk},
                                    "finish_reason": None
                                }]
                            }
                            yield f"data: {json.dumps(chunk)}\n\n"
                        
                        # Extract emotion data and clean for display
                        emotion_data = extract_emotions(full_response)
                    
                    # Final chunk
                    final_chunk = {
                        "id": "chatcmpl-bypass",
                        "object": "chat.completion.chunk", 
                        "created": int(time.time()),
                        "model": "local-llama",
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop"
                        }]
                    }
                    
                    # Add emotion data if any emotions were detected
                    if emotion_data["emotions_detected"]:
                        final_chunk["emotion_data"] = emotion_data
                    
                    yield f"data: {json.dumps(final_chunk)}\n\n"
                    yield "data: [DONE]\n\n"
                
                return StreamingResponse(
                    generate_bypass_stream(),
                    media_type="text/plain",
                    headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
                )
            else:
                response = Settings.llm.chat(simple_messages)
                response_text = str(response.message.content)
                
                # Log the LLM response
                log_raw_llm_response(response_text, "BYPASS NON-STREAMING")
                log_llm_response(response_text, "BYPASS NON-STREAMING")
                
                # Extract emotion data from bypass response
                emotion_data = extract_emotions(response_text)
                
                response_data = {
                    "id": "chatcmpl-bypass",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": "local-llama",
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": response_text},
                        "finish_reason": "stop"
                    }],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                }
                
                # Add emotion data if any emotions were detected
                if emotion_data["emotions_detected"]:
                    response_data["emotion_data"] = emotion_data
                    
                return response_data
        
        if stream:
            # Streaming with memory context - get memory-augmented context from Memary agent
            def generate_stream():
                from llama_index.core.llms import ChatMessage, MessageRole
                
                # Add user message to chat agent for memory processing 
                chat_agent.add_chat("user", user_message)
                
                # Do entity extraction and KG querying before streaming
                logger.info("[CHAT STREAMING] Processing routing agent for entity extraction...")
                try:
                    # Check if query relates to existing knowledge graph
                    cypher_query = chat_agent.check_KG(user_message)
                    
                    if cypher_query:
                        # Query exists in KG - get response with entity extraction
                        logger.info("[CHAT STREAMING] Using KG-based routing with entity extraction")
                        result = chat_agent.get_routing_agent_response(
                            user_message, return_entity=True
                        )
                        if isinstance(result, tuple):
                            rag_response, entities = result
                            # Ensure entities is a list of strings
                            if isinstance(entities, str):
                                entities = [entities]
                            elif not isinstance(entities, list):
                                entities = []
                            # Ensure all items in the list are strings
                            entities = [str(e) for e in entities] if entities else []
                        else:
                            rag_response = result
                            entities = []
                            
                        logger.info(f"[CHAT STREAMING] Extracted entities: {entities}")
                        
                        # Add routing agent response with entities (this populates memory!)
                        chat_agent.add_chat("system", "ReAct agent: " + str(rag_response), entities)
                    else:
                        # No existing knowledge - use external search
                        logger.info("[CHAT STREAMING] Using external search")
                        react_response = chat_agent.get_routing_agent_response(user_message)
                        
                        # Add routing agent response
                        chat_agent.add_chat("system", "ReAct agent: " + str(react_response))
                        
                        # Write response to temporary file for KG writeback
                        try:
                            with open("data/external_response.txt", "w") as f:
                                f.write(str(react_response))
                            chat_agent.write_back()
                            logger.info("[CHAT STREAMING] Completed KG writeback")
                        except Exception as e:
                            logger.warning(f"[CHAT STREAMING] KG writeback failed: {e}")
                
                except Exception as e:
                    logger.warning(f"[CHAT STREAMING] Entity extraction failed, using direct LLM: {e}")
                    # Continue with direct LLM if routing agent fails
                
                # Log the memory-augmented context
                llm_message_chat = chat_agent._change_llm_message_chat()
                
                # Force override system message in the final context if API provided one
                if api_system_messages:
                    combined_system = " ".join(api_system_messages)
                    for msg in llm_message_chat["messages"]:
                        if msg["role"] == "system":
                            msg["content"] = combined_system
                            break
                
                # Log memory context with rich formatting
                log_memory_context(llm_message_chat["messages"], "Memary Agent Context")
                
                # Convert to LlamaIndex format for streaming
                llm_messages = []
                for msg in llm_message_chat["messages"]:
                    role = MessageRole.USER if msg["role"] == "user" else (
                        MessageRole.SYSTEM if msg["role"] == "system" else MessageRole.ASSISTANT
                    )
                    llm_messages.append(ChatMessage(role=role, content=msg["content"]))
                
                # Log what we're sending to the LLM
                # log_llm_request(llm_messages, "STREAMING")
                
                # Stream the response with memory context
                logger.info("[CHAT STREAMING] Starting stream response...")
                response_stream = Settings.llm.stream_chat(llm_messages)
                full_response = ""
                should_stop_streaming = False
                
                # First, collect the complete response without sending chunks to client
                for token in response_stream:
                    if should_stop_streaming:
                        break
                        
                    chunk_content = str(token.delta)
                    full_response += chunk_content
                    
                    # Check for multi-turn patterns early and stop immediately
                    # But only if the response is long enough to be meaningful
                    if len(full_response.strip()) > 15:  # Minimum length check
                        multi_turn_patterns = [
                            r'\nuser:', r'\nassistant:', r'\nUser:', r'\nAssistant:',
                            r'\nhuman:', r'\nHuman:', r'\nai:', r'\nAI:',
                            r'\[INST\]', r'\[/INST\]',  # Mistral instruction markers
                            r'User said:', r'Assistant:', r'Response:',
                        ]
                        
                        # Check case-insensitive patterns
                        for pattern in multi_turn_patterns:
                            if re.search(pattern, full_response, re.IGNORECASE):
                                should_stop_streaming = True
                                full_response = clean_llm_response(full_response)
                                console.print("[bold red]🛑 STOPPING STREAM: Multi-turn pattern detected[/bold red]")
                                break
                        
                        # Check case-sensitive end patterns (to avoid false positives with "AI" etc.)
                        # Only apply if response is substantial (>30 chars) and ends with suspicious pattern
                        if not should_stop_streaming and len(full_response.strip()) > 30:
                            end_role_patterns = [
                                r'\s+user\s*$',           # "user" at the end
                                r'\s+assistant\s*$',      # "assistant" at the end
                                r'\s+human\s*$',          # "human" at the end
                                r'\s+ai\s*$',             # "ai" (lowercase only) at the end
                            ]
                            
                            for pattern in end_role_patterns:
                                if re.search(pattern, full_response):  # Case-sensitive
                                    should_stop_streaming = True
                                    full_response = clean_llm_response(full_response)
                                    console.print("[bold red]🛑 STOPPING STREAM: End role pattern detected[/bold red]")
                                    break
                
                # Log the complete response from LLM
                log_raw_llm_response(full_response, "STREAMING")
                log_llm_response(full_response, "STREAMING")
                
                # Now check if the complete response is broken and decide what to send
                if is_broken_response(full_response):
                    console.print("[bold red]🔧 BROKEN RESPONSE DETECTED - Generating recovery response[/bold red]")
                    recovery_response = generate_recovery_response()
                    logger.info(f"[RECOVERY] Generated recovery response: {recovery_response}")
                    
                    # Send ONLY the recovery response as chunks (don't send the broken response)
                    words = recovery_response.split()
                    for i, word in enumerate(words):
                        word_chunk = word + (" " if i < len(words) - 1 else "")
                        chunk = {
                            "id": "chatcmpl-memary",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": "local-llama",
                            "choices": [{
                                "index": 0,
                                "delta": {"content": word_chunk},
                                "finish_reason": None
                            }]
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                    
                    # Use recovery response for memory and emotion processing
                    final_response = recovery_response
                    emotion_data = {"emotions_detected": [], "primary_emotion": None}  # No emotions in recovery
                else:
                    # Send the good response as chunks
                    words = full_response.split()
                    for i, word in enumerate(words):
                        word_chunk = word + (" " if i < len(words) - 1 else "")
                        chunk = {
                            "id": "chatcmpl-memary",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": "local-llama",
                            "choices": [{
                                "index": 0,
                                "delta": {"content": word_chunk},
                                "finish_reason": None
                            }]
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                    
                    # Extract emotion data before cleaning
                    emotion_data = extract_emotions(full_response)
                    
                    # Clean the response before saving to memory
                    final_response = clean_llm_response(full_response)
                
                # Only save to memory if the response is meaningful and not broken
                if len(final_response.strip()) >= 10 and not is_broken_response(final_response):
                    # Save the final response to memory after streaming completes
                    chat_agent.add_chat("assistant", final_response)
                    chat_agent.message.save_contexts_to_json()
                    logger.info(f"[MEMORY SAVE] Saved response to memory: {final_response[:100]}{'...' if len(final_response) > 100 else ''}")
                else:
                    console.print("[bold yellow]⚠️ SKIPPING MEMORY SAVE: Response too short or appears to be broken[/bold yellow]")
                    logger.warning(f"[MEMORY SKIP] Not saving short/broken response: {repr(final_response)}")
                
                # Final chunk with finish_reason and emotion data
                final_chunk = {
                    "id": "chatcmpl-memary",
                    "object": "chat.completion.chunk", 
                    "created": int(time.time()),
                    "model": "local-llama",
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop"
                    }]
                }
                
                # Add emotion data to the final chunk if any emotions were detected
                if emotion_data["emotions_detected"]:
                    final_chunk["emotion_data"] = emotion_data
                
                yield f"data: {json.dumps(final_chunk)}\n\n"
                yield "data: [DONE]\n\n"
            
            return StreamingResponse(
                generate_stream(),
                media_type="text/plain",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
            )
        else:
            # Non-streaming response using proper Memary flow
            logger.info("[CHAT NON-STREAMING] Processing with proper Memary flow...")
            
            # Use the proper Memary chat flow with entity extraction and KG integration
            final_response = process_memary_chat(chat_agent, user_message)
            
            # Log the final response
            log_raw_llm_response(final_response, "NON-STREAMING")
            log_llm_response(final_response, "NON-STREAMING")
            
            # Check if the response is broken and needs recovery
            if is_broken_response(final_response):
                console.print("[bold red]🔧 BROKEN RESPONSE DETECTED - Generating recovery response[/bold red]")
                final_response = generate_recovery_response()
                logger.info(f"[RECOVERY] Generated recovery response: {final_response}")
                emotion_data = {"emotions_detected": [], "primary_emotion": None}  # No emotions in recovery
            else:
                # Extract emotion data from the response
                emotion_data = extract_emotions(final_response)
            
            response_data = {
                "id": "chatcmpl-memary",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "local-llama",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": final_response},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            }
            
            # Add emotion data if any emotions were detected
            if emotion_data["emotions_detected"]:
                response_data["emotion_data"] = emotion_data
            
            return response_data
    finally:
        # Restore original system persona if it was overridden
        if original_system_persona is not None:
            chat_agent.message.system_persona = original_system_persona
            if chat_agent.message.contexts and chat_agent.message.contexts[0].role == "system":
                if original_system_context is not None:
                    chat_agent.message.contexts[0].content = original_system_context
                else:
                    chat_agent.message.contexts[0].content = original_system_persona
            # Also restore the llm_message system context
            if chat_agent.message.llm_message["messages"] and chat_agent.message.llm_message["messages"][0].role == "system":
                if original_system_context is not None:
                    chat_agent.message.llm_message["messages"][0].content = original_system_context
                else:
                    chat_agent.message.llm_message["messages"][0].content = original_system_persona

@app.post("/v1/embeddings")
async def embeddings(request: Request):
    data = await request.json()
    input_text = data.get("input", "")
    if isinstance(input_text, str):
        input_text = [input_text]
    
    logger.info(f"[EMBEDDINGS REQUEST] input_count: {len(input_text)}")
    for i, text in enumerate(input_text):
        text_preview = text[:100] + "..." if len(text) > 100 else text
        logger.info(f"[EMBEDDINGS REQUEST] text_{i}: {text_preview}")
    
    embeds: List[List[float]] = llm._model.embeddings(input_text)
    
    logger.info(f"[EMBEDDINGS RESPONSE] embeddings_count: {len(embeds)}, embedding_dims: {len(embeds[0]) if embeds else 0}")
    
    return {
        "object": "list",
        "data": [{"object": "embedding", "embedding": emb, "index": i} for i, emb in enumerate(embeds)],
        "model": "local-llama",
        "usage": {"prompt_tokens": sum(len(t.split()) for t in input_text), "total_tokens": 0}
    }

@app.post("/regenerate-persona")
async def regenerate_persona():
    """Manually trigger persona regeneration based on current conversation history."""
    try:
        generate_dynamic_persona()
        
        # Reload the agent with new persona
        global chat_agent
        logger.info("Reloading Memary agent with updated persona...")
        chat_agent = ChatAgent(
            agent_name="CompanionBot",
            memory_stream_json=str(data_dir / "memory_stream.json"),
            entity_knowledge_store_json=str(data_dir / "entity_knowledge_store.json"),
            system_persona_txt=str(data_dir / "system_persona.txt"),
            user_persona_txt=str(data_dir / "user_persona.txt"),
            past_chat_json=str(data_dir / "past_chat.json")
        )
        
        # Read the new persona
        with open(data_dir / "user_persona.txt", "r") as f:
            new_persona = f.read().strip()
            
        return {
            "message": "Persona regenerated successfully",
            "new_persona": new_persona
        }
    except Exception as e:
        logger.error(f"Failed to regenerate persona: {e}")
        return {"error": f"Failed to regenerate persona: {str(e)}"}

def process_with_memary_agent(chat_agent, user_message: str) -> tuple[str, bool]:
    """
    Process user message through proper Memary routing agent flow.
    This extracts entities and populates memory_stream.json and entity_knowledge_store.json.
    
    Returns:
        tuple: (routing_response, used_kg) where used_kg indicates if KG was queried
    """
    try:
        console.print(Panel.fit(
            "[bold green]🧠 MEMARY ROUTING AGENT[/bold green]\n" +
            f"[dim]Processing: {user_message[:100]}{'...' if len(user_message) > 100 else ''}[/dim]",
            border_style="green"
        ))
        
        # Step 1: Check if query relates to existing knowledge graph
        cypher_query = chat_agent.check_KG(user_message)
        logger.info(f"[MEMARY KG CHECK] Cypher query: {cypher_query}")
        
        if cypher_query:
            # Step 2a: Query KG and extract entities (this populates memory!)
            console.print("[bold cyan]📊 Querying Knowledge Graph with entity extraction[/bold cyan]")
            rag_response, entities = chat_agent.get_routing_agent_response(
                user_message, return_entity=True
            )
            
            # Ensure entities is a list
            if entities is None:
                entities = []
            elif not isinstance(entities, list):
                entities = [str(entities)]
                
            logger.info(f"[MEMARY ENTITIES] Extracted {len(entities)} entities: {entities}")
            
            # Step 3a: Add ReAct response with entities to memory
            chat_agent.add_chat("system", f"ReAct agent: {rag_response}", entities)
            console.print(f"[bold green]✅ Added KG response with {len(entities)} entities to memory[/bold green]")
            
            return str(rag_response), True
            
        else:
            # Step 2b: External search via routing agent
            console.print("[bold yellow]🔍 Performing external search via routing agent[/bold yellow]")
            react_response = chat_agent.get_routing_agent_response(user_message)
            
            # Step 3b: Add external response to memory (no entities from external search)
            chat_agent.add_chat("system", f"ReAct agent: {react_response}")
            
            # Step 4b: Write external response back to KG for future queries
            try:
                with open("data/external_response.txt", "w") as f:
                    f.write(str(react_response))
                chat_agent.write_back()
                console.print("[bold blue]📝 Wrote external response back to KG[/bold blue]")
            except Exception as e:
                logger.warning(f"[MEMARY WRITEBACK] Failed to write back to KG: {e}")
            
            return str(react_response), False
            
    except Exception as e:
        logger.error(f"[MEMARY ERROR] Failed to process with routing agent: {e}")
        console.print(f"[bold red]❌ Memary routing failed: {e}[/bold red]")
        # Fallback: just add user message without routing agent
        chat_agent.add_chat("system", "Routing agent unavailable")
        return "I'm having trouble accessing my knowledge base. Let me try to help anyway.", False

def process_memary_chat(chat_agent, user_message: str) -> str:
    """
    Process chat using the proper Memary flow with entity extraction and KG integration.
    This follows the pattern from streamlit_app/app.py which properly populates memory.
    """
    logger.info("[MEMARY FLOW] Starting proper Memary chat processing...")
    
    # 1. Add user message to chat agent
    chat_agent.add_chat("user", user_message)
    logger.info(f"[MEMARY FLOW] Added user message: {user_message[:100]}{'...' if len(user_message) > 100 else ''}")
    
    # 2. Check if query relates to existing knowledge graph
    cypher_query = chat_agent.check_KG(user_message)
    logger.info(f"[MEMARY FLOW] KG check result: {cypher_query}")
    
    # 3. Get routing agent response with entity extraction
    if cypher_query:
        # Query exists in KG - get response with entity extraction
        logger.info("[MEMARY FLOW] Using KG-based routing with entity extraction")
        rag_response, entities = chat_agent.get_routing_agent_response(
            user_message, return_entity=True
        )
        logger.info(f"[MEMARY FLOW] Extracted entities: {entities}")
        logger.info(f"[MEMARY FLOW] RAG response: {rag_response[:100]}{'...' if len(rag_response) > 100 else ''}")
        
        # Add routing agent response with entities (this populates memory!)
        chat_agent.add_chat("system", "ReAct agent: " + rag_response, entities)
    else:
        # No existing knowledge - use external search, then write back to KG
        logger.info("[MEMARY FLOW] Using external search with KG writeback")
        react_response = chat_agent.get_routing_agent_response(user_message)
        logger.info(f"[MEMARY FLOW] External response: {react_response[:100]}{'...' if len(react_response) > 100 else ''}")
        
        # Add routing agent response (no entities from external search)
        chat_agent.add_chat("system", "ReAct agent: " + react_response)
        
        # Write response to temporary file for KG writeback
        try:
            with open("data/external_response.txt", "w") as f:
                f.write(react_response)
            logger.info("[MEMARY FLOW] Wrote external response to file for KG writeback")
            
            # Write back to knowledge graph for future queries
            chat_agent.write_back()
            logger.info("[MEMARY FLOW] Completed KG writeback")
        except Exception as e:
            logger.warning(f"[MEMARY FLOW] KG writeback failed: {e}")
    
    # 4. Generate final response using memory-enhanced context
    logger.info("[MEMARY FLOW] Generating final response with memory context")
    final_response = chat_agent.get_response()
    
    # Clean the response
    cleaned_response = clean_llm_response(str(final_response))
    logger.info(f"[MEMARY FLOW] Final response: {cleaned_response[:100]}{'...' if len(cleaned_response) > 100 else ''}")
    
    # 5. Save final response to memory (no entities needed, already captured above)
    if len(cleaned_response.strip()) >= 10 and not is_broken_response(cleaned_response):
        chat_agent.add_chat("assistant", cleaned_response)
        chat_agent.message.save_contexts_to_json()
        logger.info("[MEMARY FLOW] Saved final response to memory")
    else:
        logger.warning("[MEMARY FLOW] Skipping memory save - response too short or broken")
    
    logger.info("[MEMARY FLOW] Completed Memary chat processing")
    return cleaned_response

if __name__ == "__main__":
    import uvicorn
    logger.info("=" * 60)
    logger.info("Starting FastAPI Ollama-compatible server with Memary agent")
    logger.info(f"Model: {model_path}")
    logger.info(f"GPU layers: {llm.model_kwargs.get('n_gpu_layers', 'unknown')}")
    logger.info("Memary agent initialized with memory and context augmentation")
    logger.info("FIX APPLIED: Removed problematic user persona injection that caused multi-turn hallucinations")
    logger.info("Available endpoints:")
    logger.info("  - POST /v1/chat/completions (OpenAI-compatible, streaming & non-streaming)")
    logger.info("  - POST /v1/completions (OpenAI-compatible)")
    logger.info("  - POST /v1/embeddings (OpenAI-compatible)")
    logger.info("  - POST /regenerate-persona (regenerate user persona from history)")
    logger.info("  - GET /v1/models")
    logger.info("  - GET /health")
    logger.info("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")