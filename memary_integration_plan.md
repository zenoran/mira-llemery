# Memary Integration Plan

## Current Issues
The implementation is missing the core Memary functionality:
- No knowledge graph queries
- No entity extraction 
- No routing agent usage
- No memory population
- No KG writeback

## Required Changes

### 1. Fix Chat Agent Integration in main.py

**Current (Broken) Flow:**
```python
chat_agent.add_chat("user", user_message)  # No entities
response = chat_agent.get_response()       # Direct LLM
chat_agent.add_chat("assistant", response) # No entities
```

**Correct (Memary) Flow:**
```python
# 1. Add user message
chat_agent.add_chat("user", user_message)

# 2. Check if query is in knowledge graph
cypher_query = chat_agent.check_KG(user_message)

# 3. Get routing agent response with entity extraction
if cypher_query:
    rag_response, entities = chat_agent.get_routing_agent_response(
        user_message, return_entity=True
    )
    chat_agent.add_chat("system", "ReAct agent: " + rag_response, entities)
else:
    # External search, then writeback to KG
    react_response = chat_agent.get_routing_agent_response(user_message)
    chat_agent.add_chat("system", "ReAct agent: " + react_response)
    
    # Write response to KG for future queries
    with open("data/external_response.txt", "w") as f:
        f.write(react_response)
    chat_agent.write_back()

# 4. Generate final response with memory context
final_response = chat_agent.get_response()

# 5. Save final response (no entities needed, already captured above)
chat_agent.add_chat("assistant", final_response)
```

### 2. Environment Setup Requirements

Ensure these environment variables are set:
```bash
# For Knowledge Graph (choose one)
FALKORDB_URL=redis://localhost:6379
# OR
NEO4J_URL=bolt://localhost:7687
NEO4J_PW=your_password

# For external search
PERPLEXITY_API_KEY=your_key

# For location tool
GOOGLEMAPS_API_KEY=your_key

# For stocks tool
ALPHA_VANTAGE_API_KEY=your_key
```

### 3. Data Directory Structure

Ensure these files exist:
```
data/
├── memory_stream.json          # Will be populated automatically
├── entity_knowledge_store.json # Will be populated automatically
├── past_chat.json             # Chat history
├── system_persona.txt         # System persona
├── user_persona.txt           # User persona
└── external_response.txt      # Temp file for KG writeback
```

### 4. Integration Steps

#### Step 1: Update main.py chat flow
- Replace direct LLM calls with routing agent flow
- Add entity extraction and KG querying
- Implement proper memory saving

#### Step 2: Test knowledge graph connection
- Verify FalkorDB/Neo4j connection
- Test basic KG operations

#### Step 3: Validate memory system
- Confirm entities are being extracted
- Verify memory files are being updated
- Test entity knowledge store ranking

#### Step 4: Test complete flow
- Run full conversation
- Verify KG writeback
- Check memory persistence

### 5. Expected Behavior After Fix

1. **Memory Files Will Populate**: `memory_stream.json` and `entity_knowledge_store.json` will contain conversation entities
2. **Knowledge Graph Growth**: Each conversation adds entities/relationships to the graph
3. **Context-Aware Responses**: System uses previous entity knowledge to tailor responses
4. **Tool Integration**: ReAct agent can use search, vision, location tools as needed

### 6. Debugging Steps

If memory still doesn't populate:
1. Check KG connection (logs will show connection status)
2. Verify routing agent tools are working
3. Check entity extraction from queries
4. Ensure `add_chat()` is called with entities parameter

### 7. Benefits After Integration

- **Persistent Learning**: System remembers entities from conversations
- **Knowledge Growth**: Each interaction builds the knowledge graph
- **Personalized Responses**: Uses entity frequency/recency for context
- **Tool Integration**: Access to search, vision, location capabilities
- **True Memory**: Not just chat history, but semantic entity memory

## Implementation Priority

1. **High Priority**: Fix chat flow in main.py (this will immediately fix memory population)
2. **Medium Priority**: Verify KG connection and writeback
3. **Low Priority**: Enhance with additional tools and persona management

This will transform the current basic chat into a true Memary-powered agent with persistent memory and knowledge growth.
