#!/bin/bash
# Test Script for Mira-LLEmery
# Handles testing server functionality, recovery detection, and emotion handling

SERVER_URL="http://localhost:8000"

show_help() {
    echo "Test Commands for Mira-LLEmery:"
    echo ""
    echo "Status & Health:"
    echo "  ./test.sh status       - Check server status and health"
    echo "  ./test.sh ping         - Simple ping test"
    echo ""
    echo "Recovery & Diagnostics:"
    echo "  ./test.sh recovery     - Test if system needs recovery"
    echo "  ./test.sh chat \"prompt\" - Send test chat request"
    echo ""
    echo "Feature Testing:"
    echo "  ./test.sh emotion      - Test emotion detection"
    echo "  ./test.sh streaming    - Test streaming responses"
    echo "  ./test.sh bypass       - Test bypass mode (no memory)"
    echo ""
    echo "Quick Tests:"
    echo "  ./test.sh hello        - Simple hello test"
    echo "  ./test.sh name         - Ask for name"
    echo ""
    echo "Examples:"
    echo "  ./test.sh chat \"Hi there!\"           # Simple chat test"
    echo "  ./test.sh chat \"I feel sad\" emotion  # Chat with emotion checking"
    echo ""
}

check_server_status() {
    echo "Checking server status..."
    
    # Check if process is running
    if pgrep -f "python main.py" > /dev/null; then
        pid=$(pgrep -f "python main.py")
        echo "✅ Server is running (PID: $pid)"
        
        # Check if server is responding
        if curl -s -f "$SERVER_URL/health" > /dev/null 2>&1; then
            echo "✅ Server is responding"
            return 0
        else
            echo "❌ Server is not responding"
            return 1
        fi
    else
        echo "❌ Server is not running"
        return 1
    fi
}

ping_test() {
    echo "Performing ping test..."
    
    response=$(curl -s -w "%{http_code}" -o /dev/null "$SERVER_URL/health" 2>/dev/null)
    
    if [ "$response" = "200" ]; then
        echo "✅ Server ping successful (HTTP 200)"
    else
        echo "❌ Server ping failed (HTTP $response)"
        return 1
    fi
}

test_recovery() {
    echo "Testing system state with simple question..."
    
    response=$(curl -s -X POST "$SERVER_URL/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{"model": "local-llama", "messages": [{"role": "user", "content": "Hi"}], "stream": false}' \
        2>/dev/null | jq -r '.choices[0].message.content // empty' 2>/dev/null)
    
    if [ -z "$response" ]; then
        echo "❌ System appears broken - no response received"
        echo "🔧 Run 'make restart' to recover"
        return 1
    elif [ "$(echo "$response" | wc -c)" -lt 10 ]; then
        echo "❌ System appears broken - response too short: '$response'"
        echo "🔧 Run 'make restart' to recover"
        return 1
    elif echo "$response" | grep -qE "^\s*(user|assistant|human|ai)\s*$"; then
        echo "❌ System appears broken - response is role artifact: '$response'"
        echo "🔧 Run 'make restart' to recover"
        return 1
    else
        echo "✅ System appears healthy - response: '$response'"
        return 0
    fi
}

test_chat() {
    local prompt="$1"
    local check_emotion="$2"
    
    if [ -z "$prompt" ]; then
        prompt="Hello, how are you?"
    fi
    
    echo "Sending chat request: '$prompt'"
    
    response=$(curl -s -X POST "$SERVER_URL/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\": \"local-llama\", \"messages\": [{\"role\": \"user\", \"content\": \"$prompt\"}], \"stream\": false}" \
        2>/dev/null)
    
    if [ $? -ne 0 ] || [ -z "$response" ]; then
        echo "❌ Request failed or no response"
        return 1
    fi
    
    # Pretty print the response
    echo "Response:"
    echo "$response" | jq . 2>/dev/null || echo "$response"
    
    # Check for emotion data if requested
    if [ "$check_emotion" = "emotion" ]; then
        emotion_data=$(echo "$response" | jq -r '.emotion_data // empty' 2>/dev/null)
        if [ -n "$emotion_data" ] && [ "$emotion_data" != "null" ]; then
            echo ""
            echo "Emotion Detection:"
            echo "$emotion_data" | jq . 2>/dev/null
        else
            echo ""
            echo "No emotion data detected in response"
        fi
    fi
}

test_emotion() {
    echo "Testing emotion detection with sample emotional prompt..."
    
    response=$(curl -s -X POST "$SERVER_URL/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{"model": "local-llama", "messages": [{"role": "user", "content": "I am feeling really sad today. Can you help cheer me up?"}], "stream": false}' \
        2>/dev/null)
    
    if [ $? -ne 0 ] || [ -z "$response" ]; then
        echo "❌ Emotion test failed - no response"
        return 1
    fi
    
    echo "Emotion Test Response:"
    echo "$response" | jq . 2>/dev/null || echo "$response"
    
    # Check for emotion data
    emotion_data=$(echo "$response" | jq -r '.emotion_data // empty' 2>/dev/null)
    if [ -n "$emotion_data" ] && [ "$emotion_data" != "null" ]; then
        echo ""
        echo "✅ Emotion detection working:"
        echo "$emotion_data" | jq . 2>/dev/null
    else
        echo ""
        echo "⚠️  No emotion data detected (might be normal)"
    fi
}

test_streaming() {
    echo "Testing streaming response..."
    
    echo "Sending streaming request..."
    curl -s -X POST "$SERVER_URL/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{"model": "local-llama", "messages": [{"role": "user", "content": "Tell me a short joke"}], "stream": true}' \
        2>/dev/null | head -20
    
    echo ""
    echo "Streaming test completed (showing first 20 lines)"
}

test_bypass() {
    echo "Testing bypass mode (no memory)..."
    
    response=$(curl -s -X POST "$SERVER_URL/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{"model": "local-llama", "messages": [{"role": "user", "content": "Hello in bypass mode"}], "stream": false, "bypass_memory": true}' \
        2>/dev/null)
    
    if [ $? -ne 0 ] || [ -z "$response" ]; then
        echo "❌ Bypass test failed - no response"
        return 1
    fi
    
    echo "Bypass Mode Response:"
    echo "$response" | jq . 2>/dev/null || echo "$response"
}

test_hello() {
    echo "Quick hello test..."
    test_chat "Hello!"
}

test_name() {
    echo "Quick name test..."
    test_chat "What's your name?"
}

# Main script logic
case "$1" in
    "help"|""|"-h"|"--help")
        show_help
        ;;
    "status")
        check_server_status
        ;;
    "ping")
        ping_test
        ;;
    "recovery")
        test_recovery
        ;;
    "chat")
        test_chat "$2" "$3"
        ;;
    "emotion")
        test_emotion
        ;;
    "streaming")
        test_streaming
        ;;
    "bypass")
        test_bypass
        ;;
    "hello")
        test_hello
        ;;
    "name")
        test_name
        ;;
    *)
        echo "Unknown command: $1"
        echo "Use './test.sh help' to see available commands"
        exit 1
        ;;
esac
