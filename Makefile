# Makefile for Mira-Llemery FastAPI App

.PHONY: help start restart regen-persona show-persona install stop logs start-bg status clean

# Default target
help:
	@echo "Available commands:"
	@echo "  make start         - Start the FastAPI server"
	@echo "  make start-bg      - Start server in background"
	@echo "  make restart       - Emergency restart (stop, clear recent memory, start)"
	@echo "  make stop          - Stop the server"
	@echo "  make status        - Check server status"
	@echo "  make logs          - Show server logs"
	@echo ""
	@echo "Memory Management (use memory script):"
	@echo "  ./manage_memory.py clear-all    - Clear all memory files"
	@echo "  ./manage_memory.py clear-recent - Clear recent problematic entries"
	@echo "  ./manage_memory.py status       - Show memory status"
	@echo "  ./manage_memory.py backup       - Create memory backup"
	@echo "  ./manage_memory.py restore NAME - Restore from backup"
	@echo ""
	@echo "Persona Management:"
	@echo "  make regen-persona - Regenerate user persona from conversation history"
	@echo "  make show-persona  - Show current system and user personas"
	@echo ""
	@echo "Testing (use test script):"
	@echo "  ./test.sh help     - Show all test commands"
	@echo "  ./test.sh recovery - Test if system needs recovery"
	@echo "  ./test.sh emotion  - Test emotion detection"
	@echo "  ./test.sh chat \"prompt\" - Send test chat request"
	@echo ""
	@echo "Development:"
	@echo "  make install       - Install dependencies with uv"
	@echo "  make clean         - Clean up temporary files"

# Start the FastAPI server
start:
	@echo "Starting FastAPI server on port 8000..."
	uv run python main.py

# Start server in background
start-bg:
	@echo "Starting FastAPI server in background on port 8000..."
	nohup uv run python main.py > server.log 2>&1 &
	@echo "Server started in background. Use 'make logs' to view logs or 'make stop' to stop."

# Emergency restart - stop server, clear recent problems, restart
restart:
	@echo "Emergency restart: stopping server, clearing recent memory issues, and restarting..."
	@-make stop
	@sleep 2
	@./manage_memory.py clear-recent
	@echo "Restarting server..."
	@make start-bg

# Regenerate user persona based on conversation history
regen-persona:
	@echo "Regenerating user persona from conversation history..."
	@curl -s -X POST "http://localhost:8000/regenerate-persona" \
		-H "Content-Type: application/json" \
		| jq .

# Show current persona
show-persona:
	@echo "=== Current System Persona ==="
	@cat data/system_persona.txt
	@echo
	@echo "=== Current User Persona ==="
	@cat data/user_persona.txt

# Install dependencies
install:
	@echo "Installing dependencies with uv..."
	uv sync

# Stop the server (if running in background)
stop:
	@echo "Stopping FastAPI server..."
	@pkill -f "python main.py" 2>/dev/null || echo "No server process found"

# Show server logs
logs:
	@if [ -f server.log ]; then \
		echo "=== Server Logs ==="; \
		tail -f server.log; \
	else \
		echo "No server.log file found. Server may not be running in background."; \
	fi

# Show server status
status:
	@./test.sh status

# Development helpers
dev-install:
	@echo "Installing development dependencies..."
	uv add --dev pytest black flake8 mypy

format:
	@echo "Formatting code with black..."
	uv run black .

lint:
	@echo "Linting code with flake8..."
	uv run flake8 .

type-check:
	@echo "Type checking with mypy..."
	uv run mypy .

# Clean up temporary files
clean:
	@echo "Cleaning up temporary files..."
	rm -f server.log
	rm -f nohup.out
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	find . -name "*.pyc" -delete
	@echo "Cleanup complete!"
