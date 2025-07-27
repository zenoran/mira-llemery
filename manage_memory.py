#!/usr/bin/env python3
"""
Memory Management Script for Mira-LLEmery
Handles clearing and managing conversation memory and persona files.
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Optional

# Data directory
DATA_DIR = Path("data")

def ensure_data_dir():
    """Ensure data directory exists."""
    DATA_DIR.mkdir(exist_ok=True)

def clear_all_memory():
    """Clear all memory files and reset personas to defaults."""
    print("Clearing all memory files...")
    
    ensure_data_dir()
    
    # Clear JSON files
    files_to_clear = {
        "past_chat.json": [],
        "memory_stream.json": [],
        "entity_knowledge_store.json": {}
    }
    
    for filename, default_content in files_to_clear.items():
        filepath = DATA_DIR / filename
        with open(filepath, 'w') as f:
            json.dump(default_content, f, indent=2)
        print(f"  ✓ Cleared {filename}")
    
    # Reset persona files
    persona_files = {
        "system_persona.txt": "You are a helpful AI assistant.",
        "user_persona.txt": "The user is having a conversation."
    }
    
    for filename, default_content in persona_files.items():
        filepath = DATA_DIR / filename
        with open(filepath, 'w') as f:
            f.write(default_content)
        print(f"  ✓ Reset {filename}")
    
    print("Memory files and personas cleared successfully!")

def clear_recent_memory(chat_keep: int = 5, memory_keep: int = 3):
    """Clear only recent problematic entries from memory files."""
    print(f"Clearing recent problematic entries (keeping last {chat_keep} chat, {memory_keep} memory entries)...")
    
    ensure_data_dir()
    
    # Clear recent chat history
    chat_file = DATA_DIR / "past_chat.json"
    try:
        if chat_file.exists():
            with open(chat_file, 'r') as f:
                chat = json.load(f)
            
            if len(chat) > chat_keep:
                removed_count = len(chat) - chat_keep
                chat = chat[:-removed_count]  # Remove recent entries
                with open(chat_file, 'w') as f:
                    json.dump(chat, f, indent=2)
                print(f"  ✓ Removed {removed_count} recent chat entries, {len(chat)} messages remain")
            else:
                print(f"  ✓ Chat history has <= {chat_keep} entries, clearing all...")
                with open(chat_file, 'w') as f:
                    json.dump([], f)
        else:
            print("  ✓ No chat history file found, creating empty one")
            with open(chat_file, 'w') as f:
                json.dump([], f)
    except Exception as e:
        print(f"  ✗ Error processing chat history: {e}")
    
    # Clear recent memory stream
    memory_file = DATA_DIR / "memory_stream.json"
    try:
        if memory_file.exists():
            with open(memory_file, 'r') as f:
                memory = json.load(f)
            
            if len(memory) > memory_keep:
                removed_count = len(memory) - memory_keep
                memory = memory[:-removed_count]  # Remove recent entries
                with open(memory_file, 'w') as f:
                    json.dump(memory, f, indent=2)
                print(f"  ✓ Removed {removed_count} recent memory entries, {len(memory)} entries remain")
            else:
                print(f"  ✓ Memory stream has <= {memory_keep} entries, clearing all...")
                with open(memory_file, 'w') as f:
                    json.dump([], f)
        else:
            print("  ✓ No memory stream file found, creating empty one")
            with open(memory_file, 'w') as f:
                json.dump([], f)
    except Exception as e:
        print(f"  ✗ Error processing memory stream: {e}")
    
    print("Recent problematic entries cleared!")

def show_memory_status():
    """Show current memory status and file sizes."""
    print("=== Memory Status ===")
    
    ensure_data_dir()
    
    # Show configuration
    try:
        # Read the MAX_HISTORY_MESSAGES directly from main.py file
        with open('main.py', 'r') as f:
            content = f.read()
        import re
        match = re.search(r'MAX_HISTORY_MESSAGES\s*=\s*(\d+)', content)
        if match:
            max_history = int(match.group(1))
            print(f"  Max History Messages : {max_history} (configured limit for LLM context)")
        else:
            print("  Max History Messages : 30 (default - not found in config)")
    except Exception:
        print("  Max History Messages : 30 (default)")
    
    print()
    
    memory_files = [
        ("past_chat.json", "Chat History"),
        ("memory_stream.json", "Memory Stream"),
        ("entity_knowledge_store.json", "Entity Knowledge"),
        ("system_persona.txt", "System Persona"),
        ("user_persona.txt", "User Persona")
    ]
    
    for filename, description in memory_files:
        filepath = DATA_DIR / filename
        if filepath.exists():
            if filename.endswith('.json'):
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        count = len(data)
                        if filename == "past_chat.json":
                            # Special handling for chat history
                            try:
                                max_history = getattr(main, 'MAX_HISTORY_MESSAGES', 30)
                                if count > max_history:
                                    print(f"  {description:20} : {count} entries (using last {max_history} for LLM)")
                                else:
                                    print(f"  {description:20} : {count} entries (all used for LLM)")
                            except:
                                print(f"  {description:20} : {count} entries")
                        else:
                            print(f"  {description:20} : {count} entries")
                    elif isinstance(data, dict):
                        count = len(data)
                        print(f"  {description:20} : {count} keys")
                    else:
                        print(f"  {description:20} : exists (unknown format)")
                except Exception as e:
                    print(f"  {description:20} : ✗ Error reading ({e})")
            else:
                # Text file
                try:
                    with open(filepath, 'r') as f:
                        content = f.read().strip()
                    lines = len(content.split('\n')) if content else 0
                    chars = len(content)
                    print(f"  {description:20} : {lines} lines, {chars} chars")
                except Exception as e:
                    print(f"  {description:20} : ✗ Error reading ({e})")
        else:
            print(f"  {description:20} : ✗ Missing")

def backup_memory(backup_name: Optional[str] = None):
    """Create a backup of current memory state."""
    if backup_name is None:
        from datetime import datetime
        backup_name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    backup_dir = Path("backups")
    backup_dir.mkdir(exist_ok=True)
    
    backup_path = backup_dir / backup_name
    backup_path.mkdir(exist_ok=True)
    
    print(f"Creating memory backup: {backup_path}")
    
    ensure_data_dir()
    
    # Copy all data files
    import shutil
    for file_path in DATA_DIR.glob("*"):
        if file_path.is_file():
            dest_path = backup_path / file_path.name
            shutil.copy2(file_path, dest_path)
            print(f"  ✓ Backed up {file_path.name}")
    
    print(f"Backup created successfully in {backup_path}")
    return backup_path

def restore_memory(backup_name: str):
    """Restore memory from a backup."""
    backup_dir = Path("backups")
    backup_path = backup_dir / backup_name
    
    if not backup_path.exists():
        print(f"✗ Backup not found: {backup_path}")
        return False
    
    print(f"Restoring memory from backup: {backup_path}")
    
    ensure_data_dir()
    
    # Copy files from backup
    import shutil
    for file_path in backup_path.glob("*"):
        if file_path.is_file():
            dest_path = DATA_DIR / file_path.name
            shutil.copy2(file_path, dest_path)
            print(f"  ✓ Restored {file_path.name}")
    
    print("Memory restored successfully!")
    return True

def list_backups():
    """List available backups."""
    backup_dir = Path("backups")
    if not backup_dir.exists():
        print("No backups directory found.")
        return
    
    backups = list(backup_dir.iterdir())
    if not backups:
        print("No backups found.")
        return
    
    print("Available backups:")
    for backup_path in sorted(backups):
        if backup_path.is_dir():
            # Get creation time
            import os
            import time
            ctime = os.path.getctime(backup_path)
            ctime_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ctime))
            print(f"  {backup_path.name} (created: {ctime_str})")

def main():
    parser = argparse.ArgumentParser(
        description="Memory Management for Mira-LLEmery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python manage_memory.py clear-all          # Clear all memory
  python manage_memory.py clear-recent       # Clear recent problematic entries
  python manage_memory.py clear-recent -c 10 -m 5  # Keep last 10 chat, 5 memory
  python manage_memory.py status             # Show memory status
  python manage_memory.py backup             # Create timestamped backup
  python manage_memory.py backup mybackup    # Create named backup
  python manage_memory.py restore mybackup   # Restore from backup
  python manage_memory.py list-backups       # List available backups
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Clear all command
    subparsers.add_parser('clear-all', help='Clear all memory files and reset personas')
    
    # Clear recent command
    clear_recent_parser = subparsers.add_parser('clear-recent', help='Clear recent problematic entries')
    clear_recent_parser.add_argument('-c', '--chat-keep', type=int, default=5,
                                   help='Number of recent chat entries to keep (default: 5)')
    clear_recent_parser.add_argument('-m', '--memory-keep', type=int, default=3,
                                   help='Number of recent memory entries to keep (default: 3)')
    
    # Status command
    subparsers.add_parser('status', help='Show current memory status')
    
    # Backup command
    backup_parser = subparsers.add_parser('backup', help='Create memory backup')
    backup_parser.add_argument('name', nargs='?', help='Backup name (default: timestamped)')
    
    # Restore command
    restore_parser = subparsers.add_parser('restore', help='Restore from backup')
    restore_parser.add_argument('name', help='Backup name to restore from')
    
    # List backups command
    subparsers.add_parser('list-backups', help='List available backups')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'clear-all':
            clear_all_memory()
        elif args.command == 'clear-recent':
            clear_recent_memory(args.chat_keep, args.memory_keep)
        elif args.command == 'status':
            show_memory_status()
        elif args.command == 'backup':
            backup_memory(args.name)
        elif args.command == 'restore':
            restore_memory(args.name)
        elif args.command == 'list-backups':
            list_backups()
        else:
            print(f"Unknown command: {args.command}")
            parser.print_help()
    except KeyboardInterrupt:
        print("\nOperation cancelled.")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
