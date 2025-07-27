import json
import logging
from dataclasses import dataclass


def save_json(filename, data):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)


@dataclass
class Context:
    """Context class to store the role and content of the message."""
    role: str  # system or user
    content: str

    def __str__(self):
        return f"{self.role}: {self.content} |"

    def to_dict(self):
        return {'role': self.role, 'content': self.content}


class Message:
    """Message class to store the contexts, memory stream and knowledge entity store."""

    def __init__(self, system_persona_txt, user_persona_txt, past_chat_json, model):
        self.past_chat_json = past_chat_json

        self.contexts = []
        self.system_persona = self.load_persona(system_persona_txt)
        self.user_persona = self.load_persona(user_persona_txt)
        self._init_persona_to_messages()
        self.contexts.extend(self.load_contexts_from_json())

        self.llm_message = {
            "model": model,
            "messages": self.contexts,
            "memory_stream": [],
            "knowledge_entity_store": []
        }

        # self.prompt_tokens = count_tokens(self.contexts)

    def __str__(self):
        llm_message_str = f"System Persona: {self.system_persona}\nUser Persona: {self.user_persona}\n"
        for context in self.contexts:
            llm_message_str += f"{str(context)},"
        for memory in self.llm_message["memory_stream"]:
            llm_message_str += f"{str(memory)},"
        for entity in self.llm_message["knowledge_entity_store"]:
            llm_message_str += f"{str(entity)},"
        return llm_message_str

    def _init_persona_to_messages(self):
        """Initializes the system and user personas to the contexts."""
        self.contexts.append(Context("system", self.system_persona))
        self.contexts.append(Context("user", self.user_persona))

    def load_persona(self, persona_txt) -> str:
        """Loads the persona from the txt file.

        Args:
            persona_txt (str): persona txt file path

        Returns:
            str: persona
        """
        try:
            with open(persona_txt, "r") as file:
                persona = file.read()
            return persona
        except FileNotFoundError:
            logging.info(f"{persona_txt} file does not exist.")
            return ""  # Return empty string if file not found

    def load_contexts_from_json(self):
        """Loads ALL contexts from the past chat json file.
        Full history is preserved for memory/search purposes.
        Limiting for LLM context happens separately when generating responses.
        """
        try:
            with open(self.past_chat_json, "r") as file:
                data_dicts = json.load(file)

            total_stored = len(data_dicts)
            logging.info(f"[CONTEXT] Loaded all {total_stored} stored messages from {self.past_chat_json}")

            return [Context(**data_dict) for data_dict in data_dicts]
        except Exception as e:
            logging.info(
                f"{self.past_chat_json} file does not exist or error loading: {e}. Starts from empty contexts."
            )
            return []

    def save_contexts_to_json(self):
        """Saves the contexts to the json file.
        We don't save the system and user personas (first two messages)
        """
        save_json(self.past_chat_json, [message.to_dict() for message in self.llm_message['messages'][2:]])

    def get_limited_messages_for_llm(self):
        """Get messages limited to MAX_HISTORY_MESSAGES for LLM context.
        This preserves full history in self.contexts but only returns recent messages for LLM.
        """
        try:
            # Import here to avoid circular imports
            try:
                # Try to get from main.py first
                import main
                max_messages = getattr(main, 'MAX_HISTORY_MESSAGES', 30)
            except Exception:
                # Fallback to base_agent constant
                from memary.agent.base_agent import MAX_HISTORY_MESSAGES
                max_messages = MAX_HISTORY_MESSAGES
            
            # Get all messages (personas + chat history)
            all_messages = self.llm_message["messages"]
            
            # If we have more than max_messages + 2 (for personas), limit the chat portion
            if len(all_messages) > max_messages + 2:
                # Keep first 2 (personas) + last max_messages chat messages
                personas = all_messages[:2]  # System and user personas
                recent_chat = all_messages[2:][-max_messages:]  # Last N chat messages
                limited_messages = personas + recent_chat
                logging.info(f"[LLM CONTEXT] Using last {max_messages} chat messages from {len(all_messages)-2} total (full history preserved)")
                return limited_messages
            else:
                logging.info(f"[LLM CONTEXT] Using all {len(all_messages)} messages (under limit)")
                return all_messages
        except Exception as e:
            logging.warning(f"Error limiting messages for LLM: {e}, using all messages")
            return self.llm_message["messages"]
