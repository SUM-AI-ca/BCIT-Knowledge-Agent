"""Windowed conversation memory.

Replaces `langchain.memory.ConversationBufferWindowMemory`, which does not
exist in langchain 1.x. Only three things were ever used from it — the
constructor, `chat_memory.add_user_message/add_ai_message`, and
`buffer_as_messages` — so carrying a 25-line class is cheaper than tracking
whichever package the deprecated one lands in next.

The windowing semantics are copied deliberately, not reinvented:
`buffer_as_messages` returned `messages[-k * 2:]`, i.e. the last k *turns*
(a turn being one human + one AI message), and an empty list when k == 0.
Any drift here silently changes what every follow-up question sees, which
would move the eval sets without touching retrieval — so it is reproduced
exactly and covered by `_tmp_memory_parity.py`.
"""
from typing import List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


class _ChatMessageHistory:
    """The `.chat_memory` attribute: an unbounded append-only message list."""

    def __init__(self):
        self.messages: List[BaseMessage] = []

    def add_user_message(self, message: str) -> None:
        self.messages.append(HumanMessage(content=message))

    def add_ai_message(self, message: str) -> None:
        self.messages.append(AIMessage(content=message))

    def clear(self) -> None:
        self.messages.clear()


class SessionMemory:
    """Last-k-turns view over an unbounded history.

    `chat_memory.messages` is the raw list; `buffer_as_messages` is the
    windowed view. Callers formatting history for a prompt must use the
    latter — the raw list is unbounded and would grow the prompt without
    limit.
    """

    def __init__(self, k: int = 5, memory_key: str = "chat_history",
                 return_messages: bool = True):
        # memory_key/return_messages are accepted so call sites read the same
        # as before; this class only ever behaves as return_messages=True.
        self.k = k
        self.memory_key = memory_key
        self.return_messages = return_messages
        self.chat_memory = _ChatMessageHistory()

    @property
    def buffer_as_messages(self) -> List[BaseMessage]:
        return self.chat_memory.messages[-self.k * 2:] if self.k > 0 else []

    def clear(self) -> None:
        self.chat_memory.clear()
