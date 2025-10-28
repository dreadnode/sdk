import typing as t

from pydantic import PrivateAttr

from dreadnode.agent.tools import Toolset, tool_method


class Memory(Toolset):
    """
    Provides a stateful, in-memory key-value store for the toolset's lifetime.

    This toolset allows the agent to save, retrieve, and manage data, enabling it to
    remember information across multiple steps and tool calls.
    """

    _memory: dict[str, str] = PrivateAttr(default_factory=dict)

    @tool_method
    def save_memory(
        self,
        key: t.Annotated[str, "The unique key to store the value under."],
        value: t.Annotated[str, "The string value to store in memory."],
    ) -> str:
        """Saves a value to memory with the specified key, overwriting any existing value."""
        self._memory[key] = value
        return f"Value saved to memory key: '{key}'"

    @tool_method(catch=True)
    def retrieve_memory(self, key: t.Annotated[str, "The key of the value to retrieve."]) -> str:
        """Retrieves a value from memory using the specified key."""
        return self._memory[key]

    @tool_method
    def list_memory_keys(self) -> list[str]:
        """Lists all keys currently stored in memory."""
        return list(self._memory.keys())

    @tool_method(catch=True)
    def clear_memory(
        self,
        key: t.Annotated[
            str | None, "The specific key to clear. If not provided, all memory is cleared."
        ] = None,
    ) -> str:
        """
        Clears a specific key from memory, or clears all memory if no key is provided.
        """
        if key is None:
            self._memory.clear()
            return "All memory has been cleared."

        if key not in self._memory:
            return f"Key '{key}' not found in memory. Nothing to clear."

        del self._memory[key]
        return f"Cleared memory for key: '{key}'"
