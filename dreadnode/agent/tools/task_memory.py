import rigging as rg
import typing as t
import copy


class TaskMemory:
    """Basic set of memory tool for agents. TaskMemory is designed to be attached to an Agent class and then the TaskMemory tools (methods) are specified as tools to the Agent. In effect, when the Agent uses these memory tools, it is manipulating its own memory instance."""

    def __init__(self):
        """ """
        self.timestamps = {}
        self.tool_calls = []
        self._data = {}
        self._notes = []
        self._output = {}

    @rg.tool_method
    async def store_note(self, note: t.Annotated[str, "reasoning note to save"]) -> str:
        """store a reasoning or Chain-of-Thought note that occurs while reasoning and trying to achieve the task."""
        self._notes.append(note)
        return "Reasoning note was saved."
    
    @rg.tool_method
    async def get_notes(self) -> str:
        """retrieve all previously stored reasoning notes"""
        return copy.deepcopy(self._notes)

    @rg.tool_method
    async def store_data(
        self,
        key: t.Annotated[str, "key to retrieve data"],
        value: t.Annotated[str | float | int, "value to store"],
    ) -> str:
        """Store data in memory. Provide a key and value to store. The value can be retrieved later with the specified key."""
        self._data[key] = value
        return f"Data was stored in memory. Retrievable with the key '{key}'"

    @rg.tool_method
    async def get_data(self, key: t.Annotated[str, "key for data to retrieve"] = None):
        """Retrieve data from memory based on the supplied key value."""
        if key is None:
            return copy.deepcopy(self._data)
        return self._data.get(key, f"Key {key} was not found in stored data.")

    @rg.tool_method
    async def store_output(
        self,
        data_object: t.Annotated[str, "XML structured data object, encoded as a string"],
    ) -> str:
        """used to output data objects to the user as appropriate for the specified task. data can be outputted to the user at any time while executing the task."""

        saved = set([])
        for dm in self.output_data_models:
            for i in rg.parsing.try_parse_set(text=data_object, model_type=dm):
                if dm.__name__ not in self._output:
                    self._output[dm.__name__] = []
                self._output[dm.__name__].append(i[0])  # try_parse_set() returns (object, slice), not storing slice
                saved.add(dm.__name__)

        if len(saved) == 0:
            return f"Data object could not be parsed. The data object should be one of the following types: {self.output_data_models}. Also, make sure the data object is formatted in structured XML, where it should be enclosed with a data object type tag: <data_object_type> ... </data_object_type>."

        return f"Output was saved."
    
    @rg.tool_method
    async def get_output(self):
        """Retrieve all previously outputted data."""
        return copy.deepcopy(self._output)
