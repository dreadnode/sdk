import asyncio
import re
import types
import typing as t
import uuid
from dataclasses import field
from pathlib import Path

import aiodocker
import aiodocker.containers
import aiodocker.types
import aiohttp
import tenacity
from loguru import logger
from pydantic import BaseModel

from dreadnode import log_output
from dreadnode.agent.tools import Toolset, tool_method

# Helpers

ANSI_ESCAPE_PATTERN = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def strip_ansi_codes(text: str) -> str:
    return ANSI_ESCAPE_PATTERN.sub("", text)


def parse_memory_limit(limit: str) -> int:
    """Convert memory limit string to bytes integer."""
    if limit.lower().endswith("g"):
        return int(float(limit[:-1]) * 1024 * 1024 * 1024)
    if limit.lower().endswith("m"):
        return int(float(limit[:-1]) * 1024 * 1024)
    if limit.lower().endswith("k"):
        return int(float(limit[:-1]) * 1024)
    # Assume bytes if no unit specified
    return int(float(limit))


# Models

AnyDict = dict[str, t.Any]
KernelState = t.Literal["starting", "idle", "busy"]


class NotebookCell(BaseModel):
    """A cell in a Jupyter notebook."""

    cell_type: t.Literal["code", "markdown", "raw"]
    source: str | list[str]
    metadata: AnyDict = {}
    outputs: list[AnyDict] = []
    execution_count: int | None = None

    @classmethod
    def from_source(cls, source: str | list[str]) -> "NotebookCell":
        """Create a code cell from a source string."""
        return cls(cell_type="code", source=source, metadata={}, outputs=[], execution_count=None)


class Notebook(BaseModel):
    """A Jupyter notebook."""

    cells: list[NotebookCell] = field(default_factory=list)
    metadata: AnyDict = field(default_factory=dict)
    nbformat: int = 4
    nbformat_minor: int = 5

    @classmethod
    def from_source(cls, source: str | list[str]) -> "Notebook":
        """Create a notebook from a source string."""
        return cls(cells=[NotebookCell.from_source(source)])

    @classmethod
    def load(cls, path: Path | str) -> "Notebook":
        """Load a notebook from a file."""
        return cls.model_validate_json(Path(path).read_text())

    def save(self, path: Path | str) -> None:
        """Save a notebook to a file."""
        Path(path).write_text(self.model_dump_json())

    def __add__(self, other: "Notebook | NotebookCell | str") -> "Notebook":
        """Add a cell to the notebook."""
        if isinstance(other, NotebookCell):
            return Notebook(cells=[*self.cells, other])
        if isinstance(other, Notebook):
            return Notebook(cells=self.cells + other.cells)
        if isinstance(other, str):
            return self + NotebookCell.from_source(other)
        raise TypeError(f"Cannot add {type(other)} to Notebook")

    def to_markdown(self) -> str:
        """Convert the notebook to a markdown string."""
        markdown_chunks: list[str] = []

        for cell in self.cells:
            source = "".join(cell.source) if isinstance(cell.source, list) else cell.source
            if not source.strip():
                continue

            if cell.cell_type == "markdown":
                markdown_chunks.append(source.strip())
                markdown_chunks.append("\n\n")

            elif cell.cell_type == "code":
                markdown_chunks.append("```python\n")
                markdown_chunks.append(source.strip())
                markdown_chunks.append("\n```")
                markdown_chunks.append("\n\n")

        return "".join(markdown_chunks).strip()


class KernelExecution(BaseModel):
    """Result of executing code in a kernel."""

    source: str
    outputs: list[AnyDict] = []
    error: str | None = None
    execution_count: int | None = None

    @property
    def success(self) -> bool:
        """Check if the execution was successful."""
        return not self.error

    def to_cell(self) -> NotebookCell:
        """Convert the execution result to a notebook cell."""
        return NotebookCell(
            cell_type="code",
            source=self.source.splitlines(),
            metadata={},
            outputs=self.outputs,
            execution_count=self.execution_count,
        )

    def to_notebook(self) -> Notebook:
        """Convert the execution result to a notebook."""
        return Notebook(cells=[self.to_cell()])

    def to_str(self) -> str:
        """Get the stdout output as a string."""
        output_str: str = ""
        for output in self.outputs:
            if output["output_type"] == "stream":
                output_str += output["text"]
            elif (
                output["output_type"] in ["display_data", "execute_result"]
                and "text/plain" in output["data"]
            ):
                output_str += output["data"]["text/plain"]

        return output_str + (self.error or "")


# Exceptions


class PythonKernelNotRunningError(Exception):
    """Raised when trying to manage a kernel that is not running."""

    def __init__(self, message: str = "Kernel is not running") -> None:
        super().__init__(message)


class PythonKernelStartError(Exception):
    """Raised when the kernel fails to start."""

    def __init__(self, message: str = "Failed to start kernel") -> None:
        super().__init__(message)


# Main class


class PythonKernel(Toolset):
    """A Python kernel for executing code."""

    def __init__(
        self,
        image: str = "jupyter/datascience-notebook:latest",
        *,
        memory_limit: str = "4g",
        kernel_name: str = "python3",
        work_dir: Path | str | None = None,
        volumes: list[str] | None = None,
    ) -> None:
        """Create a python kernel."""
        self.image = image
        self.memory_limit = memory_limit
        self.kernel_name = kernel_name
        self.volumes = volumes or []

        self._token = uuid.uuid4().hex

        self._client: aiodocker.Docker | None = None
        self._container: aiodocker.containers.DockerContainer | None = None
        self._work_dir = Path(work_dir or f".work/{uuid.uuid4().hex[:8]}")
        self._kernel_id: str | None = None
        self._base_url: str | None = None

    @property
    def base_url(self) -> str:
        """Get the base URL for the Jupyter server."""
        if not self._base_url:
            raise PythonKernelNotRunningError
        return self._base_url

    @property
    def ws_url(self) -> str:
        """Get the websocket URL for the kernel."""
        if not self._base_url or not self._kernel_id:
            raise PythonKernelNotRunningError
        return f"{self._base_url.replace('http', 'ws')}/api/kernels/{self._kernel_id}/channels?token={self._token}"

    @property
    def client(self) -> aiodocker.Docker:
        """Get the docker client."""
        if not self._client:
            self._client = aiodocker.Docker()
        return self._client

    @property
    def container(self) -> aiodocker.containers.DockerContainer:
        """Get the running docker container."""
        if not self._container:
            raise PythonKernelNotRunningError
        return self._container

    @property
    def work_dir(self) -> Path:
        return self._work_dir

    # Internals

    @logger.catch(message="Failed to start container", reraise=True)
    async def _start_container(self) -> None:
        try:
            await self.client.images.inspect(self.image)
        except aiodocker.exceptions.DockerError:
            logger.info(f"Pulling {self.image} ...")
            await self.client.images.pull(self.image)

        # Create and start container
        container_config: aiodocker.types.JSONObject = {
            "Image": self.image,
            "ExposedPorts": {"8888/tcp": {}},
            "HostConfig": {
                "Memory": parse_memory_limit(self.memory_limit),
                "MemorySwap": -1,  # Disable swap
                "PortBindings": {
                    "8888/tcp": [{"HostPort": "0"}],  # Let Docker choose a port
                },
                "Binds": [f"{self._work_dir.absolute()!s}:/home/jovyan/work", *self.volumes],
            },
            "Env": [
                f"JUPYTER_TOKEN={self._token}",
                "JUPYTER_ALLOW_INSECURE_WRITES=true",
            ],
            "Cmd": ["jupyter", "server", "--ip=0.0.0.0", "--no-browser"],
        }

        self._container = await self.client.containers.create(config=container_config)
        await self._container.start()

        container_info = await self._container.show()
        host_port = container_info["NetworkSettings"]["Ports"]["8888/tcp"][0]["HostPort"]
        self._base_url = f"http://localhost:{host_port}"

        await self._wait_for_jupyter()

        logger.debug(
            f"Python kernel container started at {self._base_url} with token {self._token} (Memory: {self.memory_limit})",
        )

    @logger.catch(message="Jupyter server did not start", reraise=True)
    @tenacity.retry(stop=tenacity.stop_after_delay(30), wait=tenacity.wait_fixed(1))
    async def _wait_for_jupyter(self) -> None:
        container_info = await self.container.show()
        if container_info["State"]["Status"] != "running":
            raise PythonKernelStartError("Container did not stay running")

        async with (
            aiohttp.ClientSession() as session,
            session.get(
                f"{self.base_url}/api/status",
                params={"token": self._token},
                timeout=1,
            ) as response,
        ):
            response.raise_for_status()

    @logger.catch(message="Failed to start kernel", reraise=True)
    async def _start_kernel(self) -> None:
        async with (
            aiohttp.ClientSession() as session,
            session.post(
                f"{self._base_url}/api/kernels",
                params={"token": self._token},
                json={"name": self.kernel_name},
            ) as response,
        ):
            response.raise_for_status()

            kernel_info = await response.json()
            self._kernel_id = kernel_info["id"]

        logger.debug(f"Started kernel '{self.kernel_name}' ({self._kernel_id})")

    @logger.catch(message="Failed to delete kernel")
    async def _delete_kernel(self) -> None:
        if not self._kernel_id:
            return

        async with (
            aiohttp.ClientSession() as session,
            session.delete(
                f"{self._base_url}/api/kernels/{self._kernel_id}",
                params={"token": self._token},
            ) as response,
        ):
            response.raise_for_status()

        self._kernel_id = None
        logger.debug(f"Deleted kernel '{self.kernel_name}' ({self._kernel_id})")

    @logger.catch(message="Failed to delete container")
    async def _delete_container(self) -> None:
        if not self._container:
            return

        container_info = await self._container.show()
        container_id = container_info["Id"]

        logger.debug(f"Stopping container {container_id[:12]}...")
        await self._container.stop(timeout=5)
        await self._container.delete()

        self._container = None
        logger.debug(f"Removed container {container_id[:12]}")

    # Init / Shutdown

    async def init(self) -> "PythonKernel":
        """Initialize the container and kernel server."""
        await self.shutdown()

        self._client = aiodocker.Docker()

        await self._start_container()
        await self._start_kernel()
        return self

    async def shutdown(self) -> None:
        """Clean up resources and reset state."""
        if self._client is None:
            return

        logger.debug("Shutting down kernel and container...")

        # First, delete the kernel
        with logger.catch(Exception, message="Error during kernel shutdown"):
            await self._delete_kernel()

        # Then, delete the container
        with logger.catch(Exception, message="Error during container shutdown"):
            await self._delete_container()

        # Close the Docker client
        if self._client:
            with logger.catch(Exception, message="Error during Docker client shutdown"):
                await self._client.close()
            self._client = None

        logger.debug("Kernel shutdown complete")

    async def __aenter__(self) -> "PythonKernel":
        """Start a Jupyter server and kernel."""
        return await self.init()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Stop the kernel and container."""
        await self.shutdown()

    async def get_container_logs(self) -> str:
        """Get the logs of the container."""
        if not self._container:
            return ""

        logs = await self._container.log(stdout=True, stderr=True)
        return "\n".join(logs)

    @t.overload
    async def execute(
        self,
        source: str | list[str],
        *,
        format: None = ...,
        timeout: int = ...,
        log_output: bool = ...,
    ) -> KernelExecution: ...

    @t.overload
    async def execute(
        self,
        source: str | list[str],
        *,
        format: t.Literal["str"],
        timeout: int = ...,
        log_output: bool = ...,
    ) -> str: ...

    @t.overload
    async def execute(
        self,
        source: str | list[str],
        *,
        format: t.Literal["cell"],
        timeout: int = ...,
        log_output: bool = ...,
    ) -> NotebookCell: ...

    @t.overload
    async def execute(
        self,
        source: str | list[str],
        *,
        format: t.Literal["notebook"],
        timeout: int = ...,
        log_output: bool = ...,
    ) -> Notebook: ...

    async def execute(
        self,
        source: str | list[str],
        *,
        format: t.Literal["str", "cell", "notebook"] | None = None,
        timeout: int = 30,
        log_output: bool = False,
    ) -> KernelExecution | Notebook | NotebookCell | str:
        """Execute code in the kernel."""
        msg_id = str(uuid.uuid4())
        source = "".join(source) if isinstance(source, list) else source
        execute_request = {
            "header": {
                "msg_id": msg_id,
                "username": "user",
                "session": str(uuid.uuid4()),
                "msg_type": "execute_request",
                "version": "5.0",
            },
            "parent_header": {},
            "metadata": {},
            "content": {
                "code": source,
                "silent": False,
                "store_history": True,
                "user_expressions": {},
                "allow_stdin": False,
            },
        }

        outputs: list[AnyDict] = []
        error: str | None = None
        execution_count: int | None = None

        start_time = asyncio.get_event_loop().time()

        async with aiohttp.ClientSession() as session, session.ws_connect(self.ws_url) as ws:
            await ws.send_json(execute_request)

            while (start_time + timeout) > asyncio.get_event_loop().time():
                try:
                    msg = await asyncio.wait_for(ws.receive_json(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                # Ensure this is for us
                if msg.get("parent_header", {}).get("msg_id") != msg_id:
                    continue

                msg_type = msg.get("header", {}).get("msg_type")
                content = msg.get("content", {})

                if msg_type == "execute_result":
                    result_output = {
                        "output_type": "execute_result",
                        "metadata": content.get("metadata", {}),
                        "data": content.get("data", {}),
                        "execution_count": content.get("execution_count"),
                    }
                    outputs.append(result_output)
                    execution_count = content.get("execution_count")

                    if log_output:
                        logger.info(content.get("data", {}).get("text/plain", ""))

                elif msg_type == "display_data":
                    display_output = {
                        "output_type": "display_data",
                        "metadata": content.get("metadata", {}),
                        "data": content.get("data", {}),
                    }
                    outputs.append(display_output)

                    if log_output:
                        logger.info(content.get("data", {}).get("text/plain", ""))

                elif msg_type == "stream":
                    clean_text = strip_ansi_codes(content.get("text", ""))
                    stream_name = content.get("name", "stdout")

                    # Try to append to an existing stream output
                    for i, output in enumerate(outputs):
                        if output["output_type"] == "stream" and output["name"] == stream_name:
                            outputs[i]["text"] += clean_text
                            break
                    else:
                        # Create a new stream output
                        if stream_name not in ("stdout", "stderr"):
                            stream_name = "stdout"

                        stream_output = {
                            "output_type": "stream",
                            "name": stream_name,
                            "text": clean_text,
                        }
                        outputs.append(stream_output)

                    if log_output:
                        logger.info(clean_text)

                elif msg_type == "error":
                    traceback = content.get("traceback", [])
                    error_output = {
                        "output_type": "error",
                        "ename": content.get("ename", ""),
                        "evalue": content.get("evalue", ""),
                        "traceback": traceback,
                    }
                    outputs.append(error_output)
                    error = strip_ansi_codes("\n".join(traceback))

                elif msg_type == "execute_reply":
                    # In case we didn't receive an error message
                    if content.get("status") == "error" and not error:
                        error = f"{content.get('ename', '')}: {content.get('evalue', '')}"
                    # We're done processing this execution
                    break
            else:
                await self.interrupt()
                raise asyncio.TimeoutError("Execution timed out")

        execution = KernelExecution(
            source=source,
            outputs=outputs,
            error=error,
            execution_count=execution_count,
        )

        match format:
            case "str":
                return execution.to_str()
            case "cell":
                return execution.to_cell()
            case "notebook":
                return execution.to_notebook()
            case _:
                return execution

    @tool_method()
    async def execute_cell(self, cell: NotebookCell) -> NotebookCell:
        """Execute a notebook cell."""
        cell = cell.model_copy(deep=True)

        if cell.cell_type != "code":
            return cell

        result = await self.execute(cell.source)

        cell.outputs = result.outputs
        cell.execution_count = result.execution_count or cell.execution_count

        return cell

    @tool_method()
    async def execute_notebook(
        self,
        notebook: Notebook,
        *,
        stop_on_error: bool = True,
        log_output: bool = True,
    ) -> Notebook:
        """Execute all cells in a notebook."""
        notebook = notebook.model_copy(deep=True)

        # Reset all outputs
        for cell in notebook.cells:
            if cell.cell_type == "code":
                cell.outputs = []
                cell.execution_count = None

        # Execute each cell
        logger.info(f"Executing notebook with {len(notebook.cells)} cells")
        for i, cell in enumerate(notebook.cells):
            if cell.cell_type != "code":
                continue

            result = await self.execute(cell.source, log_output=log_output)
            if not result.success and stop_on_error:
                logger.error(f"Error in cell {i}: {result.error}")
                break

            cell.outputs = result.outputs
            cell.execution_count = result.execution_count or cell.execution_count

        return notebook

    @tool_method()
    async def execute_code(self, code: str) -> str:
        """
        Execute Python code in the jupyter kernel and return the output.
        """
        results = await self.execute(code, format="str")

        log_output("python_code_output", results)
        return await self.execute(code, format="str")

    async def get_kernel_state(self) -> KernelState:
        """Get the state of the kernel."""
        if not self._kernel_id:
            raise PythonKernelNotRunningError

        async with (
            aiohttp.ClientSession() as session,
            session.get(
                f"{self._base_url}/api/kernels/{self._kernel_id}",
                params={"token": self._token},
            ) as response,
        ):
            response.raise_for_status()
            kernel_info = await response.json()

        return t.cast("KernelState", kernel_info["execution_state"])

    async def busy(self) -> bool:
        """Check if the kernel is busy executing code."""
        return await self.get_kernel_state() == "busy"

    async def interrupt(self) -> None:
        """Interrupt the kernel."""
        if not self._kernel_id:
            return

        async with (
            aiohttp.ClientSession() as session,
            session.post(
                f"{self._base_url}/api/kernels/{self._kernel_id}/interrupt",
                params={"token": self._token},
            ) as response,
        ):
            response.raise_for_status()

        logger.debug(f"Kernel {self._kernel_id} interrupted")

    @tool_method()
    async def restart(self) -> None:
        """Restart the kernel."""
        if not self._kernel_id:
            return

        async with (
            aiohttp.ClientSession() as session,
            session.post(
                f"{self._base_url}/api/kernels/{self._kernel_id}/restart",
                params={"token": self._token},
            ) as response,
        ):
            response.raise_for_status()

        logger.debug(f"Kernel {self._kernel_id} restarted")
