import shlex
import typing as t

import paramiko
from pydantic import BaseModel, Field, PrivateAttr

from dreadnode.agent.tools import Toolset, tool_method


def _q(s: str) -> str:
    return shlex.quote(s)


class SSHConn(BaseModel):
    host: t.Annotated[str, "Remote host or IP"]
    user: t.Annotated[str, "SSH username"]
    password: t.Annotated[str | None, "SSH password (omit if using key)"] = None
    key_path: t.Annotated[str | None, "Path to private key (PEM/OpenSSH)"] = None
    port: t.Annotated[int, "SSH port"] = 22

    @property
    def key(self) -> str:
        return f"{self.user}@{self.host}:{self.port}"


class SSHTools(Toolset):
    profiles: dict[str, SSHConn] = Field(default_factory=dict, description="Saved SSH profiles")
    default_profile: str | None = Field(default=None, description="Default profile name")
    _clients: dict[str, paramiko.SSHClient] = PrivateAttr(default_factory=dict)

    # --- internals ---
    def _resolve_conn(self, conn: SSHConn | None, profile: str | None) -> SSHConn:
        if conn is not None:
            return conn
        name = profile or self.default_profile
        if not name or name not in self.profiles:
            raise ValueError("Provide `conn` or an existing `profile` (or set `default_profile`).")
        return self.profiles[name]

    def _client(self, c: SSHConn) -> paramiko.SSHClient:
        if c.key not in self._clients:
            cli = paramiko.SSHClient()
            cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            cli.connect(
                c.host,
                port=c.port,
                username=c.user,
                password=c.password,
                key_filename=c.key_path,
                look_for_keys=not bool(c.key_path),
                allow_agent=True,
                timeout=15,
            )
            self._clients[c.key] = cli
        return self._clients[c.key]

    def _run(
        self, cli: paramiko.SSHClient, cmd: str, timeout: int | None = None
    ) -> tuple[int, str, str]:
        stdin, stdout, stderr = cli.exec_command(cmd, timeout=timeout)
        out = stdout.read().decode(errors="replace").strip()
        err = stderr.read().decode(errors="replace").strip()
        rc = stdout.channel.recv_exit_status()
        return rc, out, err

    # --- meta: let the LLM set a profile once ---
    @tool_method(
        name="ssh.configure",
        description="Save a connection under a profile; optionally set as default.",
        catch=True,
    )
    def configure(
        self,
        profile: t.Annotated[str, "Profile name to save"],
        conn: t.Annotated[SSHConn, "Connection settings to store"],
        *,
        make_default: t.Annotated[bool, "Also set as default profile?"] = True,
    ) -> dict[str, t.Any]:
        self.profiles[profile] = conn
        if make_default:
            self.default_profile = profile
        return {"success": True, "profiles": list(self.profiles), "default": self.default_profile}

    # --- exec: takes either conn or profile ---
    @tool_method(name="ssh.exec", description="Run a shell command via SSH.", catch=True)
    def exec(
        self,
        command: t.Annotated[str, "Shell command to execute remotely"],
        conn: t.Annotated[SSHConn | None, "Inline connection (optional)"] = None,
        profile: t.Annotated[str | None, "Use a saved profile name (optional)"] = None,
    ) -> dict:
        c = self._resolve_conn(conn, profile)
        cli = self._client(c)
        rc, out, err = self._run(cli, command)
        return {"success": rc == 0, "code": rc, "output": out, "error": err or None}

    @tool_method(
        name="tmux.create", description="Create a tmux session if not present.", catch=True
    )
    def tmux_create(
        self, session: str, conn: SSHConn | None = None, profile: str | None = None
    ) -> dict:
        c = self._resolve_conn(conn, profile)
        return self.exec(
            f"tmux has-session -t {_q(session)} || tmux new-session -d -s {_q(session)}", conn=c
        )

    @tool_method(name="tmux.send", description="Send one line to tmux session.", catch=True)
    def tmux_send(
        self, session: str, line: str, conn: SSHConn | None = None, profile: str | None = None
    ) -> dict:
        c = self._resolve_conn(conn, profile)
        s, line_quoted = _q(session), _q(line)
        return self.exec(
            f"tmux send-keys -t {s} -l {line_quoted} \\; tmux send-keys -t {s} Enter", conn=c
        )

    @tool_method(name="tmux.capture", description="Capture pane text from session.", catch=True)
    def tmux_capture(
        self, session: str, conn: SSHConn | None = None, profile: str | None = None
    ) -> dict:
        c = self._resolve_conn(conn, profile)
        return self.exec(f"tmux capture-pane -pt {_q(session)}", conn=c)
