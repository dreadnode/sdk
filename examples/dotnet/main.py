import asyncio
import io
import random
import typing as t
import zipfile
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent

import aiohttp
import cyclopts
import duckdb
import litellm
import rigging as rg
from loguru import logger
from rich.console import Console
from rich.markdown import Markdown

import dreadnode as dn

console = Console()

from dotnet.reversing import DotnetReversing

if t.TYPE_CHECKING:
    from loguru import Record as LogRecord

# CLI

app = cyclopts.App()


@cyclopts.Parameter(name="*", group="args")
@dataclass
class Args:
    model: str
    """Model to use for inference"""
    path: str
    """Directory of binaries to analyze or other supported identifier"""
    nuget: bool = False
    """Treat the path as a NuGet package id or path to a list of packages"""
    task: str = "Find only critical vulnerabilities"
    """Task presented to the agent"""
    max_steps: int = 25
    """Maximum number of iterations per agent"""
    concurrency: int = 3
    """Maximum number of agents to run in parallel at any given time"""
    log_level: str = "INFO"
    """Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"""


@cyclopts.Parameter(name="*", group="dreadnode")
@dataclass
class DreadnodeArgs:
    server: str | None = None
    """Dreadnode server URL"""
    token: str | None = None
    """Dreadnode API token"""
    project: str = "dotnet-reversing-final"
    """Project name"""
    console: t.Annotated[bool, cyclopts.Parameter(negative=False)] = False
    """Show span information in the console"""


def log_formatter(record: "LogRecord") -> str:
    return "".join(
        (
            "<green>{time:HH:mm:ss.SSS}</green> | ",
            "<dim>{extra[prefix]}</dim> " if record["extra"].get("prefix") else "",
            "<level>{message}</level>\n",
        ),
    )


T = t.TypeVar("T")


async def enforce_concurrency(coros: t.Sequence[t.Awaitable[T]], limit: int) -> list[T]:
    semaphore = asyncio.Semaphore(limit)

    async def run_coroutine_with_semaphore(
        coro: t.Awaitable[T],
    ) -> T:
        async with semaphore:
            return await coro

    return await asyncio.gather(
        *(run_coroutine_with_semaphore(coro) for coro in coros),
    )


@dn.task(name="Save POC", log_inputs=False, log_output=False)
async def detailed_exploit_instructions(
    poc: str,
) -> str:
    """
    Save a Proof of Concept (PoC) for a vulnerability.
    """

    import uuid

    output_dir = Path("pocs")
    output_dir.mkdir(exist_ok=True)

    file_path = output_dir / f"{uuid.uuid4().hex[:8]}.md"
    file_path.write_text(poc)

    console.print(Markdown(f"### PoC saved to [link={file_path}]{file_path}[/link]\n\n{poc}"))

    dn.log_output("poc", poc)
    dn.tag("poc", to="run")
    dn.log_metric("num_pocs", 1, mode="count")

    return "Successfully saved PoC"


@dn.task(name="Verify findings", log_inputs=False, log_output=False)
async def verify(
    findings: t.Annotated[str, "Detailed analysis of the vulnerability"],
    vulnerable_code: t.Annotated[str, "All the supporting code to verify the vulnerability"],
) -> str:
    prompt = dedent(f"""\
        You are an expert exploit dev of .NET assemblies. Your task is to review the findings and PoCs. If they are valid please provide a detailed harness to exploit the vulnerability that can be used to verify the vulnerability. Please write it in in PowerShell. Please make sure everything is properly imported. If a server is involved, using example.com is not helpful, please provide the exact value.

        You do not have admin access, and you do not have credentials unless explicitly provided.

        Save your work to disk using the `detailed_exploit_instructions` tool.

        DO NOT write fixes or suggestions.
        DO NOT speculate, or make assumptions. Don't say could, might, maybe, or similar.
        DO NOT report encyption issues.
        DO NOT mock or pretend.
        MITM is not in scope.
        The year is 2025.

        <findings>
        {findings}
        </findings>

        <vulnerable_code>
        {vulnerable_code}
        </vulnerable_code>


        If you are confused, please ask for clarification. If you are unsure about the exploitability of a vulnerability, please provide a detailed explanation of why it is not exploitable.

        """)

    models = [
        "gpt-4o",
        "claude-3-7-sonnet-latest",
        "claude-3-5-haiku-latest",
        "gpt-4.1",
        "groq/meta-llama/llama-4-maverick-17b-128e-instruct",
        "groq/moonshotai/kimi-k2-instruct",
        "o4-mini-2025-04-16",
    ]

    _pipeline = await (
        rg.get_generator(random.choice(models))
        .chat(prompt)
        .catch(
            *litellm.exceptions.LITELLM_EXCEPTION_TYPES,
            on_failed="include",
        )
        .using(detailed_exploit_instructions, finish_task, max_depth=50)
        .run()
    )

    console.print(Markdown(_pipeline.last.content))

    return _pipeline.last.content


@dn.task(name="Report finding", log_inputs=False, log_output=False)
async def report_finding(file: str, method: str, criticality: str, content: str) -> str:
    """
    Report a finding regarding areas or interest or vulnerabilities.

    for criticality, use:
    - "critical"
    - "high"
    - "medium"
    - "low"
    - "info"
    """
    logger.success(f"Reporting finding for {file} ({method}) [{criticality}]:")
    dn.log_output(
        "finding",
        {
            "file": file,
            "method": method,
            "content": content,
            "criticality": criticality,
        },
    )
    dn.log_metric("num_reports", 1, mode="count", to="run")
    dn.tag(criticality)
    return "Reported"


@dn.task(name="Report auth", log_inputs=True, log_output=False)
async def report_auth(
    auth_material: t.Annotated[str, "The Markdown details or code that uses the auth material"],
) -> str:
    """
    Report authentication material such as hardcoded keys, tokens, or passwords.
    """

    import uuid

    output_dir = Path("auth")
    output_dir.mkdir(exist_ok=True)

    file_path = output_dir / f"{uuid.uuid4().hex[:8]}.md"
    file_path.write_text(auth_material)
    console.print(
        Markdown(f"### Auth material in {file_path}\n\n{auth_material}"),
    )
    dn.log_output("auth_material", {"script": auth_material})
    dn.log_metric("auth_material", 1, mode="count", to="run")
    dn.log_param("auth_material", auth_material, to="run")
    dn.tag("creds", to="run")

    return "Auth material reported"


@dn.task(name="Finish task", log_output=False)
async def finish_task(success: bool, markdown_summary: str) -> None:
    """
    Mark your task as complete with a success/failure status and markdown summary.
    """
    dn.log_metric("task_success", success)
    if success:
        dn.tag("success", to="run")

    log_func = logger.success if success else logger.warning
    log_func(f"Agent finished the task (success={success}): {markdown_summary}")

    dn.log_metric("task_success", success, to="run")
    dn.log_output("task_summary", markdown_summary, to="run")


@dn.task(name="Download NuGet package")
async def download_nuget_package(package: str) -> Path:
    """
    Download a NuGet package and return the path to the package.
    """

    package = package.lower()
    logger.info(f"Downloading NuGet package {package}...")

    async with aiohttp.ClientSession() as client:
        # Get the versions
        async with client.get(
            f"https://api.nuget.org/v3-flatcontainer/{package}/index.json",
        ) as response:
            if response.status != 200:  # noqa: PLR2004
                raise RuntimeError(f"Failed to get package {package} from NuGet")

            data = await response.json()
            versions = data["versions"]
            latest_version = versions[-1]
            logger.info(f" |- Latest version is {latest_version}")

        # Download the nupkg and extract it
        async with client.get(
            f"https://api.nuget.org/v3-flatcontainer/{package}/{latest_version}/{package}.{latest_version}.nupkg",
        ) as response:
            if response.status != 200:  # noqa: PLR2004
                raise RuntimeError(f"Failed to download package {package} from NuGet")

            output_dir = Path(f".nuget/{package}_{latest_version}")
            output_dir.mkdir(parents=True, exist_ok=True)

            data = await response.read()
            with io.BytesIO(data) as buffer, zipfile.ZipFile(buffer) as zip_file:
                zip_file.extractall(output_dir)

            logger.info(f" |- Extracted to {output_dir}")

    return output_dir


async def agent(args: Args) -> None:
    with (
        dn.run(),
        dn.task_span("Agent"),
        logger.contextualize(prefix=str(args.path)),
    ):
        dn.log_params(
            model=args.model,
            path=str(args.path),
            nuget=args.nuget,
            max_steps=args.max_steps,
        )

        path = await download_nuget_package(args.path) if args.nuget else Path(args.path)
        reversing = DotnetReversing.from_path(path)

        logger.info(f"Analyzing {path}")

        dn.log_inputs(
            binaries=[str(b) for b in reversing.binaries],
        )

        binary_list = "\n".join(reversing.binaries)

        prompt = dedent(f"""\
        You are an expert dotnet reverse engineer with decades of experience. Your task is to analyze the provided binaries and identify high impact vulnerabilities. You care most about exploitable bugs from a remote perspective. It is okay to review the code multiple times, or search for other files in the package to confirm vulnerabilities. Here are your steps to complete the task:

        1. Find REAL vulnerabilities using the reversing tools to explore assemblies and config files most likely to contain the following vulnerabilities:
            - Local code execution
            - Privileged file access
            - Remote code execution.
            - Live hardcoded Keys, tokens, or password or secrets that are CURRENTLY present in the code.
            - Web-related vulnerabilities
            - Internal API abuse
            - Logic flaws

        2. For each finding, write an extremely complete Proof of Concept (PoC) or harness using the `verify` tool. Provide the exact entry point, arguments, and call tree from entry point to execution. The next agent will use this to build a harness to exploit the vulnerability. You MUST provide a entry point that is reachable.
        3. If there is an executable that can be run, decompile it first and work from there to the vulnerability.
        4. If you find hardcoded credentials, report them using the `report_auth` tool and finish the task with `finish_task` tool. Be sure to provide the exact credentials found, and any details about how they are used with a simple Python script to test the credentials.

        Here are the binaries you need to analyze:

        <files>
        {binary_list}
        </files>

        DO NOT write fixes or suggestions.
        DO NOT speculate, or make assumptions. Don't say could, might, maybe, or similar.
        DO NOT report encyption issues.
        DO NOT mock or pretend.
        You do not have admin access, and you do not have credentials unless explicitly provided.
        You do not have access to dump memory, or run the code.
        The year is 2025.

        """)

        dn.log_input("binaries", binary_list, to="run")

        generator = rg.get_generator(args.model)
        chat = (
            await generator.chat(prompt)
            .catch(
                *litellm.exceptions.LITELLM_EXCEPTION_TYPES,
                on_failed="include",
            )
            .using(
                reversing.tools,
                verify,
                report_auth,
                finish_task,
                max_depth=100,
            )
            .cache("latest")
            .run()
        )

        if chat.failed and chat.error:
            if isinstance(chat.error, rg.error.MaxDepthError):
                logger.warning(f"Max steps reached ({args.max_steps})")
                dn.log_metric("max_steps_reached", 1)
                dn.log_output("task_summary", f"Max steps ({args.max_steps}) reached", to="run")
            else:
                logger.warning(f"Failed with {chat.error}")
                dn.log_metric("inference_failed", 1)
                dn.log_output("task_summary", f"Inference failed with {chat.error}", to="run")

        elif chat.last.role == "assistant":
            dn.log_output("last_message", chat.last.content)
            console.print(Markdown(str(chat.last)))  # logger.info(str(chat.last))


@app.default
async def main(*, args: Args, dn_args: DreadnodeArgs | None = None) -> None:
    # logger.remove()
    # logger.add(sys.stderr, format=log_formatter, level=args.log_level)
    # logger.enable("rigging")

    dn_args = dn_args or DreadnodeArgs()
    dn.configure(
        server=dn_args.server,
        token=dn_args.token,
        project=dn_args.project,
        console=dn_args.console,
    )

    models = [
        "gpt-4o",
        "claude-3-7-sonnet-latest",
        "claude-3-5-haiku-latest",
        "gpt-4.1",
        "groq/meta-llama/llama-4-maverick-17b-128e-instruct",
        "groq/moonshotai/kimi-k2-instruct",
        "o4-mini-2025-04-16",
    ]

    # Query for packages matching our criteria:
    # - No prerelease versions
    # - Listed packages
    # - No project URL (missing source repo info)
    # - No license expression or URL

    db_file = args.path
    logger.info(f"Loading packages from {db_file}")
    with duckdb.connect(db_file) as conn:
        query = """
SELECT
    p.package_name
FROM packages p
JOIN versions v ON p.latest_version_id = v.id
WHERE
    v.is_prerelease = FALSE
    AND v.listed = TRUE
    AND (v.project_url IS NULL OR v.project_url = '')
    AND (v.license_url IS NULL OR v.license_url = '')
    AND (v.license_expression IS NULL OR v.license_expression = '')
ORDER BY v.last_edited ASC, v.package_size DESC
LIMIT 10000
        """

        packages = conn.execute(query).fetchall()
        logger.info(f"Found {len(packages)} packages to analyze")

    packages = random.sample(packages, 500)

    tasks = [
        agent(
            Args(
                model=random.choice(models),
                path=package[0],
                nuget=True,
                task=args.task,
                max_steps=args.max_steps,
                concurrency=args.concurrency,
            ),
        )
        for package in packages
    ]
    await enforce_concurrency(tasks, args.concurrency)

    logger.success("Done.")


if __name__ == "__main__":
    app()
