# import typing as t

# from loguru import logger

# from dreadnode.agent.tools import tool


# @tool
# async def powerview(
#     self,
#     command: t.Annotated[
#         str,
#         "Powerview command line arguments to supply to the powershell instance and execute.",
#     ],
#     credential_user: t.Annotated[
#         str | None, "username to execute Powerview commands as specified user"
#     ] = None,
#     credential_password: t.Annotated[
#         str | None, "password to execute Powerview commands as specified user"
#     ] = None,
#     domain: t.Annotated[
#         str | None, "domain to execute Powerview commands as specified user"
#     ] = None,
# ) -> str:
#     """
#     Imports PowerView into Powershell (for use) and then executes the supplied command line arguments in current Powershell instance.

#     """

#     powerview_script_filename = "PowerView.ps1"
#     upload_result = await self._client.upload_file_to_mythic_server(
#         filename=SCRIPTS_DIR / powerview_script_filename,
#         reupload=False,
#     )
#     if upload_result["file_id"] is None:
#         return f"Error running 'powerview' command.\n\n Attempting to upload {powerview_script_filename} file to Mythic led to unknown error."
#     logger.info(f"Uploaded {powerview_script_filename} to Mythic.")

#     pi_result = await self.powershell_import(filename=upload_result["filename"])
#     if "will now be imported in PowerShell commands" not in pi_result:
#         return f"Error running [COMMAND] 'powershell_import': - {pi_result}."

#     if all([credential_user, credential_password, domain]):
#         powerview_cmd = f"{command} -Credential (New-Object -TypeName 'System.Management.Automation.PSCredential' -ArgumentList '{domain}\\{credential_user}', (ConvertTo-SecureString -String '{credential_password}' -AsPlainText -Force))"

#     return await self.powerpick(command=powerview_cmd)
