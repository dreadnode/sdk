import os
import typing as t

import aiohttp
from loguru import logger
from neo4j import GraphDatabase
from rich.console import Console

from dreadnode.agent.tools import Toolset, tool_method

# BloodHound & Neo4j connection details
BLOODHOUND_URL = os.getenv("BLOODHOUND_URL", "localhost:8080")
BLOODHOUND_USERNAME = os.getenv("BLOODHOUND_USERNAME", "admin")
BLOODHOUND_PASSWORD = os.getenv("BLOODHOUND_PASSWORD", "bloodhound")
BLOODHOUND_NEO4J_URL = os.getenv("BLOODHOUND_NEO4J_URL", "bolt://localhost:7687")
BLOODHOUND_NEO4J_USERNAME = os.getenv("BLOODHOUND_NEO4J_USERNAME", "neo4j")
BLOODHOUND_NEO4J_PASSWORD = os.getenv("BLOODHOUND_NEO4J_PASSWORD", "bloodhoundcommunityedition")

console = Console()


class Bloodhound(Toolset):
    """Agent Tool API for BloodHound Server"""

    def __init__(
        self,
        url: str = BLOODHOUND_URL,
        username: str = BLOODHOUND_USERNAME,
        password: str = BLOODHOUND_PASSWORD,
        neo4j_url: str = BLOODHOUND_NEO4J_URL,
        neo4j_username: str = BLOODHOUND_NEO4J_USERNAME,
        neo4j_password: str = BLOODHOUND_NEO4J_PASSWORD,
    ):
        self.config = {
            url: url,
            username: username,
            password: password,
            neo4j_url: neo4j_url,
            neo4j_username: neo4j_username,
            neo4j_password: neo4j_password,
        }

    async def initialize(self) -> None:
        """initialize connection to BloodHound server"""

        self._graph_driver = GraphDatabase.driver(
            self.config["neo4j_url"],
            auth=(self.config["neo4j_username"], self.config["neo4j_password"]),
            encrypted=False,
        )

        if await self._api_authenticate() is None:
            raise Warning("Could not authenticate to Bloodhound REST API")

    async def _api_authenticate(self) -> None:
        """authenticate to Bloodhound API and get access token to use for REST API requests"""

        url = f"http://{self.config['url']}/api/v2/login"
        auth_data = {
            "login_method": "secret",
            "username": self.config["username"],
            "secret": self.config["password"],
        }
        auth_token = None
        async with (
            aiohttp.ClientSession() as session,
            session.post(url=url, json=auth_data) as resp,
        ):
            auth_token = await resp.json()

        if auth_token is None or auth_token.get("data", None) is None:
            logger.error("Authentication to Bloodhound REST API failed")
            return

        self._api_auth_token = auth_token["data"]

    async def query_bloodhound(self, query: str) -> dict[str, t.Any]:
        databases = ["neo4j", "bloodhound"]
        last_error = None

        for db in databases:
            try:
                with self._graph_driver.session(database=db) as session:
                    result = session.run(query)
                    data = [record.data() for record in result]
                    logger.info(f"Query successful on database '{db}'")
                    return {"success": True, "data": data}
            except Exception as e:
                last_error = e
                logger.debug(f"Query failed on database '{db}': {e!s}")
                continue

        logger.error(f"Query failed on all databases. Last error: {last_error!s}")
        return {"success": False, "error": str(last_error)}

    # Domain Information
    @tool_method()
    async def find_all_domain_admins(self) -> dict[str, t.Any]:
        query = """
        MATCH p = (t:Group)<-[:MemberOf*1..]-(a)
        WHERE (a:User or a:Computer) and t.objectid ENDS WITH '-512'
        RETURN p
        LIMIT 1000
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def map_domain_trusts(self) -> dict[str, t.Any]:
        query = """
        MATCH p = (:Domain)-[:TrustedBy]->(:Domain)
        RETURN p
        LIMIT 1000
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_tier_zero_locations(self) -> dict[str, t.Any]:
        query = """
        MATCH p = (t:Base)<-[:Contains*1..]-(:Domain)
        WHERE t.highvalue = true
        RETURN p
        LIMIT 1000
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def map_ou_structure(self) -> dict[str, t.Any]:
        query = """
        MATCH p = (:Domain)-[:Contains*1..]->(:OU)
        RETURN p
        LIMIT 1000
        """
        return await self.query_bloodhound(query)

    # Dangerous Privileges
    @tool_method()
    async def find_dcsync_privileges(self) -> dict[str, t.Any]:
        query = """
        MATCH p=(:Base)-[:DCSync|AllExtendedRights|GenericAll]->(:Domain)
        RETURN p
        LIMIT 1000
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_foreign_group_memberships(self) -> dict[str, t.Any]:
        query = """
        MATCH p=(s:Base)-[:MemberOf]->(t:Group)
        WHERE s.domainsid<>t.domainsid
        RETURN p
        LIMIT 1000
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_domain_users_local_admins(self) -> dict[str, t.Any]:
        query = """
        MATCH p=(s:Group)-[:AdminTo]->(:Computer)
        WHERE s.objectid ENDS WITH '-513'
        RETURN p
        LIMIT 1000
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_domain_users_laps_readers(self) -> dict[str, t.Any]:
        query = """
        MATCH p=(s:Group)-[:AllExtendedRights|ReadLAPSPassword]->(:Computer)
        WHERE s.objectid ENDS WITH '-513'
        RETURN p
        LIMIT 1000
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_domain_users_high_value_paths(self) -> dict[str, t.Any]:
        query = """
        MATCH p=shortestPath((s:Group)-[r*1..]->(t))
        WHERE t.highvalue = true AND s.objectid ENDS WITH '-513' AND s<>t
        RETURN p
        LIMIT 1000
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_domain_users_workstation_rdp(self) -> dict[str, t.Any]:
        query = """
        MATCH p=(s:Group)-[:CanRDP]->(t:Computer)
        WHERE s.objectid ENDS WITH '-513' AND NOT toUpper(t.operatingsystem) CONTAINS 'SERVER'
        RETURN p
        LIMIT 1000
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_domain_users_server_rdp(self) -> dict[str, t.Any]:
        query = """
        MATCH p=(s:Group)-[:CanRDP]->(t:Computer)
        WHERE s.objectid ENDS WITH '-513' AND toUpper(t.operatingsystem) CONTAINS 'SERVER'
        RETURN p
        LIMIT 1000
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_domain_users_privileges(self) -> dict[str, t.Any]:
        query = """
        MATCH p=(s:Group)-[r]->(:Base)
        WHERE s.objectid ENDS WITH '-513'
        RETURN p
        LIMIT 1000
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_domain_admin_non_dc_logons(self) -> dict[str, t.Any]:
        query = """
        MATCH (s)-[:MemberOf*0..]->(g:Group)
        WHERE g.objectid ENDS WITH '-516'
        WITH COLLECT(s) AS exclude
        MATCH p = (c:Computer)-[:HasSession]->(:User)-[:MemberOf*1..]->(g:Group)
        WHERE g.objectid ENDS WITH '-512' AND NOT c IN exclude
        RETURN p
        LIMIT 1000
        """
        return await self.query_bloodhound(query)

    # Kerberos Interaction
    @tool_method()
    async def find_kerberoastable_tier_zero(self) -> dict[str, t.Any]:
        query = """
        MATCH (u:User)
        WHERE u.hasspn=true
        AND u.enabled = true
        AND NOT u.objectid ENDS WITH '-502'
        AND NOT u.gmsa = true
        AND NOT u.msa = true
        AND u.highvalue = true
        RETURN u
        LIMIT 100
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_all_kerberoastable_users(self) -> dict[str, t.Any]:
        query = """
        MATCH (u:User)
        WHERE u.hasspn=true
        AND u.enabled = true
        AND NOT u.objectid ENDS WITH '-502'
        AND NOT u.gmsa = true
        AND NOT u.msa = true
        RETURN u
        LIMIT 100
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_kerberoastable_most_admin(self) -> dict[str, t.Any]:
        query = """
        MATCH (u:User)
        WHERE u.hasspn = true
        AND u.enabled = true
        AND NOT u.objectid ENDS WITH '-502'
        AND NOT u.gmsa = true
        AND NOT u.msa = true
        MATCH (u)-[:MemberOf|AdminTo*1..]->(c:Computer)
        WITH DISTINCT u, COUNT(c) AS adminCount
        RETURN u
        ORDER BY adminCount DESC
        LIMIT 100
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_asreproast_users(self) -> dict[str, t.Any]:
        query = """
        MATCH (u:User)
        WHERE u.dontreqpreauth = true
        AND u.enabled = true
        RETURN u
        LIMIT 100
        """
        return await self.query_bloodhound(query)

    # Shortest Paths
    @tool_method()
    async def find_shortest_paths_unconstrained_delegation(self) -> dict[str, t.Any]:
        query = """
        MATCH p=shortestPath((s)-[r*1..]->(t:Computer))
        WHERE t.unconstraineddelegation = true AND s<>t
        RETURN p
        LIMIT 1000
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_paths_from_kerberoastable_to_da(self) -> dict[str, t.Any]:
        query = """
        MATCH p=shortestPath((s:User)-[r*1..]->(t:Group))
        WHERE s.hasspn=true
        AND s.enabled = true
        AND NOT s.objectid ENDS WITH '-502'
        AND NOT s.gmsa = true
        AND NOT s.msa = true
        AND t.objectid ENDS WITH '-512'
        RETURN p
        LIMIT 1000
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_shortest_paths_to_tier_zero(self) -> dict[str, t.Any]:
        query = """
        MATCH p=shortestPath((s)-[r*1..]->(t))
        WHERE t.highvalue = true AND s<>t
        RETURN p
        LIMIT 1000
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_paths_from_domain_users_to_tier_zero(self) -> dict[str, t.Any]:
        query = """
        MATCH p=shortestPath((s:Group)-[r*1..]->(t))
        WHERE t.highvalue = true AND s.objectid ENDS WITH '-513' AND s<>t
        RETURN p
        LIMIT 1000
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_shortest_paths_to_domain_admins(self) -> dict[str, t.Any]:
        query = """
        MATCH p=shortestPath((t:Group)<-[r*1..]-(s:Base))
        WHERE t.objectid ENDS WITH '-512' AND s<>t
        RETURN p
        LIMIT 1000
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_paths_from_owned_objects(self) -> dict[str, t.Any]:
        query = """
        MATCH p=shortestPath((s:Base)-[r*1..]->(t:Base))
        WHERE s.owned = true AND s<>t
        RETURN p
        LIMIT 1000
        """
        return await self.query_bloodhound(query)

    # Active Directory Certificate Services
    @tool_method()
    async def find_pki_hierarchy(self) -> dict[str, t.Any]:
        query = """
        MATCH p=()-[:HostsCAService|IssuedSignedBy|EnterpriseCAFor|RootCAFor|TrustedForNTAuth|NTAuthStoreFor*..]->(:Domain)
        RETURN p
        LIMIT 1000
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_public_key_services(self) -> dict[str, t.Any]:
        query = """
        MATCH p = (c:Container)-[:Contains*..]->(:Base)
        WHERE c.distinguishedname starts with 'CN=PUBLIC KEY SERVICES,CN=SERVICES,CN=CONFIGURATION,DC='
        RETURN p
        LIMIT 1000
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_certificate_enrollment_rights(self) -> dict[str, t.Any]:
        query = """
        MATCH p = (:Base)-[:Enroll|GenericAll|AllExtendedRights]->(:CertTemplate)-[:PublishedTo]->(:EnterpriseCA)
        RETURN p
        LIMIT 1000
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_esc1_vulnerable_templates(self) -> dict[str, t.Any]:
        query = """
        MATCH p = (:Base)-[:Enroll|GenericAll|AllExtendedRights]->(ct:CertTemplate)-[:PublishedTo]->(:EnterpriseCA)
        WHERE ct.enrolleesuppliessubject = True
        AND ct.authenticationenabled = True
        AND ct.requiresmanagerapproval = False
        AND (ct.authorizedsignatures = 0 OR ct.schemaversion = 1)
        RETURN p
        LIMIT 1000
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_esc2_vulnerable_templates(self) -> dict[str, t.Any]:
        query = """
        MATCH p = (:Base)-[:Enroll|GenericAll|AllExtendedRights]->(c:CertTemplate)-[:PublishedTo]->(:EnterpriseCA)
        WHERE c.requiresmanagerapproval = false
        AND (c.effectiveekus = [''] OR '2.5.29.37.0' IN c.effectiveekus)
        AND (c.authorizedsignatures = 0 OR c.schemaversion = 1)
        RETURN p
        LIMIT 1000
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_enrollment_agent_templates(self) -> dict[str, t.Any]:
        query = """
        MATCH p = (:Base)-[:Enroll|GenericAll|AllExtendedRights]->(ct:CertTemplate)-[:PublishedTo]->(:EnterpriseCA)
        WHERE '1.3.6.1.4.1.311.20.2.1' IN ct.effectiveekus
        OR '2.5.29.37.0' IN ct.effectiveekus
        OR SIZE(ct.effectiveekus) = 0
        RETURN p
        LIMIT 1000
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_dcs_weak_certificate_binding(self) -> dict[str, t.Any]:
        query = """
        MATCH p = (s:Computer)-[:DCFor]->(:Domain)
        WHERE s.strongcertificatebindingenforcementraw = 0 OR s.strongcertificatebindingenforcementraw = 1
        RETURN p
        LIMIT 1000
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_inactive_tier_zero_principals(self) -> dict[str, t.Any]:
        query = """
        WITH 60 as inactive_days
        MATCH (n:Base)
        WHERE n.highvalue = true
        AND n.enabled = true
        AND n.lastlogontimestamp < (datetime().epochseconds - (inactive_days * 86400))
        AND n.lastlogon < (datetime().epochseconds - (inactive_days * 86400))
        AND n.whencreated < (datetime().epochseconds - (inactive_days * 86400))
        AND NOT n.name STARTS WITH 'AZUREADKERBEROS.'
        AND NOT n.objectid ENDS WITH '-500'
        AND NOT n.name STARTS WITH 'AZUREADSSOACC.'
        RETURN n
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_tier_zero_without_smartcard(self) -> dict[str, t.Any]:
        query = """
        MATCH (u:User)
        WHERE u.highvalue = true
        AND u.enabled = true
        AND u.smartcardrequired = false
        AND NOT u.name STARTS WITH 'MSOL_'
        AND NOT u.name STARTS WITH 'PROVAGENTGMSA'
        AND NOT u.name STARTS WITH 'ADSYNCMSA_'
        RETURN u
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_domains_with_machine_quota(self) -> dict[str, t.Any]:
        query = """
        MATCH (d:Domain)
        WHERE d.machineaccountquota > 0
        RETURN d
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_smartcard_dont_expire_domains(self) -> dict[str, t.Any]:
        query = """
        MATCH (s:Domain)-[:Contains*1..]->(t:Base)
        WHERE s.expirepasswordsonsmartcardonlyaccounts = false
        AND t.enabled = true
        AND t.smartcardrequired = true
        RETURN s
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_two_way_forest_trust_delegation(self) -> dict[str, t.Any]:
        query = """
        MATCH p=(n:Domain)-[r:TrustedBy]->(m:Domain)
        WHERE (m)-[:TrustedBy]->(n)
        AND r.trusttype = 'Forest'
        AND r.tgtdelegationenabled = true
        RETURN p
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_unsupported_operating_systems(self) -> dict[str, t.Any]:
        query = """
        MATCH (c:Computer)
        WHERE c.operatingsystem =~ '(?i).*Windows.* (2000|2003|2008|2012|xp|vista|7|8|me|nt).*'
        RETURN c
        LIMIT 100
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_users_with_no_password_required(self) -> dict[str, t.Any]:
        query = """
        MATCH (u:User)
        WHERE u.passwordnotreqd = true
        RETURN u
        LIMIT 100
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_users_password_not_rotated(self) -> dict[str, t.Any]:
        query = """
        WITH 365 as days_since_change
        MATCH (u:User)
        WHERE u.pwdlastset < (datetime().epochseconds - (days_since_change * 86400))
        AND NOT u.pwdlastset IN [-1.0, 0.0]
        RETURN u
        LIMIT 100
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_nested_tier_zero_groups(self) -> dict[str, t.Any]:
        query = """
        MATCH p=(t:Group)<-[:MemberOf*..]-(s:Group)
        WHERE t.highvalue = true
        AND NOT s.objectid ENDS WITH '-512'
        AND NOT s.objectid ENDS WITH '-519'
        RETURN p
        LIMIT 1000
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_disabled_tier_zero_principals(self) -> dict[str, t.Any]:
        query = """
        MATCH (n:Base)
        WHERE n.highvalue = true
        AND n.enabled = false
        AND NOT n.objectid ENDS WITH '-502'
        AND NOT n.objectid ENDS WITH '-500'
        RETURN n
        LIMIT 100
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_principals_reversible_encryption(self) -> dict[str, t.Any]:
        query = """
        MATCH (n:Base)
        WHERE n.encryptedtextpwdallowed = true
        RETURN n
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_principals_des_only_kerberos(self) -> dict[str, t.Any]:
        query = """
        MATCH (n:Base)
        WHERE n.enabled = true
        AND n.usedeskeyonly = true
        RETURN n
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_principals_weak_kerberos_encryption(self) -> dict[str, t.Any]:
        query = """
        MATCH (u:Base)
        WHERE 'DES-CBC-CRC' IN u.supportedencryptiontypes
        OR 'DES-CBC-MD5' IN u.supportedencryptiontypes
        OR 'RC4-HMAC-MD5' IN u.supportedencryptiontypes
        RETURN u
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_tier_zero_non_expiring_passwords(self) -> dict[str, t.Any]:
        query = """
        MATCH (u:User)
        WHERE u.enabled = true
        AND u.pwdneverexpires = true
        AND u.highvalue = true
        RETURN u
        LIMIT 100
        """
        return await self.query_bloodhound(query)

    # NTLM Relay Attacks
    @tool_method()
    async def find_ntlm_relay_edges(self) -> dict[str, t.Any]:
        query = """
        MATCH p = (n:Base)-[:CoerceAndRelayNTLMToLDAP|CoerceAndRelayNTLMToLDAPS|CoerceAndRelayNTLMToADCS|CoerceAndRelayNTLMToSMB]->(:Base)
        RETURN p LIMIT 500
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_esc8_vulnerable_cas(self) -> dict[str, t.Any]:
        query = """
        MATCH (n:EnterpriseCA)
        WHERE n.hasvulnerableendpoint=true
        RETURN n
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_computers_outbound_ntlm_deny(self) -> dict[str, t.Any]:
        query = """
        MATCH (c:Computer)
        WHERE c.restrictoutboundntlm = True
        RETURN c LIMIT 1000
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_computers_in_protected_users(self) -> dict[str, t.Any]:
        query = """
        MATCH p = (:Base)-[:MemberOf*1..]->(g:Group)
        WHERE g.objectid ENDS WITH "-525"
        RETURN p LIMIT 1000
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_dcs_vulnerable_ntlm_relay(self) -> dict[str, t.Any]:
        query = """
        MATCH p = (dc:Computer)-[:DCFor]->(:Domain)
        WHERE (dc.ldapavailable = True AND dc.ldapsigning = False)
        OR (dc.ldapsavailable = True AND dc.ldapsepa = False)
        OR (dc.ldapavailable = True AND dc.ldapsavailable = True AND dc.ldapsigning = False and dc.ldapsepa = True)
        RETURN p
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_computers_webclient_running(self) -> dict[str, t.Any]:
        query = """
        MATCH (c:Computer)
        WHERE c.webclientrunning = True
        RETURN c LIMIT 1000
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_computers_no_smb_signing(self) -> dict[str, t.Any]:
        query = """
        MATCH (n:Computer)
        WHERE n.smbsigning = False
        RETURN n
        """
        return await self.query_bloodhound(query)

    # Azure - General
    @tool_method()
    async def find_global_administrators(self) -> dict[str, t.Any]:
        query = """
        MATCH p = (:AZBase)-[:AZGlobalAdmin*1..]->(:AZTenant)
        RETURN p
        LIMIT 1000
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_high_privileged_role_members(self) -> dict[str, t.Any]:
        query = """
        MATCH p=(t:AZRole)<-[:AZHasRole|AZMemberOf*1..2]-(:AZBase)
        WHERE t.name =~ '(?i)(Global Administrator|User Access Administrator|Privileged Role Administrator|Privileged Authentication Administrator|Partner Tier1 Support|Partner Tier2 Support)'
        RETURN p
        LIMIT 1000
        """
        return await self.query_bloodhound(query)

    # Azure - Shortest Paths
    @tool_method()
    async def find_paths_from_entra_to_tier_zero(self) -> dict[str, t.Any]:
        query = """
        MATCH p=shortestPath((s:AZUser)-[r*1..]->(t:AZBase))
        WHERE t.highvalue = true AND t.name =~ '(?i)(Global Administrator|User Access Administrator|Privileged Role Administrator|Privileged Authentication Administrator|Partner Tier1 Support|Partner Tier2 Support)' AND s<>t
        RETURN p
        LIMIT 1000
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_paths_to_privileged_roles(self) -> dict[str, t.Any]:
        query = """
        MATCH p=shortestPath((s:AZBase)-[r*1..]->(t:AZRole))
        WHERE t.name =~ '(?i)(Global Administrator|User Access Administrator|Privileged Role Administrator|Privileged Authentication Administrator|Partner Tier1 Support|Partner Tier2 Support)' AND s<>t
        RETURN p
        LIMIT 1000
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_paths_from_azure_apps_to_tier_zero(self) -> dict[str, t.Any]:
        query = """
        MATCH p=shortestPath((s:AZApp)-[r*1..]->(t:AZBase))
        WHERE t.highvalue = true AND s<>t
        RETURN p
        LIMIT 1000
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_paths_to_azure_subscriptions(self) -> dict[str, t.Any]:
        query = """
        MATCH p=shortestPath((s:AZBase)-[r*1..]->(t:AZSubscription))
        WHERE s<>t
        RETURN p
        LIMIT 1000
        """
        return await self.query_bloodhound(query)

    # Azure - Microsoft Graph
    @tool_method()
    async def find_service_principals_with_app_role_grant(self) -> dict[str, t.Any]:
        query = """
        MATCH p=(:AZServicePrincipal)-[:AZMGGrantAppRoles]->(:AZTenant)
        RETURN p
        LIMIT 1000
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_service_principals_with_graph_assignments(self) -> dict[str, t.Any]:
        query = """
        MATCH p=(:AZServicePrincipal)-[:AZMGAppRoleAssignment_ReadWrite_All|AZMGApplication_ReadWrite_All|AZMGDirectory_ReadWrite_All|AZMGGroupMember_ReadWrite_All|AZMGGroup_ReadWrite_All|AZMGRoleManagement_ReadWrite_Directory|AZMGServicePrincipalEndpoint_ReadWrite_All]->(:AZServicePrincipal)
        RETURN p
        LIMIT 1000
        """
        return await self.query_bloodhound(query)

    # Azure - Hygiene
    @tool_method()
    async def find_foreign_tier_zero_principals(self) -> dict[str, t.Any]:
        query = """
        MATCH (n:AZServicePrincipal)
        WHERE n.highvalue = true
        AND NOT toUpper(n.appownerorganizationid) = toUpper(n.tenantid)
        AND n.appownerorganizationid CONTAINS '-'
        RETURN n
        LIMIT 100
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_synced_tier_zero_principals(self) -> dict[str, t.Any]:
        query = """
        MATCH (ENTRA:AZBase)
        MATCH (AD:Base)
        WHERE ENTRA.onpremsyncenabled = true
        AND ENTRA.onpremid = AD.objectid
        AND AD.highvalue = true
        RETURN ENTRA
        LIMIT 100
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_external_tier_zero_users(self) -> dict[str, t.Any]:
        query = """
        MATCH (n:AZUser)
        WHERE n.highvalue = true
        AND n.name CONTAINS '#EXT#@'
        RETURN n
        LIMIT 100
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_disabled_azure_tier_zero_principals(self) -> dict[str, t.Any]:
        query = """
        MATCH (n:AZBase)
        WHERE n.highvalue = true
        AND n.enabled = false
        RETURN n
        LIMIT 100
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_devices_unsupported_os(self) -> dict[str, t.Any]:
        query = """
        MATCH (n:AZDevice)
        WHERE n.operatingsystem CONTAINS 'WINDOWS'
        AND n.operatingsystemversion =~ '(10.0.19044|10.0.22000|10.0.19043|10.0.19042|10.0.19041|10.0.18363|10.0.18362|10.0.17763|10.0.17134|10.0.16299|10.0.15063|10.0.14393|10.0.10586|10.0.10240|6.3.9600|6.2.9200|6.1.7601|6.0.6200|5.1.2600|6.0.6003|5.2.3790|5.0.2195).?.*'
        RETURN n
        LIMIT 100
        """
        return await self.query_bloodhound(query)

    # Azure - Cross Platform Attack Paths
    @tool_method()
    async def find_entra_users_in_domain_admins(self) -> dict[str, t.Any]:
        query = """
        MATCH p = (:AZUser)-[:SyncedToADUser]->(:User)-[:MemberOf]->(t:Group)
        WHERE t.objectid ENDS WITH '-512'
        RETURN p
        LIMIT 1000
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_onprem_users_owning_entra_objects(self) -> dict[str, t.Any]:
        query = """
        MATCH p = (:User)-[:SyncedToEntraUser]->(:AZUser)-[:AZOwns]->(:AZBase)
        RETURN p
        LIMIT 1000
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_onprem_users_in_entra_groups(self) -> dict[str, t.Any]:
        query = """
        MATCH p = (:User)-[:SyncedToEntraUser]->(:AZUser)-[:AZMemberOf]->(:AZGroup)
        RETURN p
        LIMIT 1000
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_templates_no_security_extension(self) -> dict[str, t.Any]:
        query = """
        MATCH p = (:Base)-[:Enroll|GenericAll|AllExtendedRights]->(ct:CertTemplate)-[:PublishedTo]->(:EnterpriseCA)
        WHERE ct.nosecurityextension = true
        RETURN p
        LIMIT 1000
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_templates_with_user_specified_san(self) -> dict[str, t.Any]:
        query = """
        MATCH p = (:Base)-[:Enroll|GenericAll|AllExtendedRights]->(ct:CertTemplate)-[:PublishedTo]->(eca:EnterpriseCA)
        WHERE eca.isuserspecifiessanenabled = True
        RETURN p
        LIMIT 1000
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_ca_administrators(self) -> dict[str, t.Any]:
        query = """
        MATCH p = (:Base)-[:ManageCertificates|ManageCA]->(:EnterpriseCA)
        RETURN p
        LIMIT 1000
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_onprem_users_with_direct_entra_roles(self) -> dict[str, t.Any]:
        query = """
        MATCH p = (:User)-[:SyncedToEntraUser]->(:AZUser)-[:AZHasRole]->(:AZRole)
        RETURN p
        LIMIT 1000
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_onprem_users_with_group_entra_roles(self) -> dict[str, t.Any]:
        query = """
        MATCH p = (:User)-[:SyncedToEntraUser]->(:AZUser)-[:AZMemberOf]->(:AZGroup)-[:AZHasRole]->(:AZRole)
        RETURN p
        LIMIT 1000
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_onprem_users_with_direct_azure_roles(self) -> dict[str, t.Any]:
        query = """
        MATCH p = (:User)-[:SyncedToEntraUser]->(:AZUser)-[:AZOwner|AZUserAccessAdministrator|AZGetCertificates|AZGetKeys|AZGetSecrets|AZAvereContributor|AZKeyVaultContributor|AZContributor|AZVMAdminLogin|AZVMContributor|AZAKSContributor|AZAutomationContributor|AZLogicAppContributor|AZWebsiteContributor]->(:AZBase)
        RETURN p
        LIMIT 1000
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_onprem_users_with_group_azure_roles(self) -> dict[str, t.Any]:
        query = """
        MATCH p = (:User)-[:SyncedToEntraUser]->(:AZUser)-[:AZMemberOf]->(:AZGroup)-[:AZOwner|AZUserAccessAdministrator|AZGetCertificates|AZGetKeys|AZGetSecrets|AZAvereContributor|AZKeyVaultContributor|AZContributor|AZVMAdminLogin|AZVMContributor|AZAKSContributor|AZAutomationContributor|AZLogicAppContributor|AZWebsiteContributor]->(:AZBase)
        RETURN p
        LIMIT 1000
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_paths_user_to_user(
        self, source_user: str, target_user: str, domain: str
    ) -> dict[str, t.Any]:
        """search for potential exploit/attack paths from source_user to target_user on the given domain"""
        query = f"""
        MATCH p=shortestPath((user1:User)-[*]->(user2:User)) 
        WHERE user1.name = "{source_user.upper()}@{domain.upper()}"
        AND user2.name = "{target_user.upper()}@{domain.upper()}"
        RETURN p
        """
        return await self.query_bloodhound(query)
