import asyncio
import json
import os
import time

import aiohttp
import rich
from dotenv import load_dotenv
from loguru import logger
from neo4j import GraphDatabase
from rich.panel import Panel

from dreadnode.agent.tools import Toolset, tool_method

# Load environment variables
load_dotenv()

# BloodHound & Neo4j connection details
BLOODHOUND_URL = os.getenv("BLOODHOUND_URL", "localhost:8080")
BLOODHOUND_USERNAME = os.getenv("BLOODHOUND_USERNAME", "admin")
BLOODHOUND_PASSWORD = os.getenv("BLOODHOUND_PASSWORD", "bloodhound")
BLOODHOUND_NEO4J_URL = os.getenv("BLOODHOUND_NEO4J_URL", "bolt://localhost:7687")
BLOODHOUND_NEO4J_USERNAME = os.getenv("BLOODHOUND_NEO4J_USERNAME", "neo4j")
BLOODHOUND_NEO4J_PASSWORD = os.getenv("BLOODHOUND_NEO4J_PASSWORD", "bloodhoundcommunityedition")


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

    async def _api_authenticate(self) -> dict | None:
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
            logger.error(self._rich_print("Couldnt authenticate to Bloodhound REST API."))
            return None

        self._api_auth_token = auth_token["data"]

        return self._api_auth_token

    @tool_method()
    async def query_bloodhound(self, query: str):
        databases = ["neo4j", "bloodhound"]
        last_error = None

        for db in databases:
            try:
                with self._graph_driver.session(database=db) as session:
                    result = session.run(query)
                    data = [record.data() for record in result]
                    logger.info(self._rich_print(f"Query successful on database '{db}'"))
                    return {"success": True, "data": data}
            except Exception as e:
                last_error = e
                logger.debug(self._rich_print(f"Query failed on database '{db}': {e!s}"))
                continue

        logger.error(self._rich_print(f"Query failed on all databases. Last error: {last_error!s}"))
        return {"success": False, "error": str(last_error)}

    # Domain Information
    @tool_method()
    async def find_all_domain_admins(self):
        query = """
        MATCH p = (t:Group)<-[:MemberOf*1..]-(a)
        WHERE (a:User or a:Computer) and t.objectid ENDS WITH '-512'
        RETURN p
        LIMIT 1000
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def map_domain_trusts(self):
        query = """
        MATCH p = (:Domain)-[:TrustedBy]->(:Domain)
        RETURN p
        LIMIT 1000
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_tier_zero_locations(self):
        query = """
        MATCH p = (t:Base)<-[:Contains*1..]-(:Domain)
        WHERE t.highvalue = true
        RETURN p
        LIMIT 1000
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def map_ou_structure(self):
        query = """
        MATCH p = (:Domain)-[:Contains*1..]->(:OU)
        RETURN p
        LIMIT 1000
        """
        return await self.query_bloodhound(query)

    # Dangerous Privileges
    @tool_method()
    async def find_dcsync_privileges(self):
        query = """
        MATCH p=(:Base)-[:DCSync|AllExtendedRights|GenericAll]->(:Domain)
        RETURN p
        LIMIT 1000
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_foreign_group_memberships(self):
        query = """
        MATCH p=(s:Base)-[:MemberOf]->(t:Group)
        WHERE s.domainsid<>t.domainsid
        RETURN p
        LIMIT 1000
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_domain_users_local_admins(self):
        query = """
        MATCH p=(s:Group)-[:AdminTo]->(:Computer)
        WHERE s.objectid ENDS WITH '-513'
        RETURN p
        LIMIT 1000
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_domain_users_laps_readers(self):
        query = """
        MATCH p=(s:Group)-[:AllExtendedRights|ReadLAPSPassword]->(:Computer)
        WHERE s.objectid ENDS WITH '-513'
        RETURN p
        LIMIT 1000
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_domain_users_high_value_paths(self):
        query = """
        MATCH p=shortestPath((s:Group)-[r*1..]->(t))
        WHERE t.highvalue = true AND s.objectid ENDS WITH '-513' AND s<>t
        RETURN p
        LIMIT 1000
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_domain_users_workstation_rdp(self):
        query = """
        MATCH p=(s:Group)-[:CanRDP]->(t:Computer)
        WHERE s.objectid ENDS WITH '-513' AND NOT toUpper(t.operatingsystem) CONTAINS 'SERVER'
        RETURN p
        LIMIT 1000
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_domain_users_server_rdp(self):
        query = """
        MATCH p=(s:Group)-[:CanRDP]->(t:Computer)
        WHERE s.objectid ENDS WITH '-513' AND toUpper(t.operatingsystem) CONTAINS 'SERVER'
        RETURN p
        LIMIT 1000
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_domain_users_privileges(self):
        query = """
        MATCH p=(s:Group)-[r]->(:Base)
        WHERE s.objectid ENDS WITH '-513'
        RETURN p
        LIMIT 1000
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_domain_admin_non_dc_logons(self):
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
    async def find_kerberoastable_tier_zero(self):
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
    async def find_all_kerberoastable_users(self):
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
    async def find_kerberoastable_most_admin(self):
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
    async def find_asreproast_users(self):
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
    async def find_shortest_paths_unconstrained_delegation(self):
        query = """
        MATCH p=shortestPath((s)-[r*1..]->(t:Computer))
        WHERE t.unconstraineddelegation = true AND s<>t
        RETURN p
        LIMIT 1000
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_paths_from_kerberoastable_to_da(self):
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
    async def find_shortest_paths_to_tier_zero(self):
        query = """
        MATCH p=shortestPath((s)-[r*1..]->(t))
        WHERE t.highvalue = true AND s<>t
        RETURN p
        LIMIT 1000
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_paths_from_domain_users_to_tier_zero(self):
        query = """
        MATCH p=shortestPath((s:Group)-[r*1..]->(t))
        WHERE t.highvalue = true AND s.objectid ENDS WITH '-513' AND s<>t
        RETURN p
        LIMIT 1000
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_shortest_paths_to_domain_admins(self):
        query = """
        MATCH p=shortestPath((t:Group)<-[r*1..]-(s:Base))
        WHERE t.objectid ENDS WITH '-512' AND s<>t
        RETURN p
        LIMIT 1000
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_paths_from_owned_objects(self):
        query = """
        MATCH p=shortestPath((s:Base)-[r*1..]->(t:Base))
        WHERE s.owned = true AND s<>t
        RETURN p
        LIMIT 1000
        """
        return await self.query_bloodhound(query)

    # Active Directory Certificate Services
    @tool_method()
    async def find_pki_hierarchy(self):
        query = """
        MATCH p=()-[:HostsCAService|IssuedSignedBy|EnterpriseCAFor|RootCAFor|TrustedForNTAuth|NTAuthStoreFor*..]->(:Domain)
        RETURN p
        LIMIT 1000
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_public_key_services(self):
        query = """
        MATCH p = (c:Container)-[:Contains*..]->(:Base)
        WHERE c.distinguishedname starts with 'CN=PUBLIC KEY SERVICES,CN=SERVICES,CN=CONFIGURATION,DC='
        RETURN p
        LIMIT 1000
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_certificate_enrollment_rights(self):
        query = """
        MATCH p = (:Base)-[:Enroll|GenericAll|AllExtendedRights]->(:CertTemplate)-[:PublishedTo]->(:EnterpriseCA)
        RETURN p
        LIMIT 1000
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_esc1_vulnerable_templates(self):
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
    async def find_esc2_vulnerable_templates(self):
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
    async def find_enrollment_agent_templates(self):
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
    async def find_dcs_weak_certificate_binding(self):
        query = """
        MATCH p = (s:Computer)-[:DCFor]->(:Domain)
        WHERE s.strongcertificatebindingenforcementraw = 0 OR s.strongcertificatebindingenforcementraw = 1
        RETURN p
        LIMIT 1000
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_inactive_tier_zero_principals(self):
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
    async def find_tier_zero_without_smartcard(self):
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
    async def find_domains_with_machine_quota(self):
        query = """
        MATCH (d:Domain)
        WHERE d.machineaccountquota > 0
        RETURN d
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_smartcard_dont_expire_domains(self):
        query = """
        MATCH (s:Domain)-[:Contains*1..]->(t:Base)
        WHERE s.expirepasswordsonsmartcardonlyaccounts = false
        AND t.enabled = true
        AND t.smartcardrequired = true
        RETURN s
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_two_way_forest_trust_delegation(self):
        query = """
        MATCH p=(n:Domain)-[r:TrustedBy]->(m:Domain)
        WHERE (m)-[:TrustedBy]->(n)
        AND r.trusttype = 'Forest'
        AND r.tgtdelegationenabled = true
        RETURN p
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_unsupported_operating_systems(self):
        query = """
        MATCH (c:Computer)
        WHERE c.operatingsystem =~ '(?i).*Windows.* (2000|2003|2008|2012|xp|vista|7|8|me|nt).*'
        RETURN c
        LIMIT 100
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_users_with_no_password_required(self):
        query = """
        MATCH (u:User)
        WHERE u.passwordnotreqd = true
        RETURN u
        LIMIT 100
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_users_password_not_rotated(self):
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
    async def find_nested_tier_zero_groups(self):
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
    async def find_disabled_tier_zero_principals(self):
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
    async def find_principals_reversible_encryption(self):
        query = """
        MATCH (n:Base)
        WHERE n.encryptedtextpwdallowed = true
        RETURN n
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_principals_des_only_kerberos(self):
        query = """
        MATCH (n:Base)
        WHERE n.enabled = true
        AND n.usedeskeyonly = true
        RETURN n
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_principals_weak_kerberos_encryption(self):
        query = """
        MATCH (u:Base)
        WHERE 'DES-CBC-CRC' IN u.supportedencryptiontypes
        OR 'DES-CBC-MD5' IN u.supportedencryptiontypes
        OR 'RC4-HMAC-MD5' IN u.supportedencryptiontypes
        RETURN u
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_tier_zero_non_expiring_passwords(self):
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
    async def find_ntlm_relay_edges(self):
        query = """
        MATCH p = (n:Base)-[:CoerceAndRelayNTLMToLDAP|CoerceAndRelayNTLMToLDAPS|CoerceAndRelayNTLMToADCS|CoerceAndRelayNTLMToSMB]->(:Base)
        RETURN p LIMIT 500
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_esc8_vulnerable_cas(self):
        query = """
        MATCH (n:EnterpriseCA)
        WHERE n.hasvulnerableendpoint=true
        RETURN n
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_computers_outbound_ntlm_deny(self):
        query = """
        MATCH (c:Computer)
        WHERE c.restrictoutboundntlm = True
        RETURN c LIMIT 1000
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_computers_in_protected_users(self):
        query = """
        MATCH p = (:Base)-[:MemberOf*1..]->(g:Group)
        WHERE g.objectid ENDS WITH "-525"
        RETURN p LIMIT 1000
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_dcs_vulnerable_ntlm_relay(self):
        query = """
        MATCH p = (dc:Computer)-[:DCFor]->(:Domain)
        WHERE (dc.ldapavailable = True AND dc.ldapsigning = False)
        OR (dc.ldapsavailable = True AND dc.ldapsepa = False)
        OR (dc.ldapavailable = True AND dc.ldapsavailable = True AND dc.ldapsigning = False and dc.ldapsepa = True)
        RETURN p
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_computers_webclient_running(self):
        query = """
        MATCH (c:Computer)
        WHERE c.webclientrunning = True
        RETURN c LIMIT 1000
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_computers_no_smb_signing(self):
        query = """
        MATCH (n:Computer)
        WHERE n.smbsigning = False
        RETURN n
        """
        return await self.query_bloodhound(query)

    # Azure - General
    @tool_method()
    async def find_global_administrators(self):
        query = """
        MATCH p = (:AZBase)-[:AZGlobalAdmin*1..]->(:AZTenant)
        RETURN p
        LIMIT 1000
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_high_privileged_role_members(self):
        query = """
        MATCH p=(t:AZRole)<-[:AZHasRole|AZMemberOf*1..2]-(:AZBase)
        WHERE t.name =~ '(?i)(Global Administrator|User Access Administrator|Privileged Role Administrator|Privileged Authentication Administrator|Partner Tier1 Support|Partner Tier2 Support)'
        RETURN p
        LIMIT 1000
        """
        return await self.query_bloodhound(query)

    # Azure - Shortest Paths
    @tool_method()
    async def find_paths_from_entra_to_tier_zero(self):
        query = """
        MATCH p=shortestPath((s:AZUser)-[r*1..]->(t:AZBase))
        WHERE t.highvalue = true AND t.name =~ '(?i)(Global Administrator|User Access Administrator|Privileged Role Administrator|Privileged Authentication Administrator|Partner Tier1 Support|Partner Tier2 Support)' AND s<>t
        RETURN p
        LIMIT 1000
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_paths_to_privileged_roles(self):
        query = """
        MATCH p=shortestPath((s:AZBase)-[r*1..]->(t:AZRole))
        WHERE t.name =~ '(?i)(Global Administrator|User Access Administrator|Privileged Role Administrator|Privileged Authentication Administrator|Partner Tier1 Support|Partner Tier2 Support)' AND s<>t
        RETURN p
        LIMIT 1000
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_paths_from_azure_apps_to_tier_zero(self):
        query = """
        MATCH p=shortestPath((s:AZApp)-[r*1..]->(t:AZBase))
        WHERE t.highvalue = true AND s<>t
        RETURN p
        LIMIT 1000
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_paths_to_azure_subscriptions(self):
        query = """
        MATCH p=shortestPath((s:AZBase)-[r*1..]->(t:AZSubscription))
        WHERE s<>t
        RETURN p
        LIMIT 1000
        """
        return await self.query_bloodhound(query)

    # Azure - Microsoft Graph
    @tool_method()(name="sp_app_role_grant")
    async def find_service_principals_with_app_role_grant(self):
        query = """
        MATCH p=(:AZServicePrincipal)-[:AZMGGrantAppRoles]->(:AZTenant)
        RETURN p
        LIMIT 1000
        """
        return await self.query_bloodhound(query)

    @tool_method()(name="find_sp_graph_assignments")
    async def find_service_principals_with_graph_assignments(self):
        query = """
        MATCH p=(:AZServicePrincipal)-[:AZMGAppRoleAssignment_ReadWrite_All|AZMGApplication_ReadWrite_All|AZMGDirectory_ReadWrite_All|AZMGGroupMember_ReadWrite_All|AZMGGroup_ReadWrite_All|AZMGRoleManagement_ReadWrite_Directory|AZMGServicePrincipalEndpoint_ReadWrite_All]->(:AZServicePrincipal)
        RETURN p
        LIMIT 1000
        """
        return await self.query_bloodhound(query)

    # Azure - Hygiene
    @tool_method()
    async def find_foreign_tier_zero_principals(self):
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
    async def find_synced_tier_zero_principals(self):
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
    async def find_external_tier_zero_users(self):
        query = """
        MATCH (n:AZUser)
        WHERE n.highvalue = true
        AND n.name CONTAINS '#EXT#@'
        RETURN n
        LIMIT 100
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_disabled_azure_tier_zero_principals(self):
        query = """
        MATCH (n:AZBase)
        WHERE n.highvalue = true
        AND n.enabled = false
        RETURN n
        LIMIT 100
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_devices_unsupported_os(self):
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
    async def find_entra_users_in_domain_admins(self):
        query = """
        MATCH p = (:AZUser)-[:SyncedToADUser]->(:User)-[:MemberOf]->(t:Group)
        WHERE t.objectid ENDS WITH '-512'
        RETURN p
        LIMIT 1000
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_onprem_users_owning_entra_objects(self):
        query = """
        MATCH p = (:User)-[:SyncedToEntraUser]->(:AZUser)-[:AZOwns]->(:AZBase)
        RETURN p
        LIMIT 1000
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_onprem_users_in_entra_groups(self):
        query = """
        MATCH p = (:User)-[:SyncedToEntraUser]->(:AZUser)-[:AZMemberOf]->(:AZGroup)
        RETURN p
        LIMIT 1000
        """
        return await self.query_bloodhound(query)

    @tool_method()(name="templates_no_security_ext")
    async def find_templates_no_security_extension(self):
        query = """
        MATCH p = (:Base)-[:Enroll|GenericAll|AllExtendedRights]->(ct:CertTemplate)-[:PublishedTo]->(:EnterpriseCA)
        WHERE ct.nosecurityextension = true
        RETURN p
        LIMIT 1000
        """
        return await self.query_bloodhound(query)

    @tool_method()(name="templates_with_user_san")
    async def find_templates_with_user_specified_san(self):
        query = """
        MATCH p = (:Base)-[:Enroll|GenericAll|AllExtendedRights]->(ct:CertTemplate)-[:PublishedTo]->(eca:EnterpriseCA)
        WHERE eca.isuserspecifiessanenabled = True
        RETURN p
        LIMIT 1000
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_ca_administrators(self):
        query = """
        MATCH p = (:Base)-[:ManageCertificates|ManageCA]->(:EnterpriseCA)
        RETURN p
        LIMIT 1000
        """
        return await self.query_bloodhound(query)

    @tool_method()(name="onprem_users_direct_entra_roles")
    async def find_onprem_users_with_direct_entra_roles(self):
        query = """
        MATCH p = (:User)-[:SyncedToEntraUser]->(:AZUser)-[:AZHasRole]->(:AZRole)
        RETURN p
        LIMIT 1000
        """
        return await self.query_bloodhound(query)

    @tool_method()(name="onprem_users_group_entra_roles")
    async def find_onprem_users_with_group_entra_roles(self):
        query = """
        MATCH p = (:User)-[:SyncedToEntraUser]->(:AZUser)-[:AZMemberOf]->(:AZGroup)-[:AZHasRole]->(:AZRole)
        RETURN p
        LIMIT 1000
        """
        return await self.query_bloodhound(query)

    @tool_method()(name="onprem_users_direct_azure_roles")
    async def find_onprem_users_with_direct_azure_roles(self):
        query = """
        MATCH p = (:User)-[:SyncedToEntraUser]->(:AZUser)-[:AZOwner|AZUserAccessAdministrator|AZGetCertificates|AZGetKeys|AZGetSecrets|AZAvereContributor|AZKeyVaultContributor|AZContributor|AZVMAdminLogin|AZVMContributor|AZAKSContributor|AZAutomationContributor|AZLogicAppContributor|AZWebsiteContributor]->(:AZBase)
        RETURN p
        LIMIT 1000
        """
        return await self.query_bloodhound(query)

    @tool_method()(name="onprem_users_group_azure_roles")
    async def find_onprem_users_with_group_azure_roles(self):
        query = """
        MATCH p = (:User)-[:SyncedToEntraUser]->(:AZUser)-[:AZMemberOf]->(:AZGroup)-[:AZOwner|AZUserAccessAdministrator|AZGetCertificates|AZGetKeys|AZGetSecrets|AZAvereContributor|AZKeyVaultContributor|AZContributor|AZVMAdminLogin|AZVMContributor|AZAKSContributor|AZAutomationContributor|AZLogicAppContributor|AZWebsiteContributor]->(:AZBase)
        RETURN p
        LIMIT 1000
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def find_paths_user_to_user(
        self, source_user: str, target_user: str, domain: str
    ) -> dict:
        """search for potential exploit/attack paths from source_user to target_user on the given domain"""
        query = f"""
        MATCH p=shortestPath((user1:User)-[*]->(user2:User)) 
        WHERE user1.name = "{source_user.upper()}@{domain.upper()}"
        AND user2.name = "{target_user.upper()}@{domain.upper()}"
        RETURN p
        """
        return await self.query_bloodhound(query)

    @tool_method()
    async def upload_collection_zip(self, filename: str) -> dict:
        """Upload a Bloodhound collection zip file (that was collected via the SharpHound tool.)"""

        if self._api_auth_token is None or self._api_auth_token.get("auth_expired", True):
            await self._api_authenticate()

        # 1. start Bloodhound server upload job
        start_job = {
            "url": f"http://{self.config['url']}/api/v2/file-upload/start",
            "headers": {
                "accept": "application/json",
                "Authorization": f"Bearer {self._api_auth_token['session_token']}",
            },
        }
        job_record = await self._async_post_request(resp_type="json", **start_job)
        job_record = job_record["data"]

        if not job_record.get("id", False):
            err_msg = (
                f"Could not start collection upload on Bloodhound server. Error: {job_record}."
            )
            logger.error(self._rich_print(err_msg))
            return err_msg

        # 2. upload Bloodhound collection files
        upload_fn = os.path.abspath(filename)
        upload_job = {
            "url": f"http://{self.config['url']}/api/v2/file-upload/{job_record['id']}",
            "headers": {
                "accept": "application/zip",
                "Authorization": f"Bearer {self._api_auth_token['session_token']}",
            },
        }
        try:
            upload_job_status = await self._async_post_file(filename=upload_fn, **upload_job)
            logger.info(
                self._rich_print(
                    f"Collection file upload initiated: {upload_fn}.\n\nStatus: {upload_job_status}"
                )
            )
        except Exception as e:
            err_msg = f"Error uploading collection file to Bloodhound: {e}"
            logger.error(self._rich_print(err_msg))
            return err_msg

        # await asyncio.sleep(45)
        # 3. end Bloodhound server upload job
        end_job = {
            "url": f"http://{self.config['url']}/api/v2/file-upload/{job_record['id']}/end",
            "headers": {
                "accept": "application/json",
                "Authorization": f"Bearer {self._api_auth_token['session_token']}",
            },
        }
        end_job_record = await self._async_post_request(resp_type="text", **end_job)

        # wait for upload to complete
        upload_job_done, upload_job_status = await self.wait_for_upload_completion(
            job_id=job_record["id"], seconds=60
        )
        if not upload_job_done:
            err_msg = f"Timeout error of collection file upload for {upload_fn}.\n\n Dumping upload job status: {upload_job_status}"
            logger.error(self._rich_print(err_msg))
            return err_msg

        success_msg = f"Successfully uploaded {filename} collection file to Bloodhound."
        logger.info(self._rich_print(success_msg))

        return success_msg

    """ Utilities """

    async def clear_database(self) -> str:
        """clears the bloodhound database"""
        clear_db_req = {
            "url": f"http://{self.config['url']}/api/v2/clear-database",
            "headers": {
                "accept": "application/plain",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_auth_token['session_token']}",
            },
            "data": json.dumps(
                {
                    "deleteCollectedGraphData": True,
                    "deleteFileIngestHistory": False,
                    "deleteDataQualityHistory": True,
                    "deleteAssetGroupSelectors": [0],
                }
            ),
        }
        clear_status = await self._async_post_request(**clear_db_req)
        logger.info(self._rich_print(f"Cleared Bloodhound database - {clear_status}"))
        return clear_status

    async def wait_for_upload_completion(self, job_id: int, seconds: int = 10) -> tuple[bool, dict]:
        """ """
        start_time = int(time.time())
        while True:
            await asyncio.sleep(2)
            job_done, job_status = await self.upload_job_status(job_id=job_id)
            if job_done:
                break
            if start_time + seconds < int(time.time()):
                break
        return job_done, job_status

    async def upload_job_status(self, job_id: int) -> tuple[bool, dict]:
        """ """
        upload_status_job = {
            "url": f"http://{self.config['url']}/api/v2/file-upload?id={job_id!s}",
            "headers": {
                "accept": "application/json",
                "Authorization": f"Bearer {self._api_auth_token['session_token']}",
            },
        }
        job_statuses = await self._async_get_request(resp_type="json", **upload_status_job)
        job_status = [j for j in job_statuses["data"] if j["id"] == job_id][0]
        job_done = True if job_status["status"] == 2 else False
        return job_done, job_status

    async def _async_get_request(self, resp_type: str = None, **kwargs) -> dict:
        """ """
        async with aiohttp.ClientSession() as session:
            async with session.get(**kwargs) as resp:
                if resp_type == "json":
                    return await resp.json()
                if resp_type == "text":
                    return await resp.text()
                return str(resp)

    async def _async_post_request(self, resp_type: str = None, **kwargs) -> dict:
        """ """
        response = None
        async with aiohttp.ClientSession() as session:
            async with session.post(**kwargs) as resp:
                if resp_type == "json":
                    response = await resp.json()
                elif resp_type == "text":
                    response = await resp.text()
                else:
                    response = str(resp)
        return response

    async def _async_post_file(self, url: str, filename: str, **kwargs) -> dict:
        """ """
        response = None
        with open(filename, "rb") as fh:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, data=fh, **kwargs) as resp:
                    if resp.status != 202:
                        resp.raise_for_status()
                        response = resp
        return response

    def _rich_print(self, text: str):
        """ """
        return rich.print(
            Panel(
                f"[white]{text}",
                title="[red]Bloodhound",
                style="red",
            )
        )
