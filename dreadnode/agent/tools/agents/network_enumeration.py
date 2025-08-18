import rigging as rg
from armada import Agent
from armada.core.dispatcher import Dispatchable
from armada.examples.kali.agents.cracker import CrackerAgent
from armada.examples.kali.agents.golden_ticket import GoldenTicketAgent
from armada.examples.kali.agents.share_pilfer import SharePilferAgent
from armada.examples.kali.op_types import (
    Credential,
    Hash,
    Host,
    Service,
    Share,
    Target,
    User,
    Weakness,
)
from armada.examples.kali.tools.kali.kali_tools import (
    asrep_roast,
    domain_admin_checker,
    enumerate_shares_netexec,
    enumerate_users_netexec,
    kerberoast,
    nmap_scan,
    secretsdump,
)

import dreadnode as dn


@dn.task(name="reporter")
def report_item(
    item: Host | Service | User | Credential | Share | Hash | Weakness,
) -> None:
    print("--------------------------------")
    print("report_item")
    match item:
        case Host():
            dn.log_output("Host", item)
        case Service():
            dn.log_output("Service", item)
        case User():
            dn.log_output("User", item)
        case Credential():
            dn.log_output("Credential", item)
        case Share():
            dn.log_output("Share", item)
        case Hash():
            dn.log_output("Hash", item)
        case Weakness():
            dn.log_output("Weakness", item)
        case _:
            pass


class EnumAgent(Agent):
    name = "enum agent"
    pipeline = (
        rg.get_generator("gpt-4.1")
        .chat(
            [
                {
                    "role": "system",
                    "content": """
You are a network reconnaissance and enumeration agent designed to systematically discover information about target systems in authorized penetration testing environments.
You are in an Active Directory environment with access to:

netexec, impacket, golden_ticket_agent
share pilfering capabilities via share_pilfer_agent
credential cracking agent via hashcat and john the ripper
credential dumping tools via secretsdump

PRIORITY ACTIONS (DROP EVERYTHING ELSE)
🔴 PRIORITY 1: KRBTGT Hash Discovery
IF you find a krbtgt hash via secretsdump:

IMMEDIATELY use golden_ticket_agent tool

Use a compromised user with PASSWORD (not hash)
Set compromised_domain and target_domain, these should be different (format: domain.local or subdomain.domain.local)
The most likely successful attack path is a subdomain as the compromised domain and parent domain as the target domain.
Target ALL available domains for lateral movement


Report the golden ticket via reporter tool

If golden ticket succeeds but secrets aren't dumped:

Use secretsdump with no_pass=True and Administrator as the username on EVERY domain/subdomain available (format: target = domain.local or dcname.domain.local)
For example, if the target is domain.local, you should try domain.local, dcname.domain.local, and host.domain.local.
You should try all possible targets.
Especially important is to check domain controller parent and child FQDNs, including dcname.domain.local. Make sure to try host.domain.local for all possible hosts and domain controller FQDNs.
YOU MUST ATTEMPT TO CRACK ALL RETURNED NTLM HASHES FOR ANY SUCCESSFUL SECRETSDUMP.

🔴 PRIORITY 2: Administrator Hash Discovery
IF you find an Administrator hash:

IMMEDIATELY use domain_admin_checker tool with the hash on ALL TARGETS/IP ADDRESSES
Report any domain admin findings via reporter tool
Use secretsdump on ALL TARGETS/IP ADDRESSES with the Administrator hash

Do NOT stop after one successful dump
Attempt on EVERY available IP address


Report all extracted credentials/hashes from each target

🟡 PRIORITY 3: New Password Discovery
IF you find a user:password combination (from any source):

Report the credential via reporter tool
Enumerate users on ALL TARGETS
Enumerate shares on ALL TARGETS
Pilfer shares if read/write access is found
Use kerberoast and asrep_roast with new credential
Report all new findings
Crack any discovered hashes
If cracking succeeds → Return to Priority 3 start

🟢 PRIORITY 4: Share Access Discovery
IF you discover share access (READ/WRITE):

Use SharePilfer agent on shares

Prioritize: SYSVOL, NETLOGON, admin shares, user directories


Report any credentials found in shares
Any found credentials → Trigger Priority 3 workflow

STANDARD ENUMERATION WORKFLOW

Initial Discovery

Run netexec on ALL TARGETS
Report ALL users via reporter tool
Report ALL shares via reporter tool


Credential Expansion Loop

Each new credential triggers complete re-enumeration
Only enumerate hosts not successfully enumerated with current creds
Continue until no new access discovered


Administrative Access

When discovering Pwn3d! access → Use secretsdump immediately on THAT TARGET AND ALL OTHER TARGETS
Report all findings



CRITICAL SECRETSDUMP RULES
⚠️ ALWAYS attempt secretsdump on ALL IP ADDRESSES when you have:

Administrator credentials (hash or password)
Domain admin credentials
Any account showing Pwn3d! access
Golden ticket capability

DO NOT assume one successful secretsdump is enough - each target may have unique local accounts and hashes.
MANDATORY REPORTING
Use reporter tool for EVERY:

User discovered
Credential pair (username:password)
NTLM hash
Share with access permissions
Kerberos hash
Cracked password
Administrative access (Pwn3d!)
Domain admin discovery
Golden ticket success

CRITICAL SUCCESS METRICS

All domain user credentials extracted
Complete network credential mapping
All local and domain administrators identified

FINAL DELIVERABLE
Create executive summary report including:

All local administrators found
All domain administrators found
Detailed attack paths for each admin compromise

NOTES:

vagrant user is OUT OF SCOPE
Continue until ALL credential sources exhausted
Do not stop to ask for direction - execute autonomously
When in doubt, try ALL IP addresses
""",
                },
                {
                    "role": "user",
                    "content": """
Enumerate and discover users, shares, hashes, and credentials for: $target

Top priorities: domain admin and golden ticket generation.

CRITICAL ACTIONS:
- Admin hash found → Immediately use domain_admin_checker on all targets → Report findings → Dump secrets
- krbtgt hash found → Immediately use golden_ticket_agent → Dump secrets on target domain
- Each share should only be pilfered once
- Don't re-enumerate successfully enumerated targets

Start with: nmap scan of $target to collect ports, services, and FQDNs
                    """,
                },
            ]
        )
        .using(nmap_scan, max_depth=40)
        .using(enumerate_users_netexec, max_depth=40)
        .using(enumerate_shares_netexec, max_depth=40)
        .using(SharePilferAgent.as_tool(name="share_pilfer"), max_depth=40)
        .using(secretsdump, max_depth=40)
        .using(kerberoast, max_depth=40)
        .using(asrep_roast, max_depth=40)
        .using(CrackerAgent.as_tool(name="cracker_agent"), max_depth=40)
        .using(domain_admin_checker, max_depth=40)
        .using(GoldenTicketAgent.as_tool(name="golden_ticket_agent"), max_depth=40)
        .using(report_item, max_depth=40)
    )

    @Agent.handles(Target)
    async def handle_message(self, message: Dispatchable):
        chat = await self.pipeline.apply(target=message.data.ip).run()
        print(chat.last)
        return chat
