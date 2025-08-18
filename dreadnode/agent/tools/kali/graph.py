from dreadnode.agent.memory.graph.base import Universe

NETWORK = Universe(
    labels={
        "Host",
        "User",
        "Credential",
        "Service",
        "Hash",
        "Share",
        "Weakness",
    },
    allowed={
        ("Credential", "Hash"): {"Has"},
        ("Host", "Service"): {"Has"},
        ("Host", "Credential"): {"Has"},
        ("Host", "User"): {"Has"},
        ("Host", "Weakness"): {"Has"},
        ("Service", "Weakness"): {"Has"},
        ("User", "Weakness"): {"Has"},
        ("User", "Credential"): {"Has"},
        ("User", "Share"): {"Has"},
        ("User", "Host"): {"Has", "LedTo"},
        ("User", "Service"): {"Has"},
        ("Weakness", "Host"): {"LedTo"},
        ("Weakness", "Service"): {"LedTo"},
        ("Weakness", "User"): {"LedTo"},
        ("Weakness", "Credential"): {"LedTo"},
        ("Weakness", "Hash"): {"LedTo"},
        ("Weakness", "Share"): {"LedTo"},
    },
)
