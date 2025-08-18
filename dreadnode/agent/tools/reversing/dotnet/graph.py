from dreadnode.agent.memory.graph.base import Universe

REV_DOTNET = Universe(
    labels={
        "Assembly",
        "Module",
        "Namespace",
        "Type",
        "Method",
        "Field",
        "Property",
        "Event",
        "Parameter",
        "Attribute",
        "Package",
    },
    allowed={
        ("Assembly", "Module"): {"Contains"},
        ("Module", "Namespace"): {"Contains"},
        ("Namespace", "Type"): {"Contains"},
        ("Type", "Method"): {"Defines"},
        ("Type", "Field"): {"Defines"},
        ("Type", "Property"): {"Defines"},
        ("Type", "Event"): {"Defines"},
        ("Method", "Method"): {"Calls", "Overrides"},
        ("Type", "Type"): {"Inherits", "Implements", "References"},
        ("Method", "Type"): {"References", "Allocates", "Throws"},
        ("Method", "Field"): {"Reads", "Writes"},
        ("Assembly", "Package"): {"DependsOn"},
    },
)
