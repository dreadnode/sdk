import typing as t

from rich.table import Table


def modules_table(
    module_loader: t.Any,
    modules: list[str] | None = None,
    mod_type: str | None = None,
    *,
    include_author: bool = False,
    include_created_date: bool = False,
) -> Table:
    """
    Creates and prints a rich table of modules.
    """
    table = Table(title="Modules Overview")

    header = [
        "Module",
        "Type",
        "Needs API Key",
        "Description",
        "Flags",
        "Consumed Events",
        "Produced Events",
    ]
    if include_author:
        header.append("Author")
    if include_created_date:
        header.append("Created Date")

    table.add_column("Module", style="cyan", no_wrap=True)
    table.add_column("Type", style="magenta")
    table.add_column("Needs API Key", justify="center")
    table.add_column("Description", width=30)
    table.add_column("Flags")
    table.add_column("Consumed Events")
    table.add_column("Produced Events")
    if include_author:
        table.add_column("Author", style="green")
    if include_created_date:
        table.add_column("Created Date")

    for module_name, preloaded in module_loader.filter_modules(modules, mod_type):
        module_type = preloaded["type"]
        consumed_events = sorted(preloaded.get("watched_events", []))
        produced_events = sorted(preloaded.get("produced_events", []))
        flags = sorted(preloaded.get("flags", []))
        meta = preloaded.get("meta", {})
        api_key_required = "Yes" if meta.get("auth_required", False) else "No"
        description = meta.get("description", "")

        row_data = [
            module_name,
            module_type,
            api_key_required,
            description,
            ", ".join(flags),
            ", ".join(consumed_events),
            ", ".join(produced_events),
        ]

        if include_author:
            author = meta.get("author", "")
            row_data.append(author)
        if include_created_date:
            created_date = meta.get("created_date", "")
            row_data.append(created_date)

        table.add_row(*row_data)

    return table


def presets_table(module_loader: t.Any, *, include_modules: bool = True) -> Table:
    """
    Prints a rich table of all available presets.
    """
    table = Table(title="Available Presets")

    # Define the columns and their styles
    table.add_column("Preset", style="cyan", no_wrap=True)
    table.add_column("Category", style="magenta")
    table.add_column("Description", width=40)
    table.add_column("# Modules", justify="right", style="green")

    if include_modules:
        table.add_column("Modules", style="yellow")

    for loaded_preset, category, preset_path, original_file in module_loader.all_presets.values():
        baked_preset = loaded_preset.bake()
        num_modules = f"{len(baked_preset.scan_modules):,}"

        row_data = [
            baked_preset.name,
            category,
            baked_preset.description,
            num_modules,
        ]

        if include_modules:
            modules_str = ", ".join(sorted(baked_preset.scan_modules))
            row_data.append(modules_str)

        table.add_row(*row_data)

    return table


def flags_table(module_loader: t.Any, flags: list[str] | None = None) -> Table:
    """
    Prints a rich table of flags, their descriptions, and associated modules.
    """
    from bbot.core.modules import flag_descriptions

    table = Table(title="Module Flags")

    # Define columns
    table.add_column("Flag", style="cyan", no_wrap=True)
    table.add_column("# Modules", justify="right", style="green")
    table.add_column("Description", width=40)
    table.add_column("Modules", style="yellow")

    _flags = module_loader.flags(flags=flags)
    for flag, modules in _flags:
        description = flag_descriptions.get(flag, "")
        table.add_row(flag, f"{len(modules)}", description, ", ".join(sorted(modules)))

    return table


def events_table(module_loader: t.Any) -> Table:
    """
    Prints a rich table of events and the modules that consume or produce them.
    """
    table = Table(title="Module Event Interactions")

    # Define columns
    table.add_column("Event Type", style="cyan", no_wrap=True)
    table.add_column("# Consuming", justify="right", style="yellow")
    table.add_column("# Producing", justify="right", style="magenta")
    table.add_column("Consuming Modules", style="yellow")
    table.add_column("Producing Modules", style="magenta")

    consuming_events, producing_events = module_loader.events()
    all_event_types = sorted(set(consuming_events).union(set(producing_events)))

    for event_type in all_event_types:
        consuming_modules = sorted(consuming_events.get(event_type, []))
        producing_modules = sorted(producing_events.get(event_type, []))

        table.add_row(
            event_type,
            str(len(consuming_modules)),
            str(len(producing_modules)),
            ", ".join(consuming_modules),
            ", ".join(producing_modules),
        )

    return table
