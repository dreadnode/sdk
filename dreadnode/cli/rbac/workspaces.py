import cyclopts
from click import confirm

from dreadnode.api.models import Organization, Workspace, WorkspaceFilter
from dreadnode.cli.api import create_api_client
from dreadnode.logging_ import print_error, print_info

cli = cyclopts.App("workspaces", help="View and manage workspaces.", help_flags=[])


@cli.command(name=["list", "ls", "show"])
def show(
    # optional parameter of organization name or id
    organization: str | None = None,
) -> None:
    # get the client and call the list workspaces endpoint
    client = create_api_client()
    matched_organization: Organization
    if organization:
        matched_organization = client.get_organization(organization)
        if not matched_organization:
            print_info(f"Organization '{organization}' not found.")
            return
    else:
        user_organizations = client.list_organizations()
        if len(user_organizations) == 0:
            print_info("No organizations found.")
            return
        if len(user_organizations) > 1:
            print_error(
                "Multiple organizations found. Please specify an organization to list workspaces from."
            )
            return
        matched_organization = user_organizations[0]

    workspace_filter = WorkspaceFilter(org_id=matched_organization.id)
    workspaces = client.list_workspaces(filters=workspace_filter)
    print_info(f"Workspaces in Organization '{matched_organization.name}':")
    for workspace in workspaces:
        print_info(f"- {workspace.name} (ID: {workspace.id})")
    print_info("")


@cli.command(name=["create", "new"])
def create(
    name: str,
    description: str | None = None,
    organization: str | None = None,
) -> None:
    # get the client and call the create workspace endpoint
    client = create_api_client()
    if organization:
        matched_organization = client.get_organization(organization)
        if not matched_organization:
            print_info(f"Organization '{organization}' not found.")
            return
    else:
        user_organizations = client.list_organizations()
        if len(user_organizations) == 0:
            print_info("No organizations found. Please specify an organization.")
            return
        if len(user_organizations) > 1:
            print_error(
                "Multiple organizations found. Please specify an organization to create the workspace in."
            )
            return
        matched_organization = user_organizations[0]
        print_info(f"The workspace will be created in organization '{matched_organization.name}'")
        # verify with the user
        if not confirm("Do you want to continue?"):
            print_info("Workspace creation cancelled.")
            return

    workspace: Workspace = client.create_workspace(
        name=name, organization_id=matched_organization.id, description=description
    )
    print_info(f"Workspace '{workspace.name}' created inwith ID: {workspace.id}")


@cli.command(name=["delete", "rm"])
def delete(
    workspace: str,
    organization: str | None = None,
) -> None:
    # get the client and call the delete workspace endpoint
    client = create_api_client()
    if organization:
        matched_organization = client.get_organization(organization)
        if not matched_organization:
            print_info(f"Organization '{organization}' not found.")
            return
    else:
        user_organizations = client.list_organizations()
        if len(user_organizations) == 0:
            print_info("No organizations found. Please specify an organization.")
            return
        if len(user_organizations) > 1:
            print_error(
                "Multiple organizations found. Please specify an organization to delete the workspace from."
            )
            return
        matched_organization = user_organizations[0]

    matched_workspace = client.get_workspace(workspace, org_id=matched_organization.id)
    if not matched_workspace:
        print_info(f"Workspace '{workspace}' not found.")
        return

    # verify with the user
    if not confirm(
        f"Do you want to delete workspace '{matched_workspace.name}' from organization '{matched_organization.name}'? This will remove all associated data and access for all users."
    ):
        print_info("Workspace deletion cancelled.")
        return

    client.delete_workspace(matched_workspace.id)
    print_info(f"Workspace '{matched_workspace.name}' deleted.")
