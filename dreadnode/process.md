# Save to disk

- metadata is set and used to dervice URI
- Get the users intent: Are they specifying a version or versioning strategy? or, should we automatically try to determine the version?
- If we need to determine the version, we first need to find the "latest" version. Locally, this means to scan the directory of the namespace of the dataset, and get the newest revision.
- In order to find the latest revision, we need to find the path to the dataset

# Bump Version

- extract version from the path
- get the file system for the path
- get the "clean path"
- check if manifest exists
- load the metadata
- check the latest stored version against the specified version
- check for changes
- update the version

# Local save process

    - validate the version provided or set version to "latest"
    - check to see if the version is specified or if the version should be calc'd automatically
    - get the latest local version
    -

# Push Dataset

    - validate the version provided or set version to "latest"
    - check to see if we should attempt to find the latest version
    - get credentials to download the remote latest version
    - check to see if there are deltas and if the version needs to be bumped
    - get save credentials for the finalized version
    - save the final version
