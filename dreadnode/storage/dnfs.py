from s3fs import S3FileSystem

from dreadnode.util import resolve_endpoint


class DreadnodeS3FS(S3FileSystem):  # type: ignore
    protocol = ("dn", "dreadnode")

    @classmethod
    def from_credentials(
        cls,
        access_key: str,
        secret_key: str,
        session_token: str | None = None,
        endpoint_url: str | None = None,
        region_name: str | None = None,
        **kwargs,
    ) -> "DreadnodeS3FS":
        """Create a DreadnodeS3FS instance from credentials."""
        endpoint_url = resolve_endpoint(endpoint_url)
        return cls(
            key=access_key,
            secret=secret_key,
            token=session_token,
            client_kwargs={"endpoint_url": endpoint_url, "region_name": region_name},
            **kwargs,
        )
