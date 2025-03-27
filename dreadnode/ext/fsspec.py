import typing as t

from fsspec import register_implementation  # type: ignore [import-untyped]
from fsspec.implementations.dirfs import DirFileSystem  # type: ignore [import-untyped]
from s3fs import S3FileSystem  # type: ignore [import-untyped]

from dreadnode.api.client import ApiClient


class DreadnodeS3Filesystem(DirFileSystem):  # type: ignore [misc]
    def __init__(
        self,
        api_client: ApiClient | None = None,
        dirfs_options: dict[str, t.Any] | None = None,
        **s3fs_options: t.Any,
    ) -> None:
        from dreadnode.main import DEFAULT_INSTANCE

        self.api_client = api_client or DEFAULT_INSTANCE.api()
        credentials = self.api_client.get_user_data_credentials()

        s3_fs = S3FileSystem(
            key=credentials.access_key_id,
            secret=credentials.secret_access_key,
            token=credentials.session_token,
            endpoint_url=credentials.endpoint,
            client_kwargs={"region_name": credentials.region},
            **s3fs_options,
        )

        super().__init__(
            path=f"{credentials.bucket}/{credentials.prefix}/",
            fs=s3_fs,
            **(dirfs_options or {}),
        )


register_implementation("dn", DreadnodeS3Filesystem)
