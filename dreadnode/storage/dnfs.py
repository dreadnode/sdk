from s3fs import S3FileSystem


class DreadnodeS3FS(S3FileSystem):  # type: ignore
    protocol = ("dn", "dreadnode")
