from s3fs import S3FileSystem


class DreadnodeS3FS(S3FileSystem):
    protocol = ("dn", "dreadnode")
