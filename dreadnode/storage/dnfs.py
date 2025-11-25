from s3fs import S3FileSystem


class DreadnodeFS(S3FileSystem):
    protocol = ("dn", "dreadnode")
