"""AWS ECR helper for air-gapped deployments."""

import re
from typing import Optional

from loguru import logger

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError

    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    boto3 = None  # type: ignore
    ClientError = Exception  # type: ignore
    NoCredentialsError = Exception  # type: ignore


class ECRHelper:
    """Manages ECR operations for air-gapped deployments."""

    def __init__(self, registry_url: str, region: Optional[str] = None):
        """
        Initialize ECR helper.

        Args:
            registry_url: ECR registry URL (e.g., 123456.dkr.ecr.us-east-1.amazonaws.com)
            region: AWS region (auto-detected from registry_url if not provided)

        Raises:
            RuntimeError: If boto3 is not installed or configuration is invalid
        """
        if not BOTO3_AVAILABLE:
            raise RuntimeError(
                "boto3 is required for ECR operations. "
                "Install it with: pip install boto3"
            )

        self.registry_url = registry_url
        self.account_id = self._extract_account_id(registry_url)
        self.region = region or self._extract_region(registry_url)

        try:
            self.ecr_client = boto3.client("ecr", region_name=self.region)
        except NoCredentialsError:
            raise RuntimeError(
                "AWS credentials not found. "
                "Configure credentials using AWS CLI or environment variables."
            )

        logger.debug(
            f"Initialized ECR helper for account {self.account_id} in region {self.region}"
        )

    def _extract_account_id(self, registry_url: str) -> str:
        """Extract AWS account ID from ECR registry URL."""
        match = re.match(r"(\d+)\.dkr\.ecr\.", registry_url)
        if not match:
            raise ValueError(
                f"Invalid ECR registry URL: {registry_url}. "
                "Expected format: <account-id>.dkr.ecr.<region>.amazonaws.com"
            )
        return match.group(1)

    def _extract_region(self, registry_url: str) -> str:
        """Extract AWS region from ECR registry URL."""
        match = re.search(r"\.ecr\.([a-z0-9-]+)\.amazonaws\.com", registry_url)
        if not match:
            raise ValueError(
                f"Invalid ECR registry URL: {registry_url}. "
                "Expected format: <account-id>.dkr.ecr.<region>.amazonaws.com"
            )
        return match.group(1)

    def verify_connectivity(self) -> bool:
        """
        Verify ECR connectivity without requiring internet access.

        Returns:
            True if ECR is accessible, False otherwise
        """
        try:
            # Simple API call to verify credentials and connectivity
            self.ecr_client.describe_repositories(maxResults=1)
            logger.debug("ECR connectivity verified")
            return True
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            logger.warning(f"ECR connectivity check failed: {error_code}")
            return False
        except Exception as e:
            logger.warning(f"ECR connectivity check failed: {e}")
            return False

    def verify_credentials(self) -> tuple[bool, Optional[str]]:
        """
        Verify AWS credentials are valid and have necessary ECR permissions.

        Returns:
            Tuple of (is_valid, error_message)
        """
        required_permissions = [
            "ecr:DescribeRepositories",
            "ecr:CreateRepository",
            "ecr:PutImage",
            "ecr:BatchCheckLayerAvailability",
            "ecr:InitiateLayerUpload",
            "ecr:UploadLayerPart",
            "ecr:CompleteLayerUpload",
        ]

        try:
            # Try to list repositories to verify read access
            self.ecr_client.describe_repositories(maxResults=1)

            # Try to get authorization token to verify push access
            self.ecr_client.get_authorization_token()

            logger.debug("AWS credentials verified with necessary permissions")
            return True, None
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            error_msg = e.response.get("Error", {}).get("Message", str(e))

            if error_code == "AccessDeniedException":
                return (
                    False,
                    f"AWS credentials lack required permissions: {', '.join(required_permissions)}",
                )
            return False, f"AWS credential verification failed: {error_msg}"
        except Exception as e:
            return False, f"Unexpected error during credential verification: {e}"

    def list_repositories(self) -> list[str]:
        """
        List all ECR repositories in the account.

        Returns:
            List of repository names
        """
        try:
            repositories = []
            paginator = self.ecr_client.get_paginator("describe_repositories")

            for page in paginator.paginate():
                for repo in page["repositories"]:
                    repositories.append(repo["repositoryName"])

            logger.debug(f"Found {len(repositories)} ECR repositories")
            return repositories
        except ClientError as e:
            logger.error(f"Failed to list ECR repositories: {e}")
            raise RuntimeError(f"Failed to list ECR repositories: {e}")

    def ensure_repositories(self, image_list: list[str]) -> list[str]:
        """
        Ensure ECR repositories exist for all images, creating if necessary.

        Args:
            image_list: List of image names (e.g., ["platform-api:v1.0.0", "postgres:14"])

        Returns:
            List of created repository names
        """
        existing_repos = set(self.list_repositories())
        created_repos = []

        for image in image_list:
            repo_name = self._extract_repo_name(image)

            if repo_name not in existing_repos:
                logger.info(f"Creating ECR repository: {repo_name}")
                try:
                    self._create_repository(repo_name)
                    created_repos.append(repo_name)
                    existing_repos.add(repo_name)
                except Exception as e:
                    logger.error(f"Failed to create repository {repo_name}: {e}")
                    raise
            else:
                logger.debug(f"Repository already exists: {repo_name}")

        if created_repos:
            logger.info(f"Created {len(created_repos)} new ECR repositories")

        return created_repos

    def _extract_repo_name(self, image: str) -> str:
        """
        Extract repository name from image string.

        Args:
            image: Image string (e.g., "platform-api:v1.0.0" or "org/platform-api:latest")

        Returns:
            Repository name without tag/digest
        """
        # Remove registry prefix if present
        if "/" in image and "." in image.split("/")[0]:
            image = "/".join(image.split("/")[1:])

        # Remove tag or digest
        if "@" in image:
            repo_name = image.split("@")[0]
        elif ":" in image:
            repo_name = image.split(":")[0]
        else:
            repo_name = image

        return repo_name

    def _create_repository(self, repo_name: str) -> None:
        """
        Create ECR repository with recommended settings for air-gapped deployments.

        Args:
            repo_name: Name of the repository to create

        Raises:
            RuntimeError: If repository creation fails (except if already exists)
        """
        try:
            self.ecr_client.create_repository(
                repositoryName=repo_name,
                imageTagMutability="IMMUTABLE",  # Prevent tag overwriting
                imageScanningConfiguration={"scanOnPush": True},  # Enable vulnerability scanning
                encryptionConfiguration={"encryptionType": "AES256"},  # Enable encryption
            )
            logger.info(f"Created ECR repository: {repo_name}")
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")

            # Repository already exists is not an error
            if error_code == "RepositoryAlreadyExistsException":
                logger.debug(f"Repository {repo_name} already exists")
                return

            logger.error(f"Failed to create repository {repo_name}: {e}")
            raise RuntimeError(f"Failed to create ECR repository: {e}")

    def delete_repository(self, repo_name: str, force: bool = False) -> None:
        """
        Delete ECR repository.

        Args:
            repo_name: Name of the repository to delete
            force: Force delete even if repository contains images

        Raises:
            RuntimeError: If deletion fails
        """
        try:
            self.ecr_client.delete_repository(repositoryName=repo_name, force=force)
            logger.info(f"Deleted ECR repository: {repo_name}")
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")

            if error_code == "RepositoryNotFoundException":
                logger.debug(f"Repository {repo_name} not found")
                return

            logger.error(f"Failed to delete repository {repo_name}: {e}")
            raise RuntimeError(f"Failed to delete ECR repository: {e}")

    def get_repository_uri(self, repo_name: str) -> str:
        """
        Get full ECR repository URI.

        Args:
            repo_name: Repository name

        Returns:
            Full repository URI (e.g., 123456.dkr.ecr.us-east-1.amazonaws.com/platform-api)
        """
        return f"{self.registry_url}/{repo_name}"
