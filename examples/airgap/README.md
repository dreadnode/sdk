# Dreadnode Air-Gapped Deployment

This directory contains the Zarf package configuration for air-gapped deployment of the Dreadnode platform.

## What is this?

Zarf packages take an existing working Helm chart and bundle it with all container images for deployment in environments without internet access. This configuration packages the complete Dreadnode platform including:

- **PostgreSQL** - Primary database
- **ClickHouse** - Analytics database
- **MinIO** - Object storage (S3-compatible)
- **Platform API** - Backend service
- **Platform UI** - Frontend service
- **DynamoDB Local** - For Crucible mode (optional)

## Prerequisites

### On your build machine (connected environment):
- [Zarf](https://zarf.dev) CLI installed
- [Helm](https://helm.sh) CLI installed
- Docker running (for pulling images)
- Access to container registries (GHCR, Docker Hub)

### On your deployment machine (air-gapped environment):
- Kubernetes cluster
- Zarf CLI installed
- The Zarf package file (transferred via USB, etc.)

## Directory Structure

```
examples/airgap/
├── README.md           # This file
└── zarf/
    ├── zarf.yaml       # Zarf package configuration
    └── values-airgap.yaml  # Helm values for air-gapped deployment
```

## Quick Start

### 1. Clone repositories side by side

```bash
cd /path/to/workspace
git clone https://github.com/dreadnode/platform-charts.git
git clone https://github.com/dreadnode/sdk.git
```

Your structure should look like:
```
/path/to/workspace/
├── platform-charts/
└── sdk/
```

### 2. Build Helm chart dependencies

```bash
cd platform-charts/platform
helm dependency build
```

### 3. Create Zarf package (connected environment)

```bash
cd sdk/examples/airgap/zarf
zarf package create . --set VERSION=0.6.10 --confirm
```

This will:
- Pull all container images from registries
- Download the Helm chart
- Generate SBOMs for all images
- Create a `.tar.zst` package (typically 2-5GB)

### 4. Transfer package to air-gapped environment

```bash
# Copy the package file to transfer media
cp zarf-package-dreadnode-platform-*.tar.zst /media/usb/
```

### 5. Deploy package (air-gapped environment)

```bash
# Initialize Zarf in the cluster (first time only)
zarf init --confirm

# Deploy the platform
zarf package deploy zarf-package-dreadnode-platform-*.tar.zst --confirm
```

### 6. Verify deployment

```bash
kubectl get pods -n dreadnode
```

All pods should be in `Running` state.

## Customization

### Changing the namespace

```bash
zarf package deploy ... --set NAMESPACE=my-namespace
```

### Using custom values

Edit `values-airgap.yaml` before creating the package to customize:
- Resource limits
- Storage sizes
- Feature flags
- Credentials (for production, use secrets management)

## Troubleshooting

### Pods not starting

Check pod events:
```bash
kubectl describe pod -n dreadnode <pod-name>
kubectl logs -n dreadnode <pod-name>
```

### Image pull failures

Zarf should handle all image references automatically. If you see pull errors:
```bash
# Check Zarf registry is running
kubectl get pods -n zarf

# Check image references were rewritten
kubectl get pod -n dreadnode <pod-name> -o jsonpath='{.spec.containers[*].image}'
```

### Database connection failures

Ensure PostgreSQL and ClickHouse are running before the API starts:
```bash
kubectl get pods -n dreadnode -l app.kubernetes.io/name=postgresql
kubectl get pods -n dreadnode -l app.kubernetes.io/name=clickhouse
```

## Advanced Usage

### Inspect package contents

```bash
zarf package inspect zarf-package-dreadnode-platform-*.tar.zst
```

### Extract SBOMs

```bash
zarf package inspect zarf-package-dreadnode-platform-*.tar.zst --sbom-out ./sboms
```

### Deploy specific components

```bash
zarf package deploy ... --components=platform
```

## Using with ECR (AWS)

For AWS environments with ECR:

```bash
# Authenticate to ECR
aws ecr get-login-password --region us-west-2 | \
  docker login --username AWS --password-stdin \
  <account-id>.dkr.ecr.us-west-2.amazonaws.com

# Create package with ECR registry
zarf package create . --registry-override ghcr.io=<account-id>.dkr.ecr.us-west-2.amazonaws.com --confirm
```

## Security Considerations

- Packages are cryptographically signed
- SBOMs are generated for all images
- Review SBOMs before deployment for compliance
- Change default passwords in production (see values-airgap.yaml)

## Support

- GitHub Issues: https://github.com/dreadnode/sdk/issues
- Documentation: https://docs.dreadnode.io
