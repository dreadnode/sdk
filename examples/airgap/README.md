# Dreadnode Air-Gapped Deployment Example

This directory contains an example configuration for packaging and deploying the Dreadnode platform in air-gapped environments.

## Prerequisites

### Tools Required

- **Zarf**: Download from [zarf.dev](https://zarf.dev)
- **kubectl**: Kubernetes command-line tool
- **AWS CLI**: For ECR authentication (optional, can use environment variables)

### Environment Requirements

- Kubernetes cluster (EKS, K3s, or other distribution)
- AWS ECR registry access
- At least 20GB free disk space
- Valid AWS credentials with ECR permissions

## Directory Structure

```
examples/airgap/
├── README.md                    # This file
├── zarf/
│   ├── zarf.yaml               # Zarf package configuration
│   ├── values-airgap.yaml      # Helm values for air-gapped deployment
│   ├── postgresql-values.yaml  # PostgreSQL configuration
│   ├── redis-values.yaml       # Redis configuration
│   └── charts/                 # Local Helm charts (if any)
└── docs/
    └── troubleshooting.md      # Common issues and solutions
```

## Quick Start

### 1. Create Package (Connected Environment)

From a machine with internet access:

```bash
# Navigate to SDK root
cd /path/to/dreadnode-sdk

# Create air-gapped package
dreadnode airgap package-create \
  --version v1.0.0 \
  --source-dir examples/airgap/zarf \
  --output-dir ./packages
```

This will:
- Pull all container images from GHCR and other registries
- Download all Helm charts
- Generate SBOMs for all images
- Create a signed `.tar.zst` bundle (typically 2-5GB)

### 2. Transfer Package

Transfer the generated package to your air-gapped environment:

```bash
# Example using SCP
scp packages/dreadnode-platform-*.tar.zst user@airgapped-host:/tmp/

# Or use removable media, sneakernet, etc.
```

### 3. Install Package (Air-Gapped Environment)

On the air-gapped machine with access to your Kubernetes cluster:

```bash
# Set your ECR registry URL
ECR_REGISTRY="123456789.dkr.ecr.us-east-1.amazonaws.com"

# Install platform
dreadnode airgap install \
  --bundle /tmp/dreadnode-platform-v1.0.0.tar.zst \
  --ecr-registry $ECR_REGISTRY \
  --namespace dreadnode
```

This will:
- Validate environment and prerequisites
- Extract images from bundle
- Push images to your ECR registry
- Deploy Helm charts with registry overrides
- Verify deployment health

## Configuration

### Customizing the Package

Edit `zarf/zarf.yaml` to customize:

- **Components**: Add/remove components (monitoring, utilities, etc.)
- **Images**: Specify additional images to include
- **Variables**: Set default values for deployment
- **Actions**: Add pre/post deployment scripts

### Customizing Deployment

Create custom Helm values files:

```yaml
# zarf/values-airgap.yaml
replicaCount: 3

resources:
  limits:
    cpu: 2000m
    memory: 4Gi

ingress:
  enabled: true
  className: nginx
  hosts:
    - host: dreadnode.example.com
      paths:
        - path: /
          pathType: Prefix
```

## Advanced Usage

### Inspect Package Contents

```bash
dreadnode airgap package-inspect \
  --bundle packages/dreadnode-platform-v1.0.0.tar.zst \
  --sbom-out ./sboms
```

### Validate Environment Before Install

```bash
dreadnode airgap validate \
  --bundle /tmp/dreadnode-platform-v1.0.0.tar.zst \
  --ecr-registry $ECR_REGISTRY
```

### Health Check After Install

```bash
dreadnode airgap health-check \
  --namespace dreadnode \
  --timeout 600
```

### Install Specific Components

```bash
# Install only core platform, skip monitoring
zarf package deploy \
  --components=platform-infrastructure,platform-dependencies \
  dreadnode-platform-v1.0.0.tar.zst
```

## Troubleshooting

### Common Issues

#### 1. "Zarf binary not found"

Install Zarf:
```bash
# Linux/macOS
brew install defenseunicorns/tap/zarf

# Or download directly
curl -sL https://zarf.dev/install.sh | sh
```

#### 2. "Cannot connect to ECR"

Ensure AWS credentials are configured:
```bash
aws configure
# or
export AWS_ACCESS_KEY_ID=xxx
export AWS_SECRET_ACCESS_KEY=xxx
```

#### 3. "Pods not becoming ready"

Check pod logs:
```bash
kubectl get pods -n dreadnode
kubectl logs -n dreadnode <pod-name>
kubectl describe pod -n dreadnode <pod-name>
```

#### 4. "ECR repository does not exist"

The SDK automatically creates repositories. If this fails, create manually:
```bash
aws ecr create-repository --repository-name platform-api
aws ecr create-repository --repository-name platform-ui
# etc.
```

### Getting Help

- Check logs: `kubectl logs -n dreadnode <pod-name>`
- View events: `kubectl get events -n dreadnode --sort-by='.lastTimestamp'`
- Zarf logs: Stored in `~/.zarf-logs/`

## Security Considerations

### Package Verification

Always verify package signatures:
```bash
# Verify with Cosign (Zarf does this automatically)
zarf package inspect dreadnode-platform-v1.0.0.tar.zst
```

### SBOM Analysis

Review SBOMs for vulnerabilities:
```bash
# Extract SBOMs
dreadnode airgap package-inspect \
  --bundle dreadnode-platform-v1.0.0.tar.zst \
  --sbom-out ./sboms

# Scan with Grype or similar
grype sbom:./sboms/platform-api.json
```

### ECR Security

Enable recommended ECR features:
- Image tag immutability (prevents tag overwriting)
- Scan on push (automatic vulnerability scanning)
- Encryption at rest (AES256)

These are automatically enabled by the SDK when creating repositories.

## Production Deployment Checklist

- [ ] Package created and verified in connected environment
- [ ] Package transferred securely to air-gapped environment
- [ ] ECR registry configured with proper access controls
- [ ] Kubernetes cluster has sufficient resources
- [ ] Storage class available for persistent volumes
- [ ] Network policies configured (if required)
- [ ] Backup strategy in place
- [ ] Monitoring and alerting configured
- [ ] Disaster recovery plan documented
- [ ] Team trained on troubleshooting procedures

## Next Steps

1. Customize `zarf.yaml` for your requirements
2. Create package in connected environment
3. Test deployment in staging air-gapped environment
4. Document any custom configurations
5. Deploy to production

## Support

For issues or questions:
- GitHub Issues: https://github.com/dreadnode/sdk/issues
- Documentation: https://docs.dreadnode.io
- Email: support@dreadnode.io
