#!/bin/bash
# ARES Edge System - Production Installation Script
# Classification: UNCLASSIFIED // FOR OFFICIAL USE ONLY
# Copyright (c) 2024 DELFICTUS I/O LLC

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
ARES_VERSION="1.0.0"
ARES_HOME="/opt/ares"
ARES_USER="ares"
ARES_GROUP="ares"
LOG_FILE="/tmp/ares-install.log"

# Functions
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}" | tee -a "$LOG_FILE"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}" | tee -a "$LOG_FILE"
    exit 1
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}" | tee -a "$LOG_FILE"
}

check_root() {
    if [[ $EUID -ne 0 ]]; then
        error "This script must be run as root (use sudo)"
    fi
}

check_os() {
    log "Checking operating system compatibility..."
    
    if [[ -f /etc/os-release ]]; then
        source /etc/os-release
        case "$ID" in
            ubuntu)
                if [[ "${VERSION_ID}" < "20.04" ]]; then
                    error "Ubuntu 20.04 LTS or newer required"
                fi
                OS_TYPE="ubuntu"
                ;;
            rhel|centos|rocky|almalinux)
                if [[ "${VERSION_ID%%.*}" -lt 8 ]]; then
                    error "RHEL 8 or newer required"
                fi
                OS_TYPE="rhel"
                ;;
            *)
                warn "Unsupported OS: $ID. Proceeding with Ubuntu/Debian assumptions."
                OS_TYPE="ubuntu"
                ;;
        esac
    else
        error "Cannot determine operating system"
    fi
    
    log "OS compatibility check passed: $ID $VERSION_ID"
}

check_hardware() {
    log "Checking hardware requirements..."
    
    # Check CPU cores
    CPU_CORES=$(nproc)
    if [[ $CPU_CORES -lt 8 ]]; then
        warn "Minimum 8 CPU cores recommended (found: $CPU_CORES)"
    fi
    
    # Check memory
    MEMORY_GB=$(free -g | awk '/^Mem:/{print $2}')
    if [[ $MEMORY_GB -lt 16 ]]; then
        warn "Minimum 16GB RAM recommended (found: ${MEMORY_GB}GB)"
    fi
    
    # Check disk space
    DISK_SPACE=$(df / | awk 'NR==2{print $4}')
    DISK_SPACE_GB=$((DISK_SPACE / 1024 / 1024))
    if [[ $DISK_SPACE_GB -lt 100 ]]; then
        warn "Minimum 100GB free space recommended (found: ${DISK_SPACE_GB}GB)"
    fi
    
    # Check for AVX2 support
    if ! grep -q avx2 /proc/cpuinfo; then
        warn "AVX2 instruction set not detected. Performance may be reduced."
    fi
    
    # Check for NVIDIA GPU (optional)
    if command -v nvidia-smi >/dev/null 2>&1; then
        GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null || echo "")
        if [[ -n "$GPU_INFO" ]]; then
            log "NVIDIA GPU detected: $GPU_INFO"
        fi
    else
        info "No NVIDIA GPU detected. CPU-only mode will be used."
    fi
    
    log "Hardware check completed"
}

install_system_dependencies() {
    log "Installing system dependencies..."
    
    case "$OS_TYPE" in
        ubuntu)
            apt-get update
            apt-get install -y \
                build-essential \
                cmake \
                git \
                python3 \
                python3-pip \
                python3-venv \
                python3-dev \
                libomp-dev \
                libssl-dev \
                libffi-dev \
                pkg-config \
                curl \
                wget \
                htop \
                rsync \
                unzip
            ;;
        rhel)
            yum groupinstall -y "Development Tools"
            yum install -y \
                cmake \
                git \
                python3 \
                python3-pip \
                python3-devel \
                openssl-devel \
                libffi-devel \
                pkgconfig \
                curl \
                wget \
                htop \
                rsync \
                unzip
            ;;
    esac
    
    log "System dependencies installed"
}

install_cuda() {
    log "Checking for CUDA installation..."
    
    if command -v nvcc >/dev/null 2>&1; then
        CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9.]*\).*/\1/')
        log "CUDA $CUDA_VERSION already installed"
        return 0
    fi
    
    if [[ "$1" == "--skip-cuda" ]]; then
        info "Skipping CUDA installation as requested"
        return 0
    fi
    
    read -p "CUDA not found. Install CUDA 11.8? [y/N]: " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        info "Skipping CUDA installation"
        return 0
    fi
    
    case "$OS_TYPE" in
        ubuntu)
            wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
            dpkg -i cuda-keyring_1.0-1_all.deb
            apt-get update
            apt-get install -y cuda-11-8
            ;;
        rhel)
            yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo
            yum install -y cuda-11-8
            ;;
    esac
    
    echo 'export PATH=/usr/local/cuda/bin:$PATH' >> /etc/environment
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> /etc/environment
    
    log "CUDA installation completed"
}

create_user() {
    log "Creating ARES system user..."
    
    if ! id "$ARES_USER" >/dev/null 2>&1; then
        groupadd -r "$ARES_GROUP"
        useradd -r -g "$ARES_GROUP" -d "$ARES_HOME" -s /bin/bash -c "ARES Edge System" "$ARES_USER"
        log "Created user: $ARES_USER"
    else
        log "User $ARES_USER already exists"
    fi
}

install_ares() {
    log "Installing ARES Edge System..."
    
    # Create directories
    mkdir -p "$ARES_HOME"/{bin,config,logs,data,certs}
    mkdir -p /var/log/ares
    
    # Copy source code
    if [[ -d "$(pwd)/ares_unified" ]]; then
        cp -r "$(pwd)/ares_unified" "$ARES_HOME/"
        cp "$(pwd)/README.md" "$ARES_HOME/"
        cp "$(pwd)/requirements.txt" "$ARES_HOME/"
        cp "$(pwd)/setup.py" "$ARES_HOME/"
        cp "$(pwd)/SBOM.spdx" "$ARES_HOME/"
    else
        error "ARES source code not found. Run this script from the repository root."
    fi
    
    # Set up Python virtual environment
    python3 -m venv "$ARES_HOME/venv"
    source "$ARES_HOME/venv/bin/activate"
    
    # Install Python dependencies
    pip install --upgrade pip wheel setuptools
    pip install -r "$ARES_HOME/requirements.txt"
    pip install -e "$ARES_HOME"
    
    # Build C++/CUDA components
    mkdir -p "$ARES_HOME/build"
    cd "$ARES_HOME/build"
    cmake ../ares_unified -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$ARES_HOME"
    make -j"$(nproc)"
    make install
    
    # Set up configuration
    cp "$ARES_HOME/ares_unified/config/config.yaml" "$ARES_HOME/config/production.yaml"
    
    # Set permissions
    chown -R "$ARES_USER:$ARES_GROUP" "$ARES_HOME"
    chown -R "$ARES_USER:$ARES_GROUP" /var/log/ares
    chmod -R 750 "$ARES_HOME"
    chmod -R 640 "$ARES_HOME/config"
    
    log "ARES installation completed"
}

create_systemd_service() {
    log "Creating systemd service..."
    
    cat > /etc/systemd/system/ares.service << EOF
[Unit]
Description=ARES Edge System
After=network.target
Wants=network.target

[Service]
Type=forking
User=$ARES_USER
Group=$ARES_GROUP
WorkingDirectory=$ARES_HOME
Environment=PATH=$ARES_HOME/venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
Environment=LD_LIBRARY_PATH=/usr/local/cuda/lib64
ExecStart=$ARES_HOME/venv/bin/python -m ares.main --config $ARES_HOME/config/production.yaml --daemon
ExecStop=/bin/kill -TERM \$MAINPID
Restart=always
RestartSec=10
LimitNOFILE=65536
LimitNPROC=32768

[Install]
WantedBy=multi-user.target
EOF
    
    systemctl daemon-reload
    systemctl enable ares.service
    
    log "Systemd service created and enabled"
}

setup_logging() {
    log "Setting up logging configuration..."
    
    # Create rsyslog configuration for ARES
    cat > /etc/rsyslog.d/50-ares.conf << EOF
# ARES Edge System logging
:programname, isequal, "ares" /var/log/ares/system.log
:programname, isequal, "ares" stop
EOF
    
    # Create logrotate configuration
    cat > /etc/logrotate.d/ares << EOF
/var/log/ares/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    copytruncate
    su $ARES_USER $ARES_GROUP
}
EOF
    
    systemctl restart rsyslog
    
    log "Logging configuration completed"
}

run_tests() {
    log "Running installation verification tests..."
    
    cd "$ARES_HOME"
    source venv/bin/activate
    
    # Test Python installation
    python -c "import ares; print('Python components: OK')" || error "Python import failed"
    
    # Test C++ components
    if [[ -f "$ARES_HOME/bin/test_cew_unified" ]]; then
        "$ARES_HOME/bin/test_cew_unified" || warn "C++ component test failed"
    fi
    
    # Test configuration
    python -m ares.config --validate "$ARES_HOME/config/production.yaml" || error "Configuration validation failed"
    
    log "Installation verification completed"
}

generate_certificates() {
    log "Generating self-signed certificates for testing..."
    
    cd "$ARES_HOME/certs"
    
    # Generate CA key and certificate
    openssl genrsa -out ca.key 4096
    openssl req -new -x509 -days 365 -key ca.key -out ca.crt -subj "/CN=ARES-CA/O=DELFICTUS/C=US"
    
    # Generate server key and certificate
    openssl genrsa -out ares.key 4096
    openssl req -new -key ares.key -out ares.csr -subj "/CN=ares.local/O=DELFICTUS/C=US"
    openssl x509 -req -days 365 -in ares.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out ares.crt
    
    # Set permissions
    chmod 600 *.key
    chmod 644 *.crt
    chown "$ARES_USER:$ARES_GROUP" *
    
    rm ares.csr
    
    log "Certificates generated (replace with production certificates before deployment)"
}

main() {
    echo -e "${GREEN}"
    echo "=================================================="
    echo "ARES Edge System - Production Installation"
    echo "Version: $ARES_VERSION"
    echo "Classification: UNCLASSIFIED // FOUO"
    echo "=================================================="
    echo -e "${NC}"
    
    log "Starting ARES installation..."
    
    check_root
    check_os
    check_hardware
    install_system_dependencies
    
    # Check for --skip-cuda flag
    if [[ "${1:-}" != "--skip-cuda" ]]; then
        install_cuda "$@"
    else
        install_cuda --skip-cuda
    fi
    
    create_user
    install_ares
    create_systemd_service
    setup_logging
    generate_certificates
    run_tests
    
    echo -e "${GREEN}"
    echo "=================================================="
    echo "ARES Edge System Installation Complete!"
    echo "=================================================="
    echo -e "${NC}"
    
    log "Installation completed successfully"
    
    echo
    echo "Next steps:"
    echo "1. Review configuration: $ARES_HOME/config/production.yaml"
    echo "2. Replace test certificates with production certificates"
    echo "3. Start the service: systemctl start ares"
    echo "4. Check status: systemctl status ares"
    echo "5. View logs: tail -f /var/log/ares/system.log"
    echo
    echo "Documentation: $ARES_HOME/README.md"
    echo "Support: contact@delfictus.io"
}

# Handle command line arguments
case "${1:-}" in
    --help|-h)
        echo "ARES Edge System Installation Script"
        echo "Usage: $0 [--skip-cuda] [--help]"
        echo "  --skip-cuda    Skip CUDA installation"
        echo "  --help         Show this help message"
        exit 0
        ;;
    *)
        main "$@"
        ;;
esac