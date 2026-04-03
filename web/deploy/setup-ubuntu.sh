#!/bin/bash
# Ubuntu server setup for LLM-Bench quiz frontend
#
# Run as root or with sudo:
#   sudo bash web/deploy/setup-ubuntu.sh
#
# This script:
# 1. Clones the repo to /opt/llm-bench
# 2. Creates a Python venv with Flask + gunicorn
# 3. Installs the systemd service
# 4. Installs the nginx config
# 5. Sets up a cron job to git pull every 5 minutes (sync results from aroma)

set -e

REPO_URL="${1:-git@github.com:SimonMcCallum/LLM-Bench.git}"
INSTALL_DIR="/opt/llm-bench"
DOMAIN="${2:-bench.simonmccallum.org.nz}"

echo "============================================"
echo "LLM-Bench Ubuntu Server Setup"
echo "  Repo:    $REPO_URL"
echo "  Install: $INSTALL_DIR"
echo "  Domain:  $DOMAIN"
echo "============================================"

# 1. Clone repo
if [ -d "$INSTALL_DIR" ]; then
    echo "Updating existing installation..."
    cd "$INSTALL_DIR"
    git pull
else
    echo "Cloning repository..."
    git clone "$REPO_URL" "$INSTALL_DIR"
    cd "$INSTALL_DIR"
fi

# 2. Python venv
echo "Setting up Python virtual environment..."
python3 -m venv venv
venv/bin/pip install --upgrade pip
venv/bin/pip install flask gunicorn

# 3. Systemd service
echo "Installing systemd service..."
cp web/deploy/llm-bench.service /etc/systemd/system/llm-bench.service
systemctl daemon-reload
systemctl enable llm-bench
systemctl restart llm-bench

# 4. Nginx config
echo "Installing nginx config..."
cp web/deploy/nginx.conf /etc/nginx/sites-available/llm-bench
sed -i "s/llm-bench.example.com/$DOMAIN/g" /etc/nginx/sites-available/llm-bench
sed -i "s|/opt/llm-bench|$INSTALL_DIR|g" /etc/nginx/sites-available/llm-bench
ln -sf /etc/nginx/sites-available/llm-bench /etc/nginx/sites-enabled/
nginx -t && systemctl reload nginx

# 5. Cron job: git pull every 5 minutes to sync results from aroma
echo "Setting up sync cron job..."
CRON_CMD="cd $INSTALL_DIR && git pull --rebase origin main 2>&1 | logger -t llm-bench-sync"
(crontab -l 2>/dev/null | grep -v "llm-bench-sync"; echo "*/5 * * * * $CRON_CMD") | crontab -

# 6. Set permissions
chown -R www-data:www-data "$INSTALL_DIR/results/"

echo ""
echo "============================================"
echo "Setup complete!"
echo ""
echo "  Quiz server: http://$DOMAIN"
echo "  Service:     systemctl status llm-bench"
echo "  Logs:        journalctl -u llm-bench -f"
echo "  Sync logs:   journalctl -t llm-bench-sync"
echo ""
echo "Next steps:"
echo "  1. Set up SSL:  sudo certbot --nginx -d $DOMAIN"
echo "  2. On aroma:    python web/export_questions.py"
echo "  3. On aroma:    git add results/questions/ && git push"
echo "============================================"
