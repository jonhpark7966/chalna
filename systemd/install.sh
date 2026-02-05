#!/bin/bash
# Chalna systemd 서비스 설치 스크립트

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Chalna 서비스 설치 ==="

# 서비스 파일 복사
sudo cp "$SCRIPT_DIR/chalna-web.service" /etc/systemd/system/
sudo cp "$SCRIPT_DIR/chalna-api.service" /etc/systemd/system/

# systemd 리로드
sudo systemctl daemon-reload

# 서비스 활성화 (부팅 시 자동 시작)
sudo systemctl enable chalna-web.service
sudo systemctl enable chalna-api.service

# 서비스 시작
sudo systemctl start chalna-web.service
sudo systemctl start chalna-api.service

echo ""
echo "=== 설치 완료 ==="
echo "  Web UI:   http://0.0.0.0:7860"
echo "  REST API: http://0.0.0.0:7861"
echo ""
echo "서비스 상태 확인:"
echo "  sudo systemctl status chalna-web"
echo "  sudo systemctl status chalna-api"
echo ""
echo "로그 확인:"
echo "  journalctl -u chalna-web -f"
echo "  journalctl -u chalna-api -f"
