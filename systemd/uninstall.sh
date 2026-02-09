#!/bin/bash
# Chalna systemd 서비스 제거 스크립트

set -e

echo "=== Chalna 서비스 제거 ==="

# 서비스 중지
sudo systemctl stop chalna-api.service 2>/dev/null || true

# 서비스 비활성화
sudo systemctl disable chalna-api.service 2>/dev/null || true

# 서비스 파일 삭제
sudo rm -f /etc/systemd/system/chalna-api.service

# 레거시 web 서비스 정리 (있으면 제거)
sudo systemctl stop chalna-web.service 2>/dev/null || true
sudo systemctl disable chalna-web.service 2>/dev/null || true
sudo rm -f /etc/systemd/system/chalna-web.service

# systemd 리로드
sudo systemctl daemon-reload

echo "=== 제거 완료 ==="
