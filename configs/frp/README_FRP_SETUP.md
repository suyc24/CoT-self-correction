# frp 内网穿透配置指南

通过 frp 从校外访问校园 GPU 服务器 (101.6.96.183)。

## 0. 购买 VPS

推荐：
- 阿里云轻量应用服务器（最低配 ~24元/月）
- 腾讯云轻量应用服务器（最低配 ~30元/月）
- 选 Ubuntu 22.04 或 Debian 12 即可
- 地域选北京/上海（离清华网络近延迟低）
- 需要在安全组/防火墙放行端口：7000（frp通信）、6022（SSH转发）

购买后记下公网 IP，下文用 `<VPS_IP>` 表示。

## 1. VPS 端配置（frps 服务端）

SSH 登录 VPS 后执行：

```bash
# 下载 frp（以 v0.61.1 为例，去 https://github.com/fatedier/frp/releases 查最新版）
cd ~
wget https://github.com/fatedier/frp/releases/download/v0.61.1/frp_0.61.1_linux_amd64.tar.gz
tar xzf frp_0.61.1_linux_amd64.tar.gz
mv frp_0.61.1_linux_amd64 frp
cd frp

# 使用准备好的配置（先把 frps.toml 传上来）
# 或者直接创建：
cat > frps.toml << 'EOF'
bindPort = 7000
auth.method = "token"
auth.token = "CHANGE_ME_TO_A_RANDOM_STRING"
EOF

# 启动（用 nohup 或 systemd）
nohup ./frps -c frps.toml > frps.log 2>&1 &

# 验证
ss -tlnp | grep 7000
```

如果有 sudo 权限可以配 systemd 开机自启（见下方 systemd 配置）。

## 2. GPU 服务器端配置（frpc 客户端）

SSH 登录 GPU 服务器后执行（不需要 sudo）：

```bash
# 下载 frp 到用户目录
cd ~
wget https://github.com/fatedier/frp/releases/download/v0.61.1/frp_0.61.1_linux_amd64.tar.gz
tar xzf frp_0.61.1_linux_amd64.tar.gz
mv frp_0.61.1_linux_amd64 frp
cd frp

# 使用准备好的配置（先把 frpc.toml 传上来）
# 或者直接创建（把 <VPS_IP> 和 token 替换成你的）：
cat > frpc.toml << 'EOF'
serverAddr = "<VPS_IP>"
serverPort = 7000
auth.method = "token"
auth.token = "CHANGE_ME_TO_A_RANDOM_STRING"

[[proxies]]
name = "gpu-ssh"
type = "tcp"
localIP = "127.0.0.1"
localPort = 22
remotePort = 6022
EOF

# 在 tmux 中启动（保持长期运行）
tmux new-session -d -s frpc './frpc -c frpc.toml'

# 验证
tmux attach -t frpc
# 应该看到 "start proxy success" 字样
# Ctrl+B, D 退出 tmux
```

注意：localPort 可能需要改。先在 GPU 服务器上确认本地 SSH 端口：
```bash
ss -tlnp | grep ssh
# 如果输出显示 *:22 就用 22
# 如果显示 *:8002 就用 8002
```

## 3. 从校外连接

```bash
ssh -p 6022 yucheng@<VPS_IP>
```

如果要用 rsync：
```bash
rsync -avz -e 'ssh -p 6022' local_file yucheng@<VPS_IP>:/home/yucheng/experiment/Qwen2.5-Math/
```

## 4. 自动重启（crontab，不需要 sudo）

在 GPU 服务器上配置 crontab 保活：

```bash
crontab -e
# 添加以下行（每 5 分钟检查 frpc 是否在运行，不在就启动）：
*/5 * * * * pgrep -f 'frpc -c' > /dev/null || cd ~/frp && tmux new-session -d -s frpc './frpc -c frpc.toml'
```

## 5. VPS 端 systemd 自启（可选，需要 VPS 的 sudo）

```bash
sudo cat > /etc/systemd/system/frps.service << 'EOF'
[Unit]
Description=frp server
After=network.target

[Service]
Type=simple
ExecStart=/home/<vps_user>/frp/frps -c /home/<vps_user>/frp/frps.toml
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable --now frps
```

## 安全建议

- auth.token 使用随机长字符串（如 `openssl rand -hex 16` 生成）
- VPS 安全组只开必要端口（22、7000、6022）
- 考虑 VPS 上的 SSH 也配置密钥登录、禁用密码登录
