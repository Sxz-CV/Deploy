# Smoke — YOLO26 烟雾/火焰实例分割 Jetson 部署

YOLO26s-seg 实例分割模型，检测 `fire` / `smoke` 两类目标，输入尺寸 640。
本文档记录：Jetson 板端部署全流程、TensorRT FP16 量化、Git 仓库上传过程与踩坑记录。

---

## 1. 项目结构

```
Deploy/
├── README.md
├── train.py                 # 训练入口（YOLO26s-seg，640，100 epoch）
├── predict.py               # 推理脚本（.pt 直接推理）
├── export_onnx.py           # 导出 ONNX（opset=12，imgsz=640）
├── data.yaml                # 数据集配置（nc=2：fire / smoke）
├── config/args.yaml         # 训练参数快照
├── .gitignore               # 排除大文件
├── dataset/                 # 数据集（不进 Git）
│   ├── images/train
│   ├── images/val
│   └── images/test
├── weights/                 # 权重（不进 Git，走 Releases）
│   ├── best.pt              # 22.3 MB
│   ├── last.pt              # 22.3 MB
│   └── best.onnx            # 39.8 MB
├── result.png               # FP16 engine 的 trtexec 基准测试截图
└── 过程.mp4                  # 演示视频（78 MB，不进 Git）
```

训练核心参数（`train.py`）：

| 参数 | 值 |
|---|---|
| 基础模型 | `yolo26s-seg.pt` |
| imgsz | 640 |
| epochs | 100 |
| batch | 8 |
| patience | 35 |
| lr0 | 0.001 |
| mosaic / copy_paste | 0.5 / 0.3 |
| amp | True |

---

## 2. 环境信息

| 项目 | 值 |
|---|---|
| 开发板 | Jetson TX2 系列（NVIDIA P3450） |
| 系统 | Ubuntu / R35.4.1 = JetPack 5.1.2 |
| 板端 Python | 3.8 |
| 板端预装依赖 | numpy 1.24.4 / OpenCV 4.10.0 / PIL 7.0.0 |
| TensorRT | **8.5.2**（v8502） |
| 板端 IP | 10.223.2.66（DHCP 动态，约 24 小时可能变） |
| 板端用户名 | mima123456 |
| 板端主机名 | houhoudesktop |
| PC 端 | Windows 11 + PowerShell 5 |

**窗口区分（最容易搞混的地方）**：

- `PS C:\Users\ASUS>` → PC 本地窗口，用来 `ssh` / `scp` / `git`
- `mima123456@houhoudesktop:~$` → 开发板窗口，用来跑 `trtexec` / `python3`

---

## 3. 板端部署操作（2026-11-25）

### 3.1 确认板子 IP 与 SSH 服务

板子上（接显示器键盘，`Ctrl+Alt+T`）：

```bash
whoami                    # mima123456
hostname -I               # 记下 IP，例如 10.223.2.66
systemctl status ssh --no-pager    # 必须 active (running)
ss -tlnp | grep :22                # 看到 LISTEN 0.0.0.0:22
```

若 SSH 未安装：

```bash
sudo apt update
sudo apt install -y openssh-server
sudo systemctl enable --now ssh
```

### 3.2 PC 端登录

```powershell
ipconfig                  # 本机 IPv4 必须是 10.223.2.x 同网段
ping 10.223.2.66
ssh mima123456@10.223.2.66
```

首次连接输入 `yes` 接受指纹，再输密码（输入时屏幕不显字符，正常）。

### 3.3 免密登录（可选）

```powershell
ssh-keygen -t rsa
type $env:USERPROFILE\.ssh\id_rsa.pub | ssh mima123456@10.223.2.66 "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"
```

### 3.4 传输文件到板子

在 **PC 本地窗口** 执行：

```powershell
ssh mima123456@10.223.2.66 "mkdir -p ~/smoke"
scp E:\desktop\deploy\weights\best.onnx mima123456@10.223.2.66:~/smoke/
scp E:\desktop\deploy\weights\best.pt   mima123456@10.223.2.66:~/smoke/
scp E:\desktop\deploy\data.yaml         mima123456@10.223.2.66:~/smoke/
```

> `scp` 必须在 PC 本地窗口跑，SSH 会话里看不到 PC 的 E 盘。

### 3.5 板端内存不足时加 swap

TX2 内存只有 8 GB，构建 engine 容易 OOM：

```bash
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
free -h
```

### 3.6 用 trtexec 构建 FP16 engine

板子上执行（不要加 `| tail`，否则看不到进度，看起来像卡死）：

```bash
cd ~/smoke
/usr/src/tensorrt/bin/trtexec --onnx=best.onnx --fp16 \
  --saveEngine=best_fp16.engine --workspace=1024
```

- TX2 构建 YOLO 分割模型的 FP16 engine 需要 **5~15 分钟**，CPU 满载属正常
- 构建期间**不会**生成目标文件，全部完成才写入；`ls` 看不到 engine 不代表失败
- 成功标志：输出末尾出现 `&&&& PASSED TensorRT.trtexec` 和 `Engine saved`

确认进程还活着：

```bash
ps aux | grep trtexec      # 状态 Rl+ 且 CPU 高占用 = 正在工作
tegrastats                 # 观察 CPU / GPU / 内存
```

### 3.7 验证产物

```bash
ls -lh ~/smoke/best_fp16.engine
```

---

## 4. 指标与数值

数据来源：板端 `trtexec --onnx=best.onnx --fp16 --saveEngine=best_fp16.engine --workspace=1024` 自带的性能汇总（见仓库 `result.png`）。

| 指标 | 数值 |
|---|---|
| **吞吐量（Throughput）** | **46.5971 qps** |
| 总延迟 Latitude mean | 22.055 ms |
| 总延迟 median | 22.0496 ms |
| 总延迟 min / max | 21.8269 / 22.2291 ms |
| 总延迟 p90 / p95 / p99 | 22.1316 / 22.1754 / 22.2144 ms |
| **GPU Compute mean** | **21.31 ms** |
| GPU Compute median | 21.3126 ms |
| GPU Compute min / max | 21.1482 / 21.4486 ms |
| Enqueue Time mean | 4.94312 ms |
| H2D Latency mean | 0.374133 ms |
| D2H Latency mean | 0.367688 ms |
| Total Host Walltime | 3.06886 s |
| Total GPU Compute Time | 3.04778 s |

**FP32 / FP16 对比（`--loadEngine`，200 次迭代）**

| 精度 | GPU Compute 均值 | GPU Compute 中位数 | 端到端延迟中位数 | 吞吐 | 相对加速 |
|---|---|---|---|---|---|
| FP32 | 58.35 ms | 58.71 ms | 59.65 ms | 16.82 qps | 1.00x |
| FP16 | 22.54 ms | 21.32 ms | 22.05 ms | 43.91 qps | **2.75x** |

FP16 相对 FP32：延迟下降约 62%，吞吐提升约 2.6 倍。
FP16 组 p99 出现一次约 90 ms 的尖峰，导致均值偏高；中位数 21.32 ms 更能代表稳态性能。

---

## 5. 上传 Git 仓库的命令（2026-11-26）

### 5.1 仓库信息

| 项 | 值 |
|---|---|
| 仓库地址 | https://github.com/Sxz-CV/Deploy |
| 默认分支 | main |
| 大文件存放 | Releases（tag `v1.0`） |

### 5.2 完整上传命令（最终可用版）

```powershell
cd "E:\desktop\deploy"

# 1) 首次配置身份（邮箱必须是 GitHub 账号绑定的真实邮箱）
git config --global user.name "Sxz-CV"
git config --global user.email "3473884372@qq.com"

# 2) 初始化仓库
git init
git add .
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/Sxz-CV/Deploy.git

# 3) 走代理推送（Clash Verge 实际端口是 7897，不是默认 7890）
git config --global http.proxy http://127.0.0.1:7897
git config --global https.proxy http://127.0.0.1:7897

# 4) 推送
git push -u origin main
```

### 5.3 大文件处理：从 Git 中剥离

`.pt` / `.onnx` / `.mp4` / `dataset/` 不进 Git，改走 GitHub Releases。

```powershell
cd "E:\desktop\deploy"

@"
*.pt
*.onnx
*.engine
*.mp4
weights/
dataset/
__pycache__/
.venv/
result.png
.DS_Store
"@ | Out-File -Encoding ascii .gitignore

# 从 Git 索引中移除（本地文件不会删）
git rm -r --cached .

git add .
git commit --amend --no-edit
git push -u origin main
```

推送结果（成功）：

```
Writing objects: 100% (9/9), 2.24 KiB | 2.24 MiB/s, done.
To https://github.com/Sxz-CV/Deploy.git
 * [new branch]      main -> main
branch 'main' set up to track 'origin/main'.
```

### 5.4 大文件改走 Releases

1. 打开 https://github.com/Sxz-CV/Deploy/releases/new
2. **Tag version** 填 `v1.0`，选择 `Create new tag on publish`
3. **Release title** 填 `v1.0 初始发布 - 模型权重与演示`
4. 把 `weights/best.pt`、`weights/best.onnx`、`weights/last.pt`、`过程.mp4`、`result.png` 拖进附件框
5. 数据集先打包成 zip 再拖（GitHub 不支持直接拖文件夹）
6. 点 **Publish release**

Release 单文件上限 **2 GB**，不占仓库体积，下载链接永久有效。

---

## 6. 遇到的问题与解决

### 问题 1：`user.email` 填成了仓库 URL

**现象**

```
git config --global user.email "`https://github.com/Sxz-CV/Deploy.git`"
```

**原因**：把命令里的格式标记（反引号）一起复制进终端，且误把 URL 当成邮箱。

**解决**

```powershell
git config --global user.email "3473884372@qq.com"
git config --global --list
```

邮箱必须是 GitHub 账号 → Settings → Emails 里绑定的地址，否则 commit 无法归属到你的账号。

---

### 问题 2：`fatal: unable to access '`https://...git/`'`

**现象**

```
fatal: unable to access '`https://github.com/Sxz-CV/Deploy.git/`': Recv failure: Connection was reset
```

**原因**：PowerShell 里反引号 `` ` `` 是转义字符，不是引号。复制命令时把 Markdown 的反引号一起粘进去，导致 remote URL 里存进了反引号。

**解决**：手打命令，URL 前后不加任何符号；或直接改 `.git/config`：

```powershell
notepad "E:\desktop\deploy\.git\config"
```

```ini
[remote "origin"]
    url = https://github.com/Sxz-CV/Deploy.git
    fetch = +refs/heads/*:refs/remotes/origin/*
```

验证：

```powershell
git remote -v
```

---

### 问题 3：`RPC failed; curl 55 Send failure: Connection was reset`

**现象**

```
Writing objects: 100% (1046/1046), 566.58 MiB | 1.50 MiB/s
error: RPC failed; curl 55 Send failure: Connection was reset
send-pack: unexpected disconnect while reading sideband packet
fatal: the remote end hung up unexpectedly
```

**原因**：一次推送 566 MB（含 `best.pt`、`best.onnx`、`过程.mp4`、整个 `dataset/`），代理链路扛不住这么大的单次上传。

**解决**：先用 `.gitignore` + `git rm -r --cached .` 把二进制产物从 Git 中剥离，只推源码（2.24 KiB），大文件走 Releases。

---

### 问题 4：`Failed to connect to github.com port 443`

**现象**

```
fatal: unable to access 'https://github.com/...': Failed to connect to github.com port 443 after 21069 ms: Could not connect to server
```

**原因**：国内直连 GitHub 被重置，且代理端口配错（按默认 7890 配，实际 Clash Verge 用的是 **7897**）。

**解决**：查 Clash Verge 首页「Clash 信息 → 混合代理端口」，按实际端口配：

```powershell
git config --global http.proxy http://127.0.0.1:7897
git config --global https.proxy http://127.0.0.1:7897
```

不用代理时清除：

```powershell
git config --global --unset http.proxy
git config --global --unset https.proxy
```

---

### 问题 5：`remote: 404 not found!` / `repository not found`

**现象**

```
remote: [session-9340a9aa] 404 not found!
fatal: repository 'https://gitee.com/Sxz-CV/Deploy.git/' not found
```

**原因**：remote 被误改成 Gitee 地址，但 Gitee 上并没有创建同名仓库；实际仓库建在 GitHub。

**解决**：把 `.git/config` 的 url 改回 GitHub，然后 `git push`。

---

### 问题 6：推送卡在 `Writing objects: 99%`

**原因**：包体太大（418~566 MB），代理链路在收尾阶段超时。

**解决**：`Ctrl + C` 中断，清理大文件后重推。判断依据：

```powershell
Get-ChildItem -Recurse -File | Sort-Object Length -Descending |
  Select-Object -First 10 FullName, @{n='MB';e={[math]::Round($_.Length/1MB,1)}}
```

---

### 问题 7：`trtexec` 看起来卡死

**现象**：终端长时间无输出，甚至出现 `^[OQ` 乱码。

**原因**：

- `trtexec ... | tail -20` 会缓冲输出，界面看起来完全静止
- TX2 构建 YOLO 分割 FP16 engine 本身要 5~15 分钟
- `^[OQ` 是按到 F2 键的终端编码，不是报错

**解决**：

```bash
ps aux | grep trtexec        # 状态 Rl+ 且 CPU 满载 = 正常运行
tegrastats                   # 看 CPU / GPU / 内存
```

不要中断，等 `&&&& PASSED` 出现。构建期间不会生成 engine 文件，属正常。

---

### 问题 8：`scp` 传输 `calib` 目录失败

**原因**：本地根本没有创建 `calib/` 目录（`No such file or directory`），且当前只做 FP16，不需要校准集。

**解决**：跳过 `calib`，只传 `best.onnx` / `best.pt` / `data.yaml`。

---

## 7. 常用命令速查

### 板端（`mima123456@houhoudesktop`）

```bash
hostname -I                                    # 查当前 IP
cd ~/smoke

# 构建 FP16 engine（5~15 分钟，勿中断）
/usr/src/tensorrt/bin/trtexec --onnx=best.onnx --fp16 \
  --saveEngine=best_fp16.engine --workspace=1024

# 加载已有 engine 测速
/usr/src/tensorrt/bin/trtexec --loadEngine=best_fp16.engine --iterations=200

# 加 swap
sudo fallocate -l 4G /swapfile && sudo chmod 600 /swapfile && \
sudo mkswap /swapfile && sudo swapon /swapfile && free -h
```

### PC（PowerShell）

```powershell
# SSH
ssh mima123456@10.223.2.66

# 传文件
scp E:\desktop\deploy\weights\best.onnx mima123456@10.223.2.66:~/smoke/

# 拉结果
scp -r mima123456@10.223.2.66:/home/mima123456/smoke/output E:\desktop\deploy\smoke_out

# Git
git remote -v
git status
git log --oneline
git push -u origin main
```

### 关键提醒

1. `scp` / `git` 在 PC 窗口跑，`trtexec` / `python3` 在板端跑，别搞混
2. 板子 IP 是 DHCP 动态分配，每次连之前先 `hostname -I`
3. `.engine` 绑定设备 + TensorRT 版本，不能跨设备复用，必须板端生成
4. `.pt` / `.onnx` / `.engine` / 视频 / 数据集不进 Git，走 Releases
5. 复制命令时不要带上反引号，PowerShell 会把它当转义符
