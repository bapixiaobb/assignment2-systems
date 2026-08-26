# NPU_ENV_SETUP — CS336 Assignment 2 Ascend 环境完整配置指南

> 用途：在任何一台 Ascend 950DT + CANN 9.1 主机上从零重建 `cs336` 环境。
> 本文档自包含：审计 → 建环境 → 安装 → 验证 → 已知坑。§7 是本机（a5-docker）的实际执行记录与结果。
> 生成：2026-08-31（首次建成于 a5-docker 容器，全部验证通过）

---

## 1. 目标与边界

- 隔离 conda 环境 `cs336`（Python 3.12，满足项目 `>=3.12,<3.14`）
- torch 2.11.0 + torch_npu 2.11.0（stable 严格配对），NPU 前向可跑
- Triton-Ascend 可用（`--no-deps` 方式，避免破坏作业依赖）
- 不动：系统/共享 Python、驱动/固件/CANN、仓库 `pyproject.toml` / `uv.lock` / 作业代码与测试、现有 conda 环境

## 2. 新机器前置审计（先跑，再动手）

```bash
# 架构 / OS
uname -m && cat /etc/os-release | head -2

# NPU 型号、卡数、健康状态（期望：Ascend950DT × 8, Health OK）
npu-smi info

# CANN 安装位置与版本（记下 set_env.sh 路径，本机为 /usr/local/Ascend/cann-9.1.0）
echo "ASCEND_TOOLKIT_HOME=${ASCEND_TOOLKIT_HOME:-NOT SET}"
ls /usr/local/Ascend/

# conda 与 python 版本
which conda && conda --version && conda env list

# pip 镜像（期望有华为云/同类镜像；无则见 §5 坑 4）
cat /etc/pip.conf 2>/dev/null

# 磁盘（torch 栈约需 5 GB）
df -h /home /root
```

判断标准：CANN 9.1.x 存在、npu-smi 能看到 950 卡、conda 可用、有可用 PyPI 镜像 → 按下述步骤执行；任一不满足先解决再继续。

## 3. 建环境与安装（按序执行）

### 3.1 conda 环境

```bash
source <conda安装路径>/etc/profile.d/conda.sh   # 本机: /home/ma-user/anaconda3
conda create -n cs336 python=3.12 -y
PY=/home/ma-user/anaconda3/envs/cs336/bin       # 本机路径，新机器按 env list 输出调整
```

若 conda 报自签证书错误（走公司代理常见），先合并系统 CA 与代理 CA 再重试：

```bash
cat /etc/pki/tls/certs/ca-bundle.crt /etc/pki/ca-trust/source/anchors/<代理CA>.crt > /tmp/combined-ca.crt
export REQUESTS_CA_BUNDLE=/tmp/combined-ca.crt
```

### 3.2 torch + torch_npu（核心栈，约 2.3 GB）

```bash
$PY/pip install torch==2.11.0 torch_npu==2.11.0
```

- 版本依据：项目要求 `torch~=2.11.0`；torch_npu **必须同版本号配对**；本机 CANN 9.1.0 与 torch_npu 2.11.0 实测兼容。
- PyPI 的 torch 会顺带装入 triton 3.6.0 与 nvidia-*-cu13 元数据包（无 CUDA 卡也拉，属正常，不删）。

### 3.3 项目依赖（按 uv.lock 锁版，不跑 uv sync）

```bash
cd <assignment2-systems 仓库路径>
$PY/pip install pyyaml numpy==2.4.4 einops==0.8.2 einx==0.4.3 jaxtyping==0.3.9 \
  psutil==7.2.2 pytest==9.0.3 pytest-timeout==2.4.0 regex==2026.4.4 tiktoken==0.12.0 \
  tqdm==4.67.3 wandb==0.26.0 pandas==3.0.2 ty==0.0.31 ruff==0.15.10 \
  humanfriendly==10.0 matplotlib==3.10.8
```

- 版本来源：仓库 `uv.lock` 的锁定值；若 lock 更新过，用
  `grep -A1 '^name = "<包名>"' uv.lock | grep version` 重新抽取。
- `pyyaml`：torch_npu 运行时 import 但**未声明**的依赖，必须手装，否则 `import torch` 直接报 backend 加载失败。

### 3.4 cs336-basics（editable，跳过依赖解析）

```bash
$PY/pip install -e ./cs336-basics --no-deps
```

- `--no-deps` 必加：其 pyproject 走 uv 专属 source 配置，裸 pip 解析会试图从镜像拉不存在的 `cs336-basics` 包。
- 仓库根项目 `cs336-systems` 不用 pip 安装；在仓库根目录跑 `python -m pytest` 即可通过 cwd 解析 `cs336_systems`。

### 3.5 Triton-Ascend（可选，做 Triton kernel 时才需要）

```bash
# 元数据里钉死 numpy==1.26.4/pytest==8.3.2/psutil==6.0.0/triton==3.5.0，
# 与作业依赖冲突 → 必须 --no-deps
$PY/pip install --no-deps triton-ascend==3.2.2 \
  --extra-index-url https://triton-ascend.osinfra.cn/pypi/simple
# 补齐其真实需要的运行时包（env 里本来没有，不与任何锁版冲突）
$PY/pip install pybind11 attrs decorator
```

⚠️ 该 wheel 是整体 fork：安装会把 site-packages 的 `triton` 包文件覆盖为 3.2.0-fork（pip 登记仍显示 3.6.0）。实测不影响 NPU 前向与 Triton kernel。还原命令：

```bash
$PY/pip uninstall -y triton-ascend && $PY/pip install --force-reinstall --no-deps triton==3.6.0
```

## 4. 验证套件（三个脚本，全部应 PASS）

所有命令前先 source CANN（**必须**，否则 `libhccl.so: cannot open shared object file`）：

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh    # 路径按 §2 审计结果调整
```

### 4.1 NPU 基础（`npu_base_check.py`）

```python
import torch
import torch_npu

print("torch:", torch.__version__, "| torch_npu:", torch_npu.__version__)
print("npu available:", torch.npu.is_available(), "| count:", torch.npu.device_count())
assert torch.npu.is_available()
print("device 0:", torch.npu.get_device_name(0))
x = torch.randn(1024, 1024, device="npu:0")
y = x @ x.T
torch.npu.synchronize()
print("matmul OK:", tuple(y.shape), "| reduce:", (x * 2 + 1).sum().item())
print("ALL BASE CHECKS PASSED")
```

### 4.2 模型前向（`npu_forward_smoke.py`，小配置）

```python
import torch
import torch_npu
from cs336_basics.model import BasicsTransformerLM

torch.manual_seed(0)
m = BasicsTransformerLM(vocab_size=1000, context_length=64,
                        d_model=64, num_layers=2, num_heads=2, d_ff=128).to("npu:0")
ids = torch.randint(0, 1000, (2, 64), device="npu:0")
m.eval()
with torch.no_grad():
    logits = m(ids)
torch.npu.synchronize()
assert logits.shape == (2, 64, 1000) and logits.device.type == "npu"
print("forward OK:", tuple(logits.shape), logits.device, "| SMOKE TEST PASSED")
```

### 4.3 Triton vector-add（`triton_vector_add.py`，装了 §3.5 才跑）

```python
import torch
import torch_npu
import triton
import triton.language as tl
import importlib.metadata


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask) + tl.load(y_ptr + offs, mask=mask), mask=mask)


x = torch.randn(4096, device="npu:0")
y = torch.randn(4096, device="npu:0")
out = torch.empty_like(x)
add_kernel[(triton.cdiv(x.numel(), 1024),)](x, y, out, x.numel(), BLOCK=1024)
torch.npu.synchronize()
ok = torch.allclose(out, x + y)
print("triton(on-disk):", triton.__version__, "| triton-ascend:", importlib.metadata.version("triton-ascend"))
assert ok
print("TRITON-ASCEND MINIMAL EXAMPLE PASSED")
```

注意：取 triton-ascend 版本用 `importlib.metadata`，它没有顶层 `triton_ascend` 模块（后端自动注册到 `triton.backends.ascend`）。

### 4.4 仓库回归

```bash
cd <assignment2-systems 仓库路径> && $PY/python benchmark.py
# 期望输出: Logits shape: torch.Size([4, 512, 10000])
```

## 5. 已知坑速查表

| # | 症状 | 原因 | 解法 |
|---|---|---|---|
| 1 | `import torch` → `Failed to load the backend extension: torch_npu`，或 `libhccl.so: cannot open shared object file` | CANN 未 source | 先 `source <cann>/set_env.sh`（或交互 shell 用 `ascend_enable`） |
| 2 | `import torch` → `No module named 'yaml'` | torch_npu 未声明 PyYAML | `pip install pyyaml`（§3.3 已含） |
| 3 | conda 报 `self-signed certificate in certificate chain` | 公司代理 MITM，conda 的 requests 不认代理 CA | §3.1 的 REQUESTS_CA_BUNDLE 合并法 |
| 4 | pip 无镜像/超时 | 新机器没配 pip.conf | `pip install ... -i https://mirrors.huaweicloud.com/repository/pypi/simple`（triton-ascend 另加 osinfra extra-index，见 §3.5） |
| 5 | pip 大量 `asc-opc-tool/ms-service-profiler requires ...` 冲突警告 | 机器把 CANN 的 python/site-packages 注入了 sys.path（环境变量） | 外观性问题，不影响 env 内依赖；确认法：env 里 `python -c "import numpy; print(numpy.__version__)"` 应为 2.4.4 |
| 6 | 装 triton-ascend 想把 numpy/pytest/psutil 降级 | 官方钉版与作业锁版冲突 | 必须 `--no-deps` + 手补 pybind11/attrs/decorator（§3.5） |
| 7 | `import triton` 版本变 3.2.0 | triton-ascend wheel 覆盖 triton 包文件 | 预期行为，见 §3.5 还原命令 |
| 8 | 多卡训练抢卡 | 8 卡共享 | `export ASCEND_RT_VISIBLE_DEVICES=<卡号>` 选卡 |

## 6. 日常使用

```bash
conda activate cs336
source /usr/local/Ascend/cann-9.1.0/set_env.sh
cd <assignment2-systems 仓库路径>
python -m pytest tests/ -x      # 测试
python benchmark.py             # 前向
```

## 7. 本机（a5-docker）实际执行记录

| 项目 | 值 |
|---|---|
| 主机 | Huawei Cloud EulerOS 3.0, aarch64, 192 核 / 2254 GB, 容器 root |
| NPU | 8 × Ascend950DT 全 OK，单卡 HBM 98304 MB，npu-smi 25.1.rc2 |
| CANN | 9.1.0 @ `/usr/local/Ascend/cann-9.1.0` |
| 环境 | `cs336` @ `/home/ma-user/anaconda3/envs/cs336`，Python 3.12.14 |
| 安装方式 | 华为云镜像在线安装（旧机器离线包目录已随旧机回收，不存在） |

验证结果（2026-08-31，全部通过）：

| 检查 | 结果 |
|---|---|
| torch / torch_npu | 2.11.0 / 2.11.0，npu available=True，8 卡，device0=Ascend950DT_9582 |
| npu:0 矩阵乘 + 规约 + sync | ✅ 无异步错误 |
| BasicsTransformerLM 小配置前向（模型+ids 均在 npu:0） | ✅ logits=(2,64,1000) |
| Triton vector-add on npu:0 | ✅ allclose 通过（triton on-disk 3.2.0 / triton-ascend 3.2.2） |
| 装 triton-ascend 后前向回归 | ✅ 不受影响 |
| benchmark.py（CPU） | ✅ [4, 512, 10000] |

包清单（关键项，全部装在 cs336 env 内）：torch 2.11.0、torch_npu 2.11.0、numpy 2.4.4、pandas 3.0.2、pytest 9.0.3、einops 0.8.2、einx 0.4.3、jaxtyping 0.3.9、tiktoken 0.12.0、wandb 0.26.0、ty 0.0.31、ruff 0.15.10、regex 2026.4.4、tqdm 4.67.3、psutil 7.2.2、matplotlib 3.10.8、humanfriendly 10.0、PyYAML 6.0.3、triton-ascend 3.2.2（+pybind11/attrs/decorator）、cs336-basics 26.0.4（editable）。
