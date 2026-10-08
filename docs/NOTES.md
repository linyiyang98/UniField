# 论文核对与复现说明

对应作者提供的 `2603.09223v2.pdf`。本目录是从原 FlashVSR MRI 实现整理得到的独立代码。

## 已补齐和修复

| 项目 | 当前处理 |
|---|---|
| FASRM / FASFL | 原代码只有 L1；现按式 (2) 加入速度场的 3D FFT 分带损失 |
| 文本条件 | 显式 CSV 模态/场强生成 prompt，使用真实冻结 UMT5-XXL 编码 |
| LoRA 梯度 | cross-attention K/V 每次重算，避免 `no_grad` 缓存切断梯度 |
| 优化计划 | Adam β=(0.5,0.999)、lr=1e-4、500 步后线性衰减，共 1,000 optimizer steps |
| MRI 预处理 | 0.5–99.5 百分位归一化、1 mm Z 重采样、256×256×160 resize；保持浮点精度 |
| VAE / 解码 | 修复末尾三张切片遗漏；双向解码按层分块并保留跨块邻域 |
| 双卡与恢复 | 默认 GPU 4、5；DDP 参数同步；保存 optimizer/scheduler/每卡 RNG |
| 推理 | 从纯噪声解 Euler ODE；不使用 HF 目标加噪或编码 |
| NIfTI / 指标 | 保存变换后 affine，可回到原网格；PSNR、SSIM、NRMSE、LPIPS |
| 数据 / 消融 | 按中心与受试者划分，检查泄漏；单模态/单任务/L1-only 配置工具 |

## 明确的实现约定

- 流匹配：`v=noise-clean`，`zt=(1-t)*clean+t*noise`，均匀采样 `t∈[0,1]`。
- FASFL：对 latent 深度/高/宽做 FP32 orthonormal FFT，误差幅值取 `alpha+2` 次方；每带平均后按任务权重加权。λ=0.1、α=1；64mT→3T 权重 `[0.2,0.5,0.3]`，3T→7T 为 `[0.1,0.3,0.6]`。
- 分带径向频率：`sqrt((fz/.5)^2+(fy/.5)^2+(fx/.5)^2)/sqrt(3)`，边界 1/3、2/3。论文未给出精确阈值和 FFT 归一化，这属于可调整的公开约定。
- rank/alpha=128、训练 crop=40、topk=2、local range=11 来自原代码，论文未明确这些设置。默认 Euler ODE 一步，也可指定多步。
- LCSA 窗口 `(2,8,8)`、128-token 块，保证对角有效；默认 SDPA 保留块掩码，逐查询块执行。可选 compiled block-sparse 后端尚未实测。
- LF/HF 独立归一化到 `[-1,1]`；灰度复制到三通道。部署推理不使用 HF 统计量。输出 RGB 取平均并保存为 `[0,1]` 灰度。
- PSNR 基于全体积 MSE、范围 1（零误差时上限 120 dB）；NRMSE=`100*sqrt(MSE)`；SSIM 为轴向切片均值、窗口 7、以百分数报告；LPIPS 为灰度复制 RGB 后的 AlexNet v0.1 切片均值。不使用前景掩码，每体积等权。
- 数据身份为 `(center,subject)`；跨中心同一人的别名需要数据整理者统一。恢复训练要求相同数据划分和 GPU 数量，保存每卡随机状态；数值仍可能随 GPU 内核变化。
- 注册辅助工具是 rigid+affine mutual information。论文没有提供原始注册参数，故不声称该辅助工具复现最终注册流程。

## 实际验证

2026-10-08，RTX A6000 物理 GPU **4、5**，PyTorch 2.10.0+cu128 / MONAI 1.5.1 / PEFT 0.18.1：

- 256×256×160 预处理、40 切片、rank 128：训练与两次恢复成功，停在第 3 步。所有 LoRA 梯度有限，cross-attention K/V 梯度非零，两卡参数哈希一致。
- 两个场强转换各一个体积，完整 256×256×160 推理、四项指标及原网格输出通过。初始训练每卡峰值 7.73 GiB，恢复 9.04 GiB，推理含 LPIPS 12.24 GiB。
- LF-only 两步 ODE 推理通过；9 项测试及仓库外安装包测试通过。
- 数据 227 对训练 / 58 对测试，共 90 / 23 位受试者。285 对均通过 shape/affine 检查，无训练测试受试者重叠；各中心人数与表 1 一致。

| 场强转换 / 模态 | PSNR (dB) | SSIM (%) | NRMSE (%) | LPIPS |
|---|---:|---:|---:|---:|
| 64mT→3T / T1 | 15.7379 | 55.5810 | 16.3345 | 0.2874 |
| 3T→7T / T1 | 13.3245 | 42.6005 | 21.5664 | 0.3708 |

以上只是 **3 步训练、2 个体积的流程验证**，尚未完成 1,000 步训练、全部 58 对测试或表 4 消融复现。最终性能仍需完整实验。论文未明确的分带、LoRA/ODE 设置及评价细节也需作者核对。

## 保留的本机产物与权重政策

- `runs/checkpoints/latest.pt`：最新私有 LoRA、optimizer 和恢复状态。
- `runs/manifests/{train,test}.csv`：私有清单；`cache/text/`：真实 UMT5 prompt 缓存。
- `configs/local.yaml`：本机配置；使用 `scripts/train.sh --resume runs/checkpoints/latest.pt` 续训。
- 第三方预训练权重和 tokenizer 公开提供，保留官方 SHA-256、版本与许可；**自训 LoRA、checkpoint 和医学数据不公开**。
