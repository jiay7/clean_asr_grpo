# 从 ms-swift 抽出的 ASR GRPO 内核

这是可独立运行和测试的 **PyTorch 单进程训练内核**，不是完整 ASR 训练仓库，也不是把 Swift Trainer 原样复制出来。实际声学模型尚未接入；测试用小型离散策略验证训练闭环，不代表 ASR 效果。

## 来源和抽取边界

固定上游版本：[ms-swift f7385b2](https://github.com/modelscope/ms-swift/tree/f7385b200c4f629dec1a70af32c9b4174067b9c1)。避免后续 main 分支更新导致实现无法对照。

| 上游位置 | 本地位置 | 保留内容 |
| --- | --- | --- |
| `swift/rl_core/advantage.py::compute_advantages` | `core.py::advantages` | 奖励加权求和、组内均值基线、group/batch/none 标准化 |
| `swift/rlhf_trainers/grpo_trainer.py::_compute_loss_and_metrics` | `core.py::grpo_loss` | token 级新旧策略比值、双边 clipping、KL-in-loss、三种归一化 |
| Swift 的训练/推理调度 | `trainer.py`，重新编写 | 同步采样、固定 old/ref logps、反向传播和梯度裁剪 |
| 无对应默认 ASR 配方 | `rewards.py`，新增 | CER/WER 示例、空标签处理 |

保留 Apache-2.0 许可证及来源声明，见 LICENSE、NOTICE。不是完整 ms-swift 功能等价替代。

## 文件

- `core.py`：纯张量算法；不依赖 Transformers、TRL、Swift。
- `trainer.py`：模型适配协议与单个 rollout 的训练闭环。
- `rewards.py`：转写奖励示例。
- `test_grpo.py`：数学边界、mask、奖励、参数更新测试。

训练过程：同一条音频生成 G 个候选 → 奖励 → 组内优势 → 固定旧策略概率和参考概率 → 对当前策略计算损失 → 更新。允许一个 rollout 做多次更新，old logps 不随更新重算。参考模型应是 RL 开始时的 SFT 快照，一直固定。

## 接入已有 SFT 模型

只需实现 `Adapter` 的两个方法，但这部分必须针对实际模型验证：

1. `sample(policy, batch, group_size)`：读取音频、按实际模型方式构造输入、随机采样，返回候选文本、包含 EOS 的 completion mask，以及后续重算概率所需的数据。按 `音频0候选0…候选G-1、音频1候选0…` 排列。所有候选必须共享同一次加噪后的音频，不能在各次前向重新随机混音。
2. `logps(model, rollout)`：对固定候选重新前向，返回 **生成部分** 每个 token 的 log probability `[B*G,T]`。需正确进行 causal shift，排除音频位置、prompt 和 padding，但保留 EOS。不能简单套用原始 prompt 长度切片而忽略模型内部音频展开。

最小用法（`policy/reference/adapter/batches` 由实际工程提供）：

```python
from functools import partial
import torch
from clean_asr_grpo import Config, Trainer
from clean_asr_grpo.rewards import asr_rewards

config = Config(group_size=4, beta=0.04, updates_per_rollout=1)
# 先设置 requires_grad，再构造 optimizer。
optimizer = torch.optim.AdamW(
    [p for p in policy.parameters() if p.requires_grad], lr=1e-6)
trainer = Trainer(
    policy, reference, adapter, optimizer,
    partial(asr_rewards, group_size=config.group_size, unit="char"), config)
for batch in batches:
    metrics = trainer.step(batch)
```

### 更新哪些参数

内核不擅自选择 Encoder、Projector 或 LoRA。由调用方显式设置 `requires_grad`，optimizer 必须恰好包含这些参数。建议首轮冻结 Encoder、训练 Projector 和 LLM LoRA；这是实验建议，不是上游默认策略。

LoRA 的创建和权重保存由现有模型工程负责。若已有 PEFT adapter，注意参考模型必须包含 SFT adapter 的效果，不能把“关闭 adapter 后的基座”误当成 SFT reference。此版采用独立 reference，不实现共享基座的省显存方式。

冻结 Encoder 时可以复用确定性的特征。训练 Encoder 时每次 logps 必须重新计算带梯度特征；缓存 detached embedding 会让 Encoder 实际不更新。Trainer 用 eval 模式关闭 dropout，但反向传播仍开启；其他依赖 train/eval 的自研模块需要检查。

### 采样分布

为了保证这个精简版本的 on-policy 假设，适配器默认应使用温度 1、不做 top-k/top-p 截断、不加重复惩罚，并让采样和 logps 使用同一分布。若改成温度 0.7 或接 vLLM，必须处理采样/训练概率差异；本版没有移植 Swift 的 rollout importance-sampling correction。

### 奖励与纯噪声

示例为 `1-CER` 或 `1-WER`，没有把负值强行截为零。空参考文本单独处理：空输出 +1，非空输出 -1。空输出仍应采到 EOS，不能生成全零 mask。文本做 NFKC、大小写和标点归一化；实际业务需统一评测规范，尤其数字、专有词和中文分词。

纯噪声一组全部输出非空时，示例奖励全部为 -1，组内优势为零；这不是能解决全部幻觉的完整奖励设计。`zero_std_fraction` 用于观察这种情况。重复/长度奖励须另做消融，避免奖励“少说话”却恶化漏字。

## 算法选择和限制

- 默认 `grpo`：每条候选按自己的有效长度归一化，再平均。
- `bnpo`：按当前单进程 batch 的有效 token 总数归一化。
- `dr_grpo`：除以候选数 × 固定最大生成长度。
- KL 沿用该上游版本的数值保护：log-ratio 限制在 [-20,20]，k3 限制在 [-10,10]。这不是无限范围的原始 KL 公式。
- 奖励标准差沿用上游的样本标准差和 `1e-4` 稳定项。
- EOS mask helper 会返回未出现 EOS 的标记；当前默认仍训练截断候选，没有抽取 overlong_filter。
- 不包含 DAPO、GSPO、动态采样、异步 rollout、多轮对话、奖励模型、分布式训练、梯度累积、混合精度、checkpoint、推理引擎权重同步。
- 所有奖励缺失时显式报错；这比上游警告后继续的行为更严格。

## 测试

在本目录的父目录运行：

```bash
python -m pip install -r clean_asr_grpo/requirements.txt
python -m unittest clean_asr_grpo.test_grpo -v
# 可选：联网读取固定版本的上游函数，检查优势计算数值一致性。
python -m clean_asr_grpo.verify_upstream
```

本次验证：Python 3.12 / PyTorch 2.14.0 / CPU，9 项单元测试通过；优势计算的 9 组配置与固定上游函数逐值一致（容差 0）。损失验证包含手算 clipping、三种归一化和梯度测试，未运行整个 Swift Trainer 做端到端对照。测试环境位于 `/tmp/asr-grpo-verify-env`，未修改现有 ASR 工程环境。

上线前还需用真实音频验证：同一候选的生成/重算概率、音频展开后的 token 对齐、EOS、padding、SFT reference 一致性、各模块梯度，以及小批次过拟合。未完成这些检查前，不应直接开展大规模 GRPO 训练。
