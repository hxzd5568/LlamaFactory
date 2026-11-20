# Plan: 在 LlamaFactory 中嵌入 compcheck/datastates checkpoint 引擎（coalition_save / split_load）

## 1. 目标

在 LlamaFactory 的 YAML 里加与 DeepSpeed bench 相同的配置：

```yaml
deepspeed: examples/deepspeed/ds_z3_config.json
checkpoint_engine: compcheck        # 新增 key（= deepspeed CLI 的 --checkpoint_engine compcheck）
async_load: false                   # classic 同步路径
```

效果：训练中每次 save_checkpoint 走 `CasyncEngine.save → ckpt_engine.coalition_save`；
load/resume 走 `CasyncEngine.load → ckpt_engine.split_load`。

## 2. 现状结论（已读代码确认）

- 本环境 site-packages 的 deepspeed 0.18.2 已是打补丁版本：
  `deepspeed/runtime/checkpoint_engine/casync_checkpoint_engine.py`、
  `datastates_checkpoint_engine.py`、`deepspeed/datastates/config.py` 均存在，
  且 `datastates` 包可 import（含编译好的 .so）。
- 引擎选择链（无需改动）：
  `DeepSpeedEngine._configure_checkpointing`（engine.py:1143）
  → `create_checkpoint_engine`（checkpoint_engine/utils.py:15）
  → ds config 有 `casync_ckpt` 键 → `datastates_config.enabled_casync=True`
  （datastates/config.py:26）→ `CasyncEngine`（casync_checkpoint_engine.py:48）。
  `datastates_ckpt` 键则选 `DataStatesCheckpointEngine`（注意 utils.py 先查
  `enabled`，两键同存时 datastates 优先）。
- CasyncEngine 行为（casync_checkpoint_engine.py:70-84）：
  - `save(state_dict, path, plain=False)`：plain=False → `coalition_save`；plain=True → `save_`
  - `load(path, map_location)`：路径含 `optim_states_plain` → plain `load`；否则 → `split_load`
  - `is_decoupled() = True`：save 只是入队，**必须显式 commit/wait 才真正落盘**
- transformers 侧的接入点（4.57.3，无需改）：
  - 保存：`Trainer._save_optimizer_and_scheduler`（trainer.py:3452）→
    `self.model_wrapped.save_checkpoint(output_dir)`（所有 rank 都调用）→
    engine `_save_checkpoint`/`_save_zero_checkpoint` → `checkpoint_engine.save(...)`
    → coalition_save。
  - 加载：`deepspeed_load_checkpoint`（integrations/deepspeed.py:485）→
    `engine.load_checkpoint(load_optimizer_states=True, ...)` → `checkpoint_engine.load(...)`
    → split_load。**classic（async=0）路径无需任何 LlamaFactory 改动。**
- `HfTrainerDeepSpeedConfig` 接受路径或 dict（accelerate/utils/deepspeed.py:136），
  未知键 `casync_ckpt`/`async_load`/`hot_weights` 会原样透传给 `deepspeed.initialize`，
  不会报错（需在验证步骤再确认一次）。

## 3. 改动清单（LlamaFactory 侧）

### A. `src/llamafactory/hparams/training_args.py` — 新增参数字段

给 `TrainingArguments` 增加一个 mixin（或直接加字段），让 HfArgumentParser 接受
YAML 里的新 key（否则 parse_dict 会因未知 key 报错）：

```python
checkpoint_engine: str | None = None      # "compcheck" | "datastates" | None
async_load: bool = False                  # 顶层 async_load
hot_weights: bool = False                 # 顶层 hot_weights
host_cache_size: int = 10                 # casync_ckpt 子键（默认值对齐 bench）
error_bound: float = 0.001
overlap_reserve_mb: int = 10000
stage_requirement_mb: int = 2850
```

### B. `src/llamafactory/hparams/parser.py` — 注入 ds config

在 `get_train_args` 的 post-process 段（约 611 行 "Post-process training arguments" 处）新增：

```python
if training_args.deepspeed and training_args.checkpoint_engine in ("compcheck", "datastates"):
    _inject_checkpoint_engine_config(training_args)
```

实现 `_inject_checkpoint_engine_config`：
1. `json.load` 读 `training_args.deepspeed` 指向的 ds JSON（保留原有 zero/bf16 等键）；
2. `compcheck → casync_ckpt`，`datastates → datastates_ckpt`，写入
   `{host_cache_size, error_bound, overlap_reserve_mb, stage_requirement_mb}`；
3. `async_load=True` → 顶层 `"async_load": true`；`hot_weights=True` → 顶层 `"hot_weights": true`；
4. dump 到 `os.path.join(training_args.output_dir, f"ds_config_{engine}.json")`
   （`os.makedirs(exist_ok=True)`，多 rank 写同名同内容文件无害）；
5. `training_args.deepspeed = merged_json_path`（发生在 Trainer 初始化之前，
   `HfTrainerDeepSpeedConfig` 届时读到完整配置）。

### C. `src/llamafactory/train/trainer_utils.py` — commit 助手

```python
def commit_checkpoint_engine(trainer) -> None:
    model_wrapped = getattr(trainer, "model_wrapped", None)
    ckpt_engine = getattr(model_wrapped, "checkpoint_engine", None)
    if ckpt_engine is not None and ckpt_engine.is_decoupled():
        # decoupled 引擎必须显式排空队列 + fsync，否则 checkpoint 不完整
        commit = getattr(model_wrapped, "_commit_decoupled_checkpoint", None)
        if commit is not None:
            commit()                      # 写 latest + barrier
        else:
            ckpt_engine.wait(True)        # CasyncEngine.wait(persist=True)
```

### D. trainer 子类覆盖 `_save_checkpoint` — save 后 commit

所有 rank 都会进 `_save_checkpoint`（`DefaultFlowCallback` 不按 rank 过滤），
deepspeed 侧 commit 内含 barrier，安全。

- `src/llamafactory/train/sft/trainer.py`：`CustomSeq2SeqTrainer._save_checkpoint`
  → `super()._save_checkpoint(model, trial)` 之后 `commit_checkpoint_engine(self)`。
- `src/llamafactory/train/pt/trainer.py`：`CustomTrainer` 同样处理。
- （可选，二期）dpo/rm/kto 等 trainer 同模式接入。

### E. 示例 YAML：`examples/dpu/qwen3_lora_sft_compcheck.yaml`

复制 `qwen3_lora_sft_ds3_ckpt.yaml`，仅改：

```yaml
deepspeed: examples/deepspeed/ds_z3_config.json
checkpoint_engine: compcheck
async_load: false
save_steps: 200
save_total_limit: 3
save_only_model: false
```

### F. 环境前置（非代码，部署时校验）

1. 运行 env 的 deepspeed 必须是带 `casync_checkpoint_engine.py` / `datastates/config.py`
   与补丁 `engine.py` 的版本（本机 xia-dev 已具备）；
2. `datastates` 包可 import（含 `datastates_core_py` 的 .so）；
3. 冒烟：`python -c "from datastates import CheckpointEngine; from deepspeed.runtime.checkpoint_engine.casync_checkpoint_engine import CasyncEngine"`。

## 4. 不改的部分（结论）

- load 路径：classic `load_checkpoint` → `split_load` 由 deepspeed 引擎自身完成，
  transformers 调用链不变。
- `trainer.save_model()`（最终 HF 权重）：走 `accelerator.get_state_dict(self.deepspeed)`
  直写 sharded safetensors，与 checkpoint engine 无关，不需要 commit。
- `stage3_gather_16bit_weights_on_model_save: true` 保持（HF 可加载副本照常产出）。

## 5. 风险与注意

1. **decoupled 持久化**：C 未落地前，engine 文件可能尚未落盘就训练结束 → 必改项 D。
2. **`latest` 文件**：只有 commit 才写（engine.py:4051）；transformers resume 靠
   `get_last_checkpoint` 扫 `checkpoint-*` 目录，不依赖 `latest`，无影响。
3. **有损压缩**：`error_bound: 0.001`，weight 恢复有量化误差（sanity diff 需 ≈0，
   bench 已验证）；如需无损走 `datastates` 引擎或 plain 布局（async=1 的 optim）。
4. **async_load=1** 仅对两阶段 `load_checkpoint_stage` 有意义，transformers 不走该
   路径；本方案默认 async_load=false（classic，save→coalition_save / load→split_load
   双向压缩），符合需求。
5. **z3 + CasyncEngine**：bench 主要在 z1/z2 验证；z3 先冒烟（save→load→diff）。
6. `parse_dict` 现在能接受新 key 的前提是 A 已落地，否则先报 unknown args。

## 6. 验证计划

1. `llamafactory-cli train examples/dpu/qwen3_lora_sft_compcheck.yaml`（先单卡
   `FORCE_TORCHRUN=1` 或直接多卡）；
2. 日志确认：`CasyncEngine` 初始化成功（datastates import 无回退 torch）；
3. `checkpoint-200/` 产物检查：engine 文件为 compcheck 布局（压缩后的
   `*_model_states.pt` / `*_optim_states.pt`，可对比文件大小显著小于 torch 版），
   且 HF sharded safetensors 同目录存在；
4. 中断后 `resume_from_checkpoint: auto` 续训 → 日志确认走 split_load（可临时在
   CasyncEngine.load 加 print 或看 datastates 引擎日志）；
5. `async_load: true` 冒烟（可选）：权重 coalition_save、optim 文件带
   `optim_states_plain` 后缀，load 走 plain `load`。
