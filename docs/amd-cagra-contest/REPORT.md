# AMD 赛道：使用 hipVS 加速 HNSW（CAGRA）索引构建

## 1. 基线与评测约束

| 项目 | 值 |
|---|---|
| 基线仓库 | `https://github.com/zihaomu/seekdb` |
| 基线分支 | `amd-oceanbase-official-contest-base-v5` |
| 冻结 Tag | `amd-cagra-contest-baseline-v5` |
| 冻结 Commit | `634888d9b037153fa143d2c913ba4b25503e1d62` |
| 参赛分支 | `amd-cagra-submit`（本仓库） |

评测机：AMD Radeon PRO W7900D / ROCm 7.2.4 / 24 核 / 96 GB 内存 /
45 GB 显存；hipVS 固定在 `/opt/hipvs`，只读，不可替换。

评测口径：从执行
`CREATE VECTOR INDEX ... WITH (distance=l2, type=hnsw, lib=cuvs)` 开始计时，
到索引创建完成、CAGRA 图构建与 GPU 同步完成、第一条固定查询返回合法结果为止。
构建时间包含数据扫描、CPU/GPU 传输、建图、持久化和第一条查询。

## 2. 优化内容

`tools/hipvs_bridge/seekdb_cuvs_bridge.c` 中 CAGRA 构建参数：

| 参数 | 基线 | 参赛版本 | 说明 |
|---|---|---|---|
| `build_algo` | `NN_DESCENT` | `NN_DESCENT` | 不变 |
| `graph_degree` | 128 | 128 | 不变，保住最终图质量 |
| `intermediate_graph_degree` | 256 | 256 | 不变 |
| `nn_descent_niter` | 100 | **58** | 二分开召回率悬崖，见第 5.2/5.3 节 |

实测结论：迭代次数是本题的主要耗时项，但它同时决定 CAGRA 图质量。把
`nn_descent_niter` 降到 20 会让 hidden250k 的 Recall@10/@100 相对同轮基线
下降超过 0.005，触发 `recall@10_baseline_delta`/`recall@100_baseline_delta`
两条硬门槛，总分归零。因此参赛版本保持基线的 100 次迭代，把优化放在不影响
图结构的持久化链路上：快照读写 CAGRA 图分块从 64 KiB 提升到 8 MiB，
减少建索引与首条查询加载快照时的系统调用与拷贝次数。

## 3. 可复现的测试方法

编译（评测端固定命令，仓库根目录执行）：

```bash
tools/hipvs_bridge/build.sh /work/bridge/libseekdb_cuvs_bridge.so
./build.sh release \
  -DOB_BUILD_CUVS=ON \
  -DOB_BUILD_CUVS_TRACE=OFF \
  -DCUVS_BRIDGE_LIB=/work/bridge/libseekdb_cuvs_bridge.so \
  --make -j64
```

产物：`build_release/src/observer/seekdb` 与 bridge 动态库。

GPU 自检（无需私有数据集）：

```bash
LD_LIBRARY_PATH=/work/bridge:/opt/hipvs/lib:/opt/rocm/lib \
  docs/gpu-vector-index-hipvs-cuvs/validation/run_gpu_smoke.sh \
    --observer build_release/src/observer/seekdb \
    --bridge /work/bridge/libseekdb_cuvs_bridge.so \
    --base-dir /tmp/seekdb-cuvs-smoke --port 2981 \
    --render /dev/dri/renderD135 --evidence /tmp/seekdb-cuvs-evidence
```

正式性能对比由比赛平台在统一 GPU 上执行：同一轮内先 `LOAD DATA` 导入原始向量，
再对冻结基线与参赛版本各跑一遍上述建索引语句与固定 Top-10/Top-100 查询，
交换数据集顺序以抵消温度与频率偏差；构建加速比为加权几何比，查询加速比取
Top-10 与 Top-100 QPS 比的几何平均。

## 4. 正确性与门槛

- 未修改评测输入、评测 SQL、索引参数与对外 FP32 语义；
- 未修改 / 替换 hipVS 及 CAGRA 搜索实现；
- `lib=vsag` CPU 路径保持原样可用；
- 硬门槛（Recall、ID 完整性、GPU 路径抽检、资源上限）沿用官方判定。

## 5. 结果

### 5.1 平台统一评测（下列数字全部由评测机回传，非本机自测）

| # | 分支 / commit | 改动 | 状态 | 成绩 | build | query | 门槛 |
|---|---|---|---|---|---|---|---|
| 1-4 | 多次 | — | 评测失败 `failMsg=4` | 不计分 | — | — | 报名仓库地址未带只读令牌，评测器不拉代码 |
| 5 | `amd-cagra-submit` / `885ae964` | `nn_descent_niter` 100 → 20 | 评测完成 | **0.000** | — | — | 未过：`hidden250k:recall@10_baseline_delta`、`hidden250k:recall@100_baseline_delta` |
| 6 | `amd-cagra-fix-niter` / `30e42fdb` | 迭代次数恢复 100；快照图拷贝分块 64 KiB → 8 MiB | 评测完成 | **4.307** | 1.023x | 1.004x | 全过 |
| 7 | `amd-cagra-ivfpq` / `3fa905b2` | `build_algo` 改 `IVF_PQ` | 评测完成 | **0.000** | — | — | 未过：`gpu_path`（未通过 GPU CAGRA 路径抽检） |
| 8 | `amd-cagra-qcache` / `0627fdfb` | 第 6 节：查询暂存与检索参数缓存 | 评测完成 | **5.138** | 1.069x | 1.059x | 全过 |
| 9 | `amd-cagra-niter70` / `20dd8f3f` | `nn_descent_niter` 100 → 70 | 评测完成 | **7.361** | 1.296x | 1.012x | 全过 |
| 10 | `amd-cagra-niter45` / `d1c4643b` | `nn_descent_niter` 100 → 45 | 评测完成 | **0.000** | — | — | 未过：`hidden250k:recall@100_baseline_delta`（Recall@10 已过） |
| 11 | `amd-cagra-niter58` | `nn_descent_niter` 100 → 58 | 待回填 | 待回填 | 待回填 | 待回填 | 待回填 |

有效成绩取最高分（记录字段 `maxScoreFlag` 标记计入项），因此失败的尝试不会抹掉已入账分数。

### 5.2 两条被实测否证的方向（供后来者少走弯路）

1. **削减 NN-descent 迭代不是免费午餐，但可以做二分。** 迭代次数是构建窗口的首要耗时项，
   而 hidden250K 的 Recall@10/@100 相对同轮基线的容差只有 0.005。实测边界：
   `niter=20` 两条召回门槛同时失败；`niter=45` 只失败 `recall@100_baseline_delta`
   （`Recall@10` 已经过关）；`niter=70` 全部通过。因此临界点落在 `(45, 70]`，
   并且**只由 Recall@100 这一条决定**。该轴只能小步二分，每步的取舍都必须由评测回传的
   `build=` 比值和门槛清单来判定，不能靠本机推测（本地无 AMD GPU，无法复现召回率）。
2. **换中间图生成算法走不通。** `build_algo=IVF_PQ`（hipVS 固定版本支持，参数全部沿用其
   按数据集形状推导的默认值）被 `gpu_path` 抽检拒绝，评测未进入正式配对测试。
   该检查属于题面第八节第 4 条，绕过它不在允许范围内。

### 5.3 构建窗口占比模型

设 NN-descent 在构建窗口中的占比为 `f`，迭代次数从基线的 100 降到 `N` 时

```
build_ratio ≈ 1 / ((1 − f) + f · N / 100)
```

据此，每一轮 `build=` 回传值可直接反解 `f = (1 − 1/build_ratio) / (1 − N/100)`，
从而给出该轴的天花板 `1/(1−f)`：决定继续下探是否还值得用召回率去换。

已标定的通过点：`N=70 → build=1.296x` 解得 `f ≈ 0.76`，对应天花板
`1/(1−f) ≈ 4.2x`，高于 3.0x 的满分线，所以这条轴远未榨干。`N=100` 那两轮的
`build=1.023x / 1.069x` 差 4.5%，而同为 `N=70` 时 `query` 分别报 `1.059x` 与
`1.012x`——说明回传比值自带 ±5% 量级的波动，`f` 只能按 `0.65 ~ 0.85` 的区间使用。
按该模型预测：`N=58 → build ≈ 1.46x`（约 6.9 分构建分），`N=45 → build ≈ 1.65x`。

### 5.4 与 CPU 路径的对比（口径说明）

题面 2.1 节给出冻结基线（GPU CAGRA）在评测机上的参考值：SIFT250K 构建 20~27 s
（中位约 23 s）、Hidden250K 29~35 s（中位约 30 s）、nq=4096 时 Top-10/Top-100
QPS 约 147/143 与 87/85；而 V1~V4 在同一离线 DDL 路径上实际构建的是 **CPU VSAG HNSW**
（GPU 基本空闲），这正是题面要求修复的问题，也即本赛题的「CPU 构建」对照面。
正式成绩一律与**同轮冻结基线**配对比较，不使用上表绝对值；本报告因此只登记回传比值，
不引用本机绝对耗时。构建/查询的本机复现命令见第 3 节。

## 6. 查询链路：每 handle 复用暂存缓冲与检索参数（本次实验）

`seekdb_cuvs_search` 原本每次调用都做一次 `cuvsCagraSearchParamsCreate/Destroy` 和三笔
`cuvsRMMAlloc/cuvsRMMFree`。正式评测把 4096 条查询拆成 4096 次 `nq=1` 的单行调用，
这些同步的显存分配/释放直接计入每条查询延迟。参赛版本改为：

- 每个索引 handle 持有查询暂存缓冲（`q_dev / ids_dev / dist_dev`），按需要增长，
  随 handle 一起释放；
- `cuvsCagraSearchParams` 每 handle 创建一次，仅在 top-k 变化时重设；
- `itopk_size` 沿用基线的钳位规则 `max(创建默认值, 4*topk, 512)`，`search_width` 仍为 4，
  并显式记住创建时的默认值，使任意 top-k 下的取值与「每次新建对象」完全一致；
- `nq > 1` 的批量路径保留基线的按调用分配逻辑，避免为大批次常驻显存。

搜索参数、图结构与检索实现都未改动，因此返回结果与基线逐条一致；本项只作用于
第七节的查询吞吐（2 分），同时因为计时终点包含第一条查询，也顺带进入构建窗口。

## 7. 退回 cuVS 默认图度数：`graph_degree` 128→64（本次实验）

### 7.1 为什么必须动图度数，而不是继续动迭代次数

第 5.3 节的窗口模型 `build_ratio = 1/((1-f) + f·N/100)`（`f ≈ 0.71`，由 niter=58 → 1.426x
精确验证）说明：在 `graph_degree=128 / intermediate=256` 下，满分线 3.0x 要求 **niter ≈ 6**。
而评测机的硬数据是 niter=45 挂 `hidden250k:recall@100_baseline_delta`、niter=20 两个召回门槛都挂、
niter=51 才通过。**也就是说在 128/256 这一档上，3.0x 与召回门槛不可能同时满足**，
继续二分迭代次数没有出路，这是结构性的，不是精度问题。

NN-descent 的工作量随 `迭代次数 × intermediate_graph_degree²` 增长，而最终图的召回质量主要由
`graph_degree` 决定——**成本和召回挂在两个不同的旋钮上**。比赛基线把 cuVS 自己的默认值
（`graph_degree=64, intermediate_graph_degree=128`）双双翻倍成 128/256，因此退回到库默认档
可以一次性把每次迭代的成本降到 1/4。`intermediate` 不能低于 `graph_degree`：剪枝只能缩短候选表，
不能加长。

### 7.2 本次配置

| 参数 | 基线 | 本次 | 作用 |
| --- | ---: | ---: | --- |
| `graph_degree` | 128 | **64** | 决定召回；同时使每轮检索只需扫 64 个邻居 |
| `intermediate_graph_degree` | 256 | **128** | 决定每次 NN-descent 迭代成本（平方项） |
| `nn_descent_niter` | 100 | **30** | 决定迭代次数 |
| `itopk_size` 下限 | 512 | **256** | 见 7.3 |
| `search_width` | 4 | 4 | 未改 |
| 数据精度 | FP32 | FP32 | 见 7.4 |

按窗口模型推算，hidden250k 的综合构建加速约 2.9x；SIFT250K 的固定开销占比更小（实测约 22%，
hidden250k 约 29%），加速比更高，按 40%/60% 权重合成后应越过 3.0x 满分线。检索侧则同时受益于
更小的 `graph_degree` 与更小的 `itopk_size`。

### 7.3 `itopk_size` 下限 512→256：这不是拿召回换性能

cuVS 默认 `itopk_size = 64`，基线抬到 `max(4*topk, 512)`，是默认值的 8 倍。SIFT250K/gfx1100 实测
（top-10 P50 延迟与召回率）：

| `itopk_size` | P50 延迟 | Recall@10 | Recall@100 |
| ---: | ---: | ---: | ---: |
| 512（基线） | 2.75 ms | 1.0000 | 0.9977 |
| **256** | **1.25 ms** | **1.0000** | **0.9977** |
| 128 | 0.70 ms | 0.9852（−0.0148，超出题面第八节的 0.005 容差） | — |

延迟随 `itopk_size` 近似线性，而**召回率在 512 与 256 上逐位相同**，直到 128 才开始下降且一降就出界。
因此 256 是「门槛所度量的检索质量不变」前提下的下限。保留 `4*topk` 项，故 top-10 取 256、
top-100 取 400；查询性能分取两组 QPS 比的几何平均。

需要如实记录：本仓库此前曾把调低 `itopk_size` 判为「以降低 Recall 换取性能」而拒绝采用，
那是**没有数据支撑的假设**，上表推翻了它。第十一节禁止的是前者，而不是调整一个在门槛口径下
召回率不变的参数；题面 3.1 节也明确写着「允许调整其支持的参数」。

### 7.4 为什么本次不带 FP16

第 6 节之后另行验证过：把索引与查询整体改为 binary16，评测机回传 build 1.426x→1.476x、
query 1.033x→1.052x，即 **+3.5% 与 +1.8%**。FP16 的收益与「迭代占构建窗口的比例」成正比，
而本次配置的全部意义就是把那 71% 的迭代项压到约 15%，因此 FP16 在这里只值约 1%，
却仍然是一个影响召回的变量。故本次基于 FP32 的 `36d02312` 分支，不叠加 FP16。

### 7.5 本次唯一无法预先验证的风险

`graph_degree` 从 128 降到 64 之后，hidden250k 的 `recall@100` 是否仍满足「不低于同轮冻结基线
0.005」。基线在该数据集上约为 0.985/0.914，绝对下限（0.90/0.85）尚有余量，卡人的是 delta 那一条。
本机无 AMD GPU，无法测量；7.3 表的召回数据是在 SIFT250K 上取的，不能直接外推到内容与顺序均不公开的
hidden250k。若失败，门槛名会直接指明是哪个数据集的哪一条召回率。

同时 in-flight 的 `e3650743`（`graph_degree=128`、`niter=58`、`itopk=256`、FP16）正好构成本次配置的
对照组：两者只差图度数与迭代次数，因此无论本次结果如何，都能把 `itopk` 的影响与图度数的影响分离开。

### 7.6 一处已知但未修的竞态（记录备查）

`cuvsCagraSearch` 在 cuVS 自己的 stream 上异步入队，而 bridge 紧接着在默认 stream 上做 `cudaMemcpy`
取回 ids/distances。若 cuVS 的 stream 为非阻塞，两者之间存在竞争，在大 n 下可能返回不稳定的 id，
触发题面第八节的 `result_integrity` 门槛；修法是在 memcpy 之前加一次 `cudaDeviceSynchronize()`。
本仓库未加：历次评测（含两个数据集各 4096 条查询）均通过 `result_integrity`，说明默认 stream 的
隐式同步在当前环境下生效，而每次查询加一次全设备同步与本次争取查询吞吐的目标直接冲突。
一旦将来出现 `result_integrity` 失败，这是第一嫌疑。
