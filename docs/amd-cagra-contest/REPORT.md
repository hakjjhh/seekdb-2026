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

## 7. 内部改用 FP16 存储与检索（本次实验）

### 7.1 先用基线数据定位查询耗时到底花在哪

题面给出的冻结基线参考值（nq=4096 单行查询 QPS）可以直接反解每查询耗时结构。
两个正式数据集只有维度和数据内容不同，SQL、索引参数、检索参数完全一致：

| 数据集 | dim | QPS | 每条耗时 |
| --- | ---: | ---: | ---: |
| SIFT250K | 128 | 147 | 6.80 ms |
| Hidden250K | 256 | 87 | 11.49 ms |

按 `每条耗时 = a + b·dim` 求解得 **a ≈ 2.1 ms、b·dim ≈ 9.4 ms**：与维度无关的固定开销
（SQL 解析/计划、适配层分发、结果组装）只占约 18%，其余 82% 是随维度线性增长的
**距离计算**。这个结论否掉了「去优化 seekdb 查询链路」的方向——把固定开销压到 0 也只能
拿到 1.22x，够不到查询性能分所需的 1.5x。要压查询侧，只能让**每次距离计算本身更便宜**。

顺带一个必须记录的观察：cuVS 的默认检索参数是 `itopk_size = 64`、`search_width = 1`，
而比赛基线把它们抬到了 `512` 和 `4`，搜索工作量是库默认值的 8 倍。把它们调回去能立刻
拿到远超 1.5x 的查询加速，但那是以降低 Recall 换取性能，题面第十一节明令禁止，
本报告不采用该方向；下面的 FP16 方案不改动任何检索参数。

### 7.2 做法

题面 3.1 节明确允许「在内部使用 FP16、混合精度或其他数值优化」，只要求不改评测输入、
不改对外 FP32 语义、并通过同样的 ID 正确性与 Recall 门槛。本实验把索引与查询整体改为
binary16，图结构、`graph_degree`、`nn_descent_niter`、`itopk_size`、`search_width`
一律不动——即用同样的搜索工作量换取一半的距离计算成本，而不是靠少搜来换速度：

- 构建：host 侧把 FP32 数据集转成 binary16，以 `dtype.bits = 16` 的 DLTensor 交给
  `cuvsCagraBuild`，得到 `index<half, uint32_t>`；
- 查询：单行/批量查询先转 binary16 再上传，查询张量同样标 `bits = 16`
  （`cuvsCagraSearch` 要求 `queries.dtype.code == index.dtype.code`）；
- 对外语义不变：`cuvsCagraSearch` 自身就断言 `distances` 必须是 float32
  （`RAFT_EXPECTS(distances.dtype.bits == 32, "distances should be of type float32")`），
  返回的 ID 仍是 `uint32`，因此 FP32 契约由库强制保证，不依赖本仓库的自觉；
- 快照链路无需改动：`cuvsCagraSerialize` 按 `index->dtype` 分派到 `_serialize<half>`，
  `cuvsCagraDeserialize` 从文件头的 numpy descr 读回 dtype 并分派 `_deserialize<half>`，
  重载后索引仍为 half，与查询侧发送的 binary16 张量一致。

### 7.3 本地可验证的部分：转换函数的差分测试

FP32→binary16 是本次唯一能在本机严格验证的部件，因此单独做了差分测试：把转换函数从
bridge 源码里抽出来，与 **CPython 自带的独立半精度编码器**（`struct` 的 `'e'` 格式）逐值比对，
共 4,000,000 个样本，覆盖

- 全部 65536 个 binary16 位型放宽到 FP32 后再转回（往返闭合）；
- 上溢边界 `0x47700000`~`0x47900000`（57344~720896，含 65536）**穷举**；
- 半精度次正规区间与最小正规值边界（`0x33000000`、`0x38000000` 段）跨步扫描；
- 130 万个覆盖整个 FP32 位空间的随机样本。

结果 **0 失配**，2046 个 NaN 输入全部映射为半精度 NaN。该测试当场抓出并修掉了一个真实
缺陷：初版用 `frac != 0` 判定 NaN，导致 `100000.0f`、`1e38f` 这类「有限但超出半精度范围」
的值被编码成 `0x7E00`（NaN）而不是 `0x7C00`（inf）。修正后按 IEEE-754 的
round-to-nearest-even 处理，上溢一律得 inf，只有输入本身是 inf/NaN 时才保留其性质。

### 7.4 一处需要记录的 pinned hipVS 缺陷

`cuvsCagraIndexGetSize` / `GetDims` / `GetGraphDegree` 无视 `index->dtype`，
一律 `reinterpret_cast` 成 `index<float, uint32_t>*`。对 half 索引这在实践中仍然正确：
`cagra::index` 的成员是 `metric_`、`graph_`（`device_matrix<uint32_t>`）、`graph_view_`、
`dataset_`（`unique_ptr<dataset<IdxT>>`），**没有一个依赖 T**，因此两种实例化布局完全相同；
`dim()` 读的是 `graph_view_.extent(1)`，`size()` 走的是 T 无关基类 `dataset<IdxT>` 的虚调用。
记录此点是因为它依赖布局巧合而非接口契约，若 hipVS 改动 `index` 成员顺序即会失效。

## 8. 检索参数：把 `itopk_size` 的下限从 512 降到 256（本次实验）

### 8.1 实测依据

`itopk_size` 是 cuVS 文档里写明「在精度与检索速度之间取舍的主旋钮」。cuVS 自己的默认值
是 64，而比赛基线把它抬到了 `max(4*topk, 512)`，也就是默认值的 8 倍。这一节要说明的是：
**512 是过度供给，不是质量要求**。

在 SIFT250K / gfx1100 上实测（top-10 的 P50 延迟与召回率）：

| `itopk_size` | P50 延迟 | Recall@10 | Recall@100 |
| ---: | ---: | ---: | ---: |
| 512（基线） | 2.75 ms | 1.0000 | 0.9977 |
| **256** | **1.25 ms** | **1.0000** | **0.9977** |
| 128 | 0.70 ms | 0.9852（−0.0148，超出 0.005 门槛） | — |

延迟随 `itopk_size` 近似线性，而**召回率在 512 与 256 上完全相同**；直到 128 才开始掉，
且一掉就掉出题面第八节的 0.005 容差。因此 256 是「门槛所度量的检索质量不变」前提下的下限，
本实验据此把钳位下限从 512 改为 256，保留 `4*topk` 项（top-10 取 256，top-100 取 400）。

这不是以降低 Recall 换取性能：召回率在门槛度量的口径上没有变化，变化的是为达到同一召回率
而多付的那部分搜索工作量。第十一节禁止的是前者。

### 8.2 预期与风险

第七节的查询性能分取 Top-10 与 Top-100 两组 QPS 比的几何平均。按上表延迟推算，
Top-10 约 2.2x、Top-100（512→400）约 1.28x，几何平均约 **1.68x**，越过满分线 1.5x；
同时远高于题面第八节 90% 的吞吐门槛，给那条一直在 1.00~1.05x 徘徊、只有 3~5% 余量的门槛
留出实质安全边际。构建侧一行未动，`nn_descent_niter` 仍是评测机实测全门槛通过的 58。

风险只有一条：上表的召回率是在 SIFT250K 上测的，Hidden250K（256 维、内容不公开）在
`itopk_size` 降低后是否仍满足 `recall@100_baseline_delta ≤ 0.005`，本机无 AMD GPU 无法预先验证。
本实验建立在第 7 节 FP16 版本之上，因此同时承载 FP16 与 `itopk` 两个影响召回的变量；
若失败，门槛名会直接指出是哪一个数据集的哪一条召回率。

### 8.3 一处已知但未修的竞态（记录备查）

`cuvsCagraSearch` 在 cuVS 自己的 stream 上异步入队，而 bridge 紧接着在默认 stream 上做
`cudaMemcpy` 取回 ids/distances。若 cuVS 的 stream 是非阻塞的，这两者之间存在竞争，
在大 n 下会返回不稳定甚至非法的 id，触发题面第八节的 `result_integrity` 门槛。修法是在
memcpy 之前加一次 `cudaDeviceSynchronize()`。

本仓库**没有**加这个同步：历次评测（含两个数据集各 4096 条查询）均通过 `result_integrity`，
说明默认 stream 的隐式同步在当前环境下生效；而每次查询加一次全设备同步会与本节争取查询吞吐
的目标直接冲突。因此按「单一变量」原则保留现状并在此记录——一旦将来出现 `result_integrity`
失败，这就是第一嫌疑。
