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
| `nn_descent_niter` | 100 | 100 | 保持基线迭代次数 |

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

平台统一评测结果：

<!-- 平台结果回填 -->

| 提测 | Commit | 任务状态 | 成绩 | 构建加速 | 查询加速 | 门槛 |
|---|---|---|---|---|---|---|
| 基线占位 | `634888d9` | 待回填 | 待回填 | 待回填 | 待回填 | 待回填 |
| NN-descent 20 | `92cfabde` | 待回填 | 待回填 | 待回填 | 待回填 | 待回填 |

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
