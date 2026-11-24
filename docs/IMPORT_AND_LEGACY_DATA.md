# 导入精简版本和恢复历史原始文件

`fix/patch-freshness-v1` 延续完整研究历史。`codex/import-29829e4-lite` 是本次清理后的同一文件树的单提交导入快照，不随以后的实验更新。按分支导入时可选择后者。默认 `main` 不是这条研究分支。

仅移出了 [清单](data/archived_legacy_artifacts.json) 中的三个原始附件。它们不是都错了；其中包括有价值的旧对照。历史报告中的路径仍指向当时版本。如需复算涉及它们的旧实验，先在完整仓库根目录恢复所需文件：

```sh
git restore --source=29829e48da41bf421074434c031ff5609643aa20 --worktree -- docs/data/b3_boundary.json
```

其他两个文件同理。仅有导入快照的浅克隆可先 `git fetch origin fix/patch-freshness-v1` 获取完整分支（ZIP 本身不含 Git 历史）。对照清单逐文件检查 SHA-256；不要用旧文件覆盖 `rgb_oracle_pairs_fresh.npz` 或 `repair_value_raw.tar.gz`。
