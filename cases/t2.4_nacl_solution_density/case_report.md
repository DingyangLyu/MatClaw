# QUASAR Tier II.4: NaCl Solution Density

**Result: ✅ PASS — 展示自主安装软件能力**

## 核心亮点：自主安装缺失软件

本 case 的核心目的是验证 Agent **在容器中自主发现、安装并使用未预装软件**的能力。

### 安装过程（三次尝试，逐步适应）

1. ❌ `apt-get install packmol` → **Permission denied**（node 用户无 apt 权限）
2. ❌ `conda install -c conda-forge packmol` → **EnvironmentNotWritableError**（base 环境 `/opt/conda` 属于 root）
3. ✅ `conda create -n sim -c conda-forge packmol` → **成功！** 在用户目录 `/home/node/.conda/envs/sim/` 创建新环境

Agent 没有放弃，也没有跳过 packmol 用别的方法凑合。它**理解了权限模型**（用户无 root 权限 → base conda 不可写 → 创建用户级新环境），展现了真正的自主问题解决能力。

## 测试信息

- **Date:** 2026-03-06
- **Image:** `matclaw-agent:latest`
- **Log:** `quasar_20260306_203646.log`
- **参考答案:** ρ ≈ 1.038 g/cm³ (1 mol/L NaCl, 298K, 1 atm)

## 结果

| | Agent | 参考值 | 偏差 |
|---|---|---|---|
| 密度 | **1.032 ± 0.008 g/cm³** | 1.038 g/cm³ | **-0.6%** |

## Agent 完整工作流

```
发现 packmol 未安装
    ↓
apt 失败 → conda base 失败 → conda create 新环境 ✅
    ↓
编写 packmol 输入（water.xyz + na.xyz + cl.xyz + packmol.inp）
    ↓
packmol 生成 solution.xyz（1000 水 + 18 Na⁺ + 18 Cl⁻）
    ↓
编写 xyz2lammps.py 转换为 LAMMPS data file
    ↓
设置力场（SPC/E 水 + OPLS-AA 离子）
    ↓
LAMMPS: 能量最小化 → NVT 平衡 → NPT 平衡 → Production
    ↓
编写 extract_density.py 提取密度
    ↓
结果: 1.032 ± 0.008 g/cm³ ✅
```

## 模拟参数

- **体系:** 1000 SPC/E 水 + 18 Na⁺ + 18 Cl⁻ = 3036 atoms
- **浓度:** ~1 mol/L NaCl
- **力场:**
  - 水: SPC/E (ε_OO = 0.1554 kcal/mol, σ_OO = 3.166 Å)
  - Na⁺: OPLS-AA (ε = 0.130 kcal/mol, σ = 2.35 Å, q = +1.0)
  - Cl⁻: OPLS-AA (ε = 0.100 kcal/mol, σ = 4.40 Å, q = -1.0)
  - 混合规则: Lorentz-Berthelot
  - 静电: PPPM, cutoff 10 Å
- **模拟流程:**
  - 能量最小化 (CG)
  - SHAKE 约束水几何
  - NVT 平衡: 2000 steps (4 ps), 298 K
  - NPT 平衡: 5000 steps (10 ps), 298 K, 1 atm
  - Production NPT: 7197 steps (14.4 ps)
  - 时间步: 2 fs
- **统计:** 73 个采样点 (每 100 步一个)

## 为什么这个 case 重要

1. **自主安装软件** — 不是预装好的工具，Agent 需要自己判断、安装、配置
2. **多步工作流** — packmol 构型 → 格式转换 → LAMMPS 模拟 → 后处理
3. **多组分体系** — 水 + 阳离子 + 阴离子，比纯水复杂得多
4. **权限适应** — 面对权限限制时能自主找到替代方案
5. **结果准确** — 0.6% 偏差，物理上完全合理

## 文件

- `packmol.inp` — packmol 输入
- `water.xyz`, `na.xyz`, `cl.xyz` — 分子模板
- `solution.xyz` — packmol 生成的初始构型
- `xyz2lammps.py` — 坐标转换脚本
- `data.nacl` — LAMMPS data file
- `run2.in` — LAMMPS 输入脚本
- `simulation.out` — 模拟完整输出
- `extract_density.py` — 密度提取脚本
- `RESULTS.md` — Agent 生成的结果总结
