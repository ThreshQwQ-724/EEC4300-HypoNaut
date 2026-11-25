"""Build human-readable evidence and a static plot from actual offline result files."""
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path
from common import ROOT, HERE, RESULTS, COMMIT, digest, save


def main():
    files=['3d_audit','boolean_audit','causal_audit','causal_subset_audit','offline_3d_baseline']
    reports={k:json.loads((RESULTS/(k+'.json')).read_text(encoding='utf-8')) for k in files}
    a,b,c,cs,baseline=(reports[k] for k in files)
    versions={}
    for package in ['numpy','networkx','PyYAML','scipy','requests','sympy','matplotlib','ijson']:
        try: versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: versions[package]='not installed'
    provenance={'python':sys.version,'executable':sys.executable,'platform':platform.platform(),
                'packages':versions,'upstream_commit':COMMIT,
                'paper_sha256':digest(ROOT/'references/HypoSpace_ICML2026.pdf'),
                'scripts_sha256':{p.name:digest(p) for p in HERE.glob('*.py')}}
    save('environment.json',provenance)
    (HERE/'requirements-offline.txt').write_text('\n'.join(f'{k}=={v}' for k,v in versions.items() if v!='not installed')+'\n',encoding='utf-8')
    names={'repeat_ground_layer':'重复地面层','uniform_heights_iid':'均匀独立采样柱高','enumerate_heights':'完整枚举柱高（算法参照）'}
    metric_rows=[]
    for method,stats in baseline['aggregate'].items():
        metric_rows.append('| '+names[method]+' | '+' | '.join(f'{stats[k]["mean"]:.2%}' for k in ['valid_rate','novelty_rate','recovery_rate'])+' |')
    causal_rows=[]
    for n,g in c['groups'].items():
        causal_rows.append(f'| {n} | {g["instances"]} | {g["gt_min"]:,} | {g["gt_max"]:,} | {g["total_gt_entries"]:,} |')
    text=f'''# HypoSpace 数据与离线基线验证

本轮已完成论文阅读、原始数据检查、官方评测链路回放、小规模非 LLM 参照实验，以及可复现的评测修复。依照当前要求，没有调用模型 API、没有训练或优化模型，费用为 0。下表不代表任何 LLM 的性能。

## 已得到的基线

3D 数据取固定的 5 个任务，3 个种子（33550336–33550338），每个任务每种方法生成 27 个答案，总计 1,215 个候选。每种方法有 15 个任务与种子的组合；确定性方法在不同种子下相同。所有原始输出和逐次覆盖率保存在 `experiments/results/offline_3d_baseline.json`。

| 非 LLM 方法 | Validity | Uniqueness | Recovery |
|---|---:|---:|---:|
{chr(10).join(metric_rows)}

均匀采样的理论期望覆盖率为 `1 - (26/27)^27 = 63.904%`，实测为 {baseline['aggregate']['uniform_heights_iid']['recovery_rate']['mean']:.3%}，15 个任务与种子组合的标准差为 {baseline['aggregate']['uniform_heights_iid']['recovery_rate']['std_across_instance_seed_pairs']:.3%}（不是置信区间）。完整枚举利用公开的投影与高度约束生成答案，没有读取标准答案作为生成输入；它是这个小型可解任务的参照，不能声称为 LLM 改进或新方法。

![离线参照实验](experiments/results/offline_3d_controls.png)

## 数据检查结果

| 数据域 | 任务数 | 本轮检查范围 | 结论 |
|---|---:|---|---|
| 3D | {a['observations_checked']} | 全部 {a['gt_entries_checked']:,} 个答案；独立枚举所有投影及柱高组合 | 数据的合法性、唯一性、完整性检查通过 |
| Boolean | {b['observation_sets']} | {b['gt_entries']} 条答案记录，{b['distinct_stored_keys']} 种结构、{b['distinct_semantics']} 种真值表；重新运行公开文法生成器 | 数据键值与真值表匹配，但官方评测器与生成器的深度定义不一致 |
| Causal | {c['observation_sets']} | 流式扫描整个 {c['size_bytes']/1e9:.3f} GB 文件；核对 {c['graph_entries_counted']:,} 条答案记录的数量 | 结构与计数检查通过；抽查 {c['graph_entries_validated']:,} 个图的合法性全部通过 |

因果数据中的 12,256,219 是各任务的答案记录总数，包含跨任务重复的图；不是全局不同图的数量。本轮没有验证全部图的合法性，也没有独立证明因果答案集合的完整性。源数据的 `complete_gt_coverage=true` 只是源文件声明。额外导出 20 个答案数在 2–32 之间的完整小任务，共 {cs['gt_entries_checked']} 个答案，经官方解析、验证、计分回放均为 100%。该子集按文件顺序选择，仅用于流程验证，不代表总体难度。

因果数据每组任务的答案数量如下。直接执行组员 Guide 中的 `--query-multiplier 1.0` 会令每个任务的请求数等于其答案数；首轮不能照搬。

| 已观测干预数 | 任务数 | 最少答案 | 最多答案 | 答案记录总数 |
|---|---:|---:|---:|---:|
{chr(10).join(causal_rows)}

## 已复现的评测问题与局部修复

1. **Boolean 深度定义不一致。** 数据生成器把 NOR 算作一个基本操作，评测器却在 SymPy 树上将其展开成 NOT 加 OR 两层。例如 `x AND NOR(x,y)` 的文法深度是 2，但原评测器计算为 3。65 种标准表达式中有 {b['in_space_rejections']} 种被拒绝，影响 {b['gt_entries_rejected_by_official']} / {b['gt_entries']} 条答案记录；{b['instances_with_oracle_rr_below_one']} / 80 个任务即使回放全部标准答案也无法满分。修正 NOR 深度、递归检查 NOR 子表达式，并以公开文法重新生成的候选空间验证成员关系后，80 个任务的标准答案回放 VR、NR、RR 均为 100%。解析失败也明确判为无效，避免被误当作恒零函数。此处使用文法，不使用每道题的答案指导生成。
2. **3D 漏检高度上限。** 最大高度为 3 时，原评测器会把连续 4 层同一投影判为有效。修正版补上高度、形状、二值及地面层检查。
3. **3D 全零结构崩溃。** 原版归一化把 NumPy 数组重新交给使用 `if layer` 的构造函数，触发数组布尔值歧义错误。修正版统一转换为列表，非法空结构计为无效而不崩溃。

修复位于 `experiments/validated_evaluators.py`，通过子类适配完成，原始数据和固定版本官方代码均未修改。上面的 3D 基线在原版、修正版及独立计分程序上结果一致；Boolean 修正版与原版分数必须分别标注，不能直接声称优于论文结果。局部修复不等于已穷尽所有解析器边界情况。

## 如何复跑

在 PowerShell 中执行：

```powershell
& '.\\Final Project\\experiments\\run_offline.ps1'
# 同时重新扫描整个因果文件（本机约 81 秒）
& '.\\Final Project\\experiments\\run_offline.ps1' -RescanCausal
```

脚本使用现有 `D:\\Anaconda\\python.exe`，不会修改全局环境；缺失的 ijson 已单独安装到 `experiments/.deps`。完整包版本写入 `experiments/requirements-offline.txt`，运行时与脚本 SHA256 写入 `experiments/results/environment.json`。更换环境可使用 `-Python '新环境的 python.exe 路径'`。需要 Python 3.11 及该 requirements 中的包。

普通复跑复用已导出的因果小子集，不重新核验大型源文件；更新因果源文件后务必使用 `-RescanCausal`。

## 文件导航

- `PAPER_NOTES.md`：论文方法、理论、实验结论、复现差异和后续研究判断。
- `references/HypoSpace_ICML2026.pdf`：课程指定的正式发表版本，共 15 页。
- `upstream/_HypoSpace-{COMMIT}/`：官方代码固定快照。
- `experiments/offline_3d.py`：3D 全量数据验证、错误用例和小规模离线基线。
- `experiments/audit_boolean.py`：布尔数据检查、标准答案回放、原版与修正版对照。
- `experiments/stream_causal.py`：因果大文件流式检查和完整小任务提取。
- `experiments/audit_causal_subset.py`：因果小子集官方评测链路检查。
- `experiments/results/`：原始输出、指标、验证报告、源文件 SHA256 及图表。

## 接入真实 LLM 前的实验约定

先固定 3D 的这 5 个任务和相同预算，跑无历史独立采样与官方历史提示采样两个基础对照；每次候选输出均计入预算，不静默丢弃失败或重复答案。记录请求数、token、耗时、失败类型、模型与配置。连接模型后需固定开发集与最终测试集，避免反复在测试集上挑选提示词。3D 适合检查链路，Boolean 适合后续研究探索能力，但应双轨报告原版与修正版指标。

当前尚未做真实模型性能评测、参数优化、创新方法实现、显著性检验或课程最终成果制作。

来源：[课程指定论文](https://proceedings.mlr.press/v306/chen26bz.html)，[官方代码固定版本](https://github.com/CTT-Pavilion/_HypoSpace/tree/{COMMIT})。
'''
    (ROOT/'README.md').write_text(text,encoding='utf-8')
    os.environ.setdefault('MPLCONFIGDIR',str(HERE/'.mplconfig'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
    methods=list(names)
    labels=['Repeat one valid answer','Uniform IID heights','Enumerate all heights']
    colors=['#ad5050','#2677ac','#328b69']
    x=np.arange(3)
    for i,(method,label,color) in enumerate(zip(methods,labels,colors)):
        vals=[baseline['aggregate'][method][k]['mean']*100 for k in ['valid_rate','novelty_rate','recovery_rate']]
        axes[0].bar(x+(i-1)*.24,vals,.22,label=label,color=color)
        curves=np.array([[point['rr'] for point in r['coverage_curve']] for r in baseline['runs'] if r['method']==method])
        axes[1].plot(range(1,28),curves.mean(axis=0)*100,label=label,color=color,lw=2)
        if method=='uniform_heights_iid':
            axes[1].fill_between(range(1,28),np.maximum(0,curves.mean(axis=0)-curves.std(axis=0))*100,
                                 np.minimum(1,curves.mean(axis=0)+curves.std(axis=0))*100,color=color,alpha=.13)
    axes[0].set_xticks(x,['Validity','Uniqueness','Recovery'])
    axes[0].set_ylabel('Percent')
    axes[0].set_title('Final metrics after 27 proposals')
    axes[1].plot(range(1,28),[100*(1-(26/27)**n) for n in range(1,28)],'--',color='#222222',lw=1.2,label='Uniform theoretical expectation')
    axes[1].set_xlabel('Number of proposals')
    axes[1].set_ylabel('Recovery (%)')
    axes[1].set_title('Coverage as the budget grows')
    axes[1].legend(loc='upper left',fontsize=8,frameon=False)
    for ax in axes:
        ax.set_ylim(0,108)
        ax.grid(axis='y',alpha=.15)
        ax.set_axisbelow(True)
    fig.suptitle('Offline controls on HypoSpace 3D | NOT LLM results',fontsize=14,fontweight='bold')
    fig.savefig(RESULTS/'offline_3d_controls.png',dpi=170)
    plt.close(fig)
    print('Saved README.md, plot, requirements, and environment provenance.')


if __name__=='__main__':
    main()
