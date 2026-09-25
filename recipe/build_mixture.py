# -*- coding: utf-8 -*-
"""Qwen3.6-35B-A3B 离线蒸馏 SFT 配方生成器。
改 THROUGHPUT / 各域 records & epochs 即可重算全部预算。"""
import csv, json

# ---- 实测吞吐基线：上一轮 2.87B token / ~3.25h / 64xH200 ----
PREV_TOKENS_B, PREV_HOURS, GPUS = 2.87, 3.25, 64
TPUT_MEASURED = PREV_TOKENS_B / PREV_HOURS          # B input-token / hour @64GPU
TPUT_PLAN     = 0.80                                # 长轨迹占比更高，保守规划值

# domain: (id, 名称, 记录数, 输入均长, 监督均长, 轮数, 类别)
DOM = [
 ("A1","通用工具调用 / MCP schema", 200_000,  4_500, 1_300, 2.5,"agent"),
 ("A2","多轮事务与客服",             95_000, 15_000, 3_500, 2.5,"agent"),
 ("A3","工具边界 / 歧义 / 不调用",    35_000,  2_500,   600, 3.0,"agent"),
 ("A4","搜索与证据综合",             55_000, 20_000, 6_000, 2.5,"agent"),
 ("A5","终端与环境操作",             70_000, 16_000, 4_500, 1.5,"agent"),
 ("A6","仓库修复 SWE",               40_000, 32_000, 8_000, 1.0,"agent"),
 ("A7","技能 / 办公 / 文件",          42_000, 18_000, 5_000, 2.5,"agent"),
 ("A8","数据分析与产物构建",          25_000, 18_000, 5_500, 2.5,"agent"),
 ("A9","GUI / 浏览器（多模态）",       5_000, 12_000, 3_000, 3.0,"agent"),
 ("G1","中文对话与写作",            240_000,  2_500, 1_700, 2.0,"general"),
 ("G2","指令遵循与结构化输出",       140_000,  2_200, 1_300, 2.0,"general"),
 ("G3","数学",                       90_000,  7_000, 6_300, 1.0,"general"),
 ("G4","代码（非 agent）",            90_000,  6_000, 5_000, 1.0,"general"),
 ("G5","知识与 STEM",                90_000,  3_500, 2_600, 2.0,"general"),
 ("G6","英文与多语言",               70_000,  2_800, 1_800, 2.0,"general"),
 ("G7","安全与边界",                 25_000,  1_800,   900, 2.5,"general"),
 ("G8","多模态保持（图/视频）",       25_000,  3_000,   600, 1.5,"general"),
 ("G9","长上下文保持",               10_000, 64_000, 2_000, 1.0,"general"),
]

rows=[]
for i,name,rec,li,ll,ep,cat in DOM:
    uin, ulab = rec*li/1e9, rec*ll/1e9
    rows.append(dict(id=i,name=name,cat=cat,rec=rec,li=li,ll=ll,ep=ep,
                     uin=uin,ulab=ulab,ein=uin*ep,elab=ulab*ep))

T=lambda k,c=None: sum(r[k] for r in rows if c is None or r["cat"]==c)
ein,elab = T("ein"), T("elab")
a_ein,a_elab = T("ein","agent"), T("elab","agent")

print(f"实测吞吐 {TPUT_MEASURED:.2f} B/h @{GPUS}xH200（{PREV_TOKENS_B}B / {PREV_HOURS}h）；规划用 {TPUT_PLAN:.2f} B/h\n")
print(f"{'域':<4}{'名称':<26}{'记录':>9}{'输入均长':>9}{'监督均长':>9}{'轮':>5}{'曝光输入B':>11}{'监督B':>9}{'监督占比':>9}")
print("-"*95)
for r in rows:
    if r["id"]=="G1": print("-"*95)
    print(f"{r['id']:<4}{r['name']:<26}{r['rec']:>9,}{r['li']:>9,}{r['ll']:>9,}{r['ep']:>5.1f}"
          f"{r['ein']:>11.2f}{r['elab']:>9.2f}{r['elab']/elab*100:>8.1f}%")
print("-"*95)
print(f"{'':<4}{'AGENT 小计':<26}{T('rec','agent'):>9,}{'':>9}{'':>9}{'':>5}{a_ein:>11.2f}{a_elab:>9.2f}{a_elab/elab*100:>8.1f}%")
print(f"{'':<4}{'GENERAL 小计':<26}{T('rec','general'):>9,}{'':>9}{'':>9}{'':>5}{ein-a_ein:>11.2f}{elab-a_elab:>9.2f}{(elab-a_elab)/elab*100:>8.1f}%")
print(f"{'':<4}{'合计':<26}{T('rec'):>9,}{'':>9}{'':>9}{'':>5}{ein:>11.2f}{elab:>9.2f}")
print(f"\n唯一记录 {T('rec'):,} 条 / 唯一输入 {T('uin'):.2f}B / 平均 {ein/T('uin'):.2f} 遍")
print(f"曝光输入 {ein:.2f}B token → 64 卡约 {ein/TPUT_PLAN:.1f} h（{ein/TPUT_MEASURED:.1f} h 按实测）")
print(f"监督 token {elab:.2f}B，其中 agent {a_elab/elab*100:.0f}%，通用 {(elab-a_elab)/elab*100:.0f}%")
print(f"全局监督率 {elab/ein*100:.0f}%（agent {a_elab/a_ein*100:.0f}% / 通用 {(elab-a_elab)/(ein-a_ein)*100:.0f}%）")

with open("domain_budget.csv","w",newline="",encoding="utf-8-sig") as f:
    w=csv.writer(f); w.writerow(["域","名称","类别","唯一记录","输入均长","监督均长","轮数",
                                 "唯一输入B","曝光输入B","曝光监督B","监督占比%"])
    for r in rows: w.writerow([r["id"],r["name"],r["cat"],r["rec"],r["li"],r["ll"],r["ep"],
                               round(r["uin"],3),round(r["ein"],3),round(r["elab"],3),round(r["elab"]/elab*100,2)])
json.dump({"throughput_measured_B_per_h":round(TPUT_MEASURED,3),"throughput_plan_B_per_h":TPUT_PLAN,
           "unique_records":T("rec"),"exposure_input_B":round(ein,2),"exposure_label_B":round(elab,2),
           "agent_label_share":round(a_elab/elab,4),"est_hours_64gpu":round(ein/TPUT_PLAN,1),
           "domains":rows}, open("domain_budget.json","w"), ensure_ascii=False, indent=2)
