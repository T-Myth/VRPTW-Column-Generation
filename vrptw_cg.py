import gurobipy as gp
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import networkx as nx
from gurobipy import GRB

# ================== 数据读取 ==================
df_node = pd.read_excel("参考算例.xlsx", sheet_name="节点属性信息", index_col=0)
df_time = pd.read_excel("参考算例.xlsx", sheet_name="旅行时间矩阵", index_col=0)

n_customers = len(df_time) - 1
customers = list(range(1, n_customers + 1))
dist = df_time.loc[[0] + customers, [0] + customers].values

e = df_node.loc[[0] + customers, "开始服务时间下界"].values
l = df_node.loc[[0] + customers, "开始服务时间上界"].values
q = df_node.loc[[0] + customers, "需求量"].values
s = df_node.loc[[0] + customers, "服务时间"].values
Q = df_node.loc[0, "车容量"]

EPS = 1e-6

# ================== 标签类 ==================
class Label:
    def __init__(self, cost=0.0, time=0.0, load=0.0, visited=0, node=0, parent=None):
        self.cost = cost
        self.time = time
        self.load = load
        self.visited = visited
        self.node = node
        self.parent = parent

    def dominates(self, other):
        if self.cost > other.cost + EPS:
            return False
        if self.time > other.time + EPS:
            return False
        if self.load > other.load + EPS:
            return False
        if (self.visited & ~other.visited) != 0:
            return False
        return True

# ================== ESPPRC 子问题 ==================
def solve_espprc(e, l, q, s, Q, dist, pi):
    n = dist.shape[0]
    best_rc = 0.0
    best_label = None
    labels = [[] for _ in range(n)]

    start = Label(cost=0.0, time=0.0, load=0.0, visited=0, node=0, parent=None)
    labels[0].append(start)

    for cur_node in range(n):
        for lab in labels[cur_node]:
            for j in range(1, n):
                if (lab.visited >> j) & 1:
                    continue
                new_load = lab.load + q[j]
                if new_load > Q + EPS:
                    continue
                arrival = lab.time + dist[lab.node][j]
                start_time = max(arrival, e[j])
                if start_time > l[j] + EPS:
                    continue

                new_time = start_time + s[j]
                new_cost = lab.cost + dist[lab.node][j] - pi[j - 1]
                new_visited = lab.visited | (1 << j)
                new_label = Label(cost=new_cost, time=new_time, load=new_load,
                                  visited=new_visited, node=j, parent=lab)

                return_cost = new_cost + dist[j][0]
                if return_cost < best_rc - EPS:
                    best_rc = return_cost
                    best_label = new_label

                dominated = False
                survivors = []
                for old in labels[j]:
                    if old.dominates(new_label):
                        dominated = True
                        break
                    if not new_label.dominates(old):
                        survivors.append(old)
                if not dominated:
                    survivors.append(new_label)
                    labels[j] = survivors

    if best_rc < -EPS and best_label is not None:
        route = [0]
        temp = []
        ptr = best_label
        while ptr is not None and ptr.node != 0:
            temp.append(ptr.node)
            ptr = ptr.parent
        route.extend(reversed(temp))
        route.append(0)
        real_cost = sum(dist[route[k]][route[k+1]] for k in range(len(route)-1))
        return route, real_cost, best_rc
    else:
        return None, None, 0.0

# ================== 路线可行性检查 ==================
def is_feasible(route):
    load = 0.0
    t = 0.0
    for k in range(len(route) - 1):
        i, j = route[k], route[k+1]
        t += dist[i][j]
        if j != 0:
            t = max(t, e[j])
            if t > l[j] + EPS:
                return False
            t += s[j]
            load += q[j]
            if load > Q + EPS:
                return False
    return True

# ================== 初始路线随机生成 ==================
def generate_initial_routes():
    routes = []
    custs = list(range(1, n_customers + 1))
    random.shuffle(custs)

    for c in custs:
        inserted = False
        if routes:
            indices = list(range(len(routes)))
            random.shuffle(indices)
            for idx in indices:
                rt = routes[idx]
                positions = list(range(1, len(rt)))
                random.shuffle(positions)
                for pos in positions:
                    new_rt = rt[:pos] + [c] + rt[pos:]
                    if is_feasible(new_rt):
                        routes[idx] = new_rt
                        inserted = True
                        break
                if inserted:
                    break
        if not inserted:
            if is_feasible([0, c, 0]):
                routes.append([0, c, 0])
            else:
                raise RuntimeError(f"客户 {c} 不可行")
    return routes

def routes_to_columns(routes):
    nr_list = []
    cost_list = []
    for rt in routes:
        nr = np.zeros(n_customers, dtype=int)
        for node in rt[1:-1]:
            nr[node - 1] = 1
        nr_list.append(nr)
        cost_list.append(sum(dist[rt[k]][rt[k+1]] for k in range(len(rt)-1)))
    return nr_list, cost_list

# ================== 主问题构建 ==================
def build_master(Ar, cr_list):
    m = gp.Model("master")
    n_c, n_r = Ar.shape
    lam = m.addVars(n_r, vtype=GRB.CONTINUOUS, lb=0, name="lambda")
    m.setObjective(gp.quicksum(cr_list[r] * lam[r] for r in range(n_r)), GRB.MINIMIZE)
    for i in range(n_c):
        m.addConstr(gp.quicksum(Ar[i, r] * lam[r] for r in range(n_r)) >= 1)
    m.setParam('OutputFlag', 0)
    return m

# ================== 列生成主流程 ==================
random.seed(519)
init_routes = generate_initial_routes()
nr_list, cost_list = routes_to_columns(init_routes)
route_list = init_routes.copy()    # 保存完整路径

obj_hist = []
rc_hist = []
iter_count = 0

while True:
    iter_count += 1
    Ar = np.array(nr_list).T
    master = build_master(Ar, cost_list)
    master.optimize()

    obj_hist.append(master.objVal)
    pi = [c.Pi for c in master.getConstrs()]

    route, real_cost, rc = solve_espprc(e, l, q, s, Q, dist, pi)
    rc_hist.append(rc)
    print(f"Iter {iter_count:3d} | LP Obj: {master.objVal:8.2f} | RC: {rc:8.4f}")

    if rc > -EPS:
        break

    nr = np.zeros(n_customers, dtype=int)
    for node in route[1:-1]:
        nr[node - 1] = 1
    nr_list.append(nr)
    cost_list.append(real_cost)
    route_list.append(route)        # 记录路径

print("列生成收敛。")

# ================== 最终整数解 ==================
Ar_final = np.array(nr_list).T
mip = gp.Model("mip")
n_routes = Ar_final.shape[1]
lam_mip = mip.addVars(n_routes, vtype=GRB.BINARY, name="lambda")
mip.setObjective(gp.quicksum(cost_list[r] * lam_mip[r] for r in range(n_routes)), GRB.MINIMIZE)
for i in range(n_customers):
    mip.addConstr(gp.quicksum(Ar_final[i, r] * lam_mip[r] for r in range(n_routes)) >= 1)
mip.setParam('OutputFlag', 0)
mip.optimize()

# 输出方案
print("\n最终调度方案：")
used_routes = []
for r in range(n_routes):
    if lam_mip[r].X > 0.5:
        rt = route_list[r]
        used_routes.append(rt)
        print(f"车辆 {len(used_routes)}: {rt}")

print(f"\n使用车辆数: {len(used_routes)}")
print(f"最优总成本: {mip.objVal:.2f}")

# ================== 可视化 ==================
plt.rcParams['font.family'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))

ax1.plot(range(1, iter_count+1), obj_hist, 'o-', color='#2f7ed8')
ax1.set_title("线性松弛目标值收敛曲线")
ax1.set_xlabel("迭代次数")
ax1.set_ylabel("总旅行时间")
ax1.grid(alpha=0.3)

ax2.plot(range(1, iter_count+1), rc_hist, 'o-', color='#e43955')
ax2.axhline(y=0, color='green', linestyle='--')
ax2.set_title("子问题 Reduced Cost 收敛曲线")
ax2.set_xlabel("迭代次数")
ax2.set_ylabel("Reduced Cost")
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# ================== 路径网络图 ==================
G = nx.DiGraph()
G.add_node(0, label="仓库")
for i in range(1, n_customers+1):
    G.add_node(i, label=f"{i}")

# 为每辆车准备不同颜色
colors = plt.cm.tab10(range(len(used_routes)))

pos = nx.circular_layout(G)   # 仓库在中心
# 把仓库放在中心
pos[0] = np.array([0, 0])
# 客户均匀放在外圈
radius = 1.0
for i in range(1, n_customers+1):
    angle = 2 * np.pi * (i-1) / n_customers
    pos[i] = np.array([radius * np.cos(angle), radius * np.sin(angle)])

plt.figure(figsize=(8, 8))
# 先画节点
nx.draw_networkx_nodes(G, pos, node_size=300, node_color='lightgray')
nx.draw_networkx_labels(G, pos, font_size=9)

# 画每条路径
for idx, rt in enumerate(used_routes):
    edges = [(rt[k], rt[k+1]) for k in range(len(rt)-1)]
    nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=colors[idx % len(colors)],
                           width=2, alpha=0.8, arrows=True, arrowstyle='-|>', arrowsize=15)

plt.title("最终车辆路线图")
plt.axis('off')
plt.show()
