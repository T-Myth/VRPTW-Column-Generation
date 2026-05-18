import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 初始路线生成
def generate_initial_columns(READY, DUE, SERVICE, DEMAND, dist, N, CAP):
    DEPOT = 0
    all_routes = []
    all_costs = []
    unvisited = set(range(1, N+1))

    while unvisited:
        start_cus = next(iter(unvisited))
        route = [DEPOT, start_cus]
        unvisited.remove(start_cus)
        now_time = max(dist[0][start_cus], READY[start_cus])
        now_load = DEMAND[start_cus]
        last = start_cus

        while True:
            best = -1
            min_arr = 1e18
            for j in unvisited:
                if now_load + DEMAND[j] > CAP:
                    continue
                arr = now_time + dist[last][j]
                st = max(arr, READY[j])
                if st > DUE[j]:
                    continue
                if arr < min_arr:
                    min_arr = arr
                    best = j
            if best == -1:
                break
            route.append(best)
            unvisited.remove(best)
            arr = now_time + dist[last][best]
            now_time = max(arr, READY[best]) + SERVICE[best]
            now_load += DEMAND[best]
            last = best

        route.append(DEPOT)
        cost = sum(dist[route[i]][route[i+1]] for i in range(len(route)-1))
        all_routes.append(route)
        all_costs.append(cost)

    nr_list = []
    cr_list = all_costs
    route_list = all_routes
    for rt in all_routes:
        nr = np.zeros(N, dtype=int)
        for c in rt[1:-1]:
            nr[c-1] = 1
        nr_list.append(nr)
    return nr_list, cr_list, route_list

# 贪心子问题
def greedy_subproblem(pi, dist, READY, DUE, SERVICE, DEMAND, CAP, n_cus):
    route = [0]
    current_time = 0
    current_load = 0
    visited = set()

    while True:
        best_rc = float('inf')
        best_j = -1
        for j in range(1, n_cus+1):
            if j in visited:
                continue
            if current_load + DEMAND[j] > CAP:
                continue
            arr = current_time + dist[route[-1]][j]
            st = max(arr, READY[j])
            if st > DUE[j]:
                continue
            rc = dist[route[-1]][j] - pi[j-1]
            if rc < best_rc:
                best_rc = rc
                best_j = j
        if best_j == -1:
            break
        route.append(best_j)
        visited.add(best_j)
        arr = current_time + dist[route[-2]][best_j]
        current_time = max(arr, READY[best_j]) + SERVICE[best_j]
        current_load += DEMAND[best_j]

    route.append(0)
    total_cost = sum(dist[route[k]][route[k+1]] for k in range(len(route)-1))
    reduced_cost = total_cost - sum(pi[j-1] for j in route[1:-1])
    return route, total_cost, reduced_cost

# 数据读取
df_time = pd.read_excel("参考算例.xlsx", sheet_name="旅行时间矩阵", index_col=0)
df_nodes = pd.read_excel("参考算例.xlsx", sheet_name="节点属性信息", index_col=0)

n_customers = 10
customers = list(range(1, n_customers+1))
dist = df_time.loc[[0]+customers, [0]+customers].values

a = df_nodes.loc[customers, '开始服务时间下界'].values
b = df_nodes.loc[customers, '开始服务时间上界'].values
service = df_nodes.loc[customers, '服务时间'].values
capacity = df_nodes.loc[0, '车容量']
demand = df_nodes.loc[customers, '需求量'].values

READY = np.zeros(n_customers + 1)
DUE = np.zeros(n_customers + 1)
SERVICE = np.zeros(n_customers + 1)
DEMAND = np.zeros(n_customers + 1)
READY[1:] = a
DUE[1:] = b
SERVICE[1:] = service
DEMAND[1:] = demand

# 列生成主问题
nr_list, cr_list, route_list = generate_initial_columns(READY, DUE, SERVICE, DEMAND, dist, n_customers, capacity)

MainModel = gp.Model("VRPTW_ColumnGeneration")
vars_list = [MainModel.addVar(vtype=GRB.CONTINUOUS, lb=0, obj=cr_list[r]) for r in range(len(nr_list))]

MainModel.setObjective(gp.quicksum(vars_list[r] * cr_list[r] for r in range(len(nr_list))), GRB.MINIMIZE)

cons = []
for i in range(n_customers):
    c = MainModel.addConstr(gp.quicksum(nr_list[r][i] * vars_list[r] for r in range(len(nr_list))) == 1)
    cons.append(c)

obj_history = []
rc_history = []
iter_history = []
iter_num = 0

while True:
    iter_num += 1
    MainModel.optimize()

    obj_history.append(MainModel.objVal)
    iter_history.append(iter_num)

    pi = [cons[i].Pi for i in range(n_customers)]
    new_route, cost_route, rc = greedy_subproblem(pi, dist, READY, DUE, SERVICE, DEMAND, capacity, n_customers)
    rc_history.append(rc)

    if rc >= -1e-4:
        print("列生成收敛完成")
        break

    nr = np.zeros(n_customers, dtype=int)
    for j in new_route[1:-1]:
        nr[j-1] = 1

    new_var = MainModel.addVar(vtype=GRB.CONTINUOUS, lb=0, obj=cost_route)
    MainModel.update()

    for i in range(n_customers):
        if nr[i] == 1:
            MainModel.chgCoeff(cons[i], new_var, 1.0)

    nr_list.append(nr)
    route_list.append(new_route)
    vars_list.append(new_var)
    MainModel.update()

# 输出结果
print("\n" + "="*60)
print("最终车辆调度方案")
print("="*60)

vehicle_count = 0
all_used_routes = []

for r_idx in range(len(vars_list)):
    if vars_list[r_idx].X > 1e-4:
        vehicle_count += 1
        rt = route_list[r_idx]
        all_used_routes.append(rt)
        print(f"第 {vehicle_count} 辆车路线: {rt}")

print(f"\n车辆总数: {vehicle_count}")
print(f"最优总成本: {round(MainModel.objVal, 2)}")

# 绘图
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))

ax1.plot(iter_history, obj_history, linewidth=2, color='#2f7ed8')
ax1.set_title("目标函数收敛曲线", fontsize=12)
ax1.set_xlabel("迭代次数")
ax1.grid(alpha=0.3)

ax2.plot(iter_history, rc_history, linewidth=2, color='#e43955')
ax2.axhline(0, color='green', linestyle='--')
ax2.set_title("Reduced Cost 收敛曲线", fontsize=12)
ax2.set_xlabel("迭代次数")
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.show()