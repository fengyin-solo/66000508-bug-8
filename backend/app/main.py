import math
import random
from typing import Optional

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI(title="Optimization Visualizer")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

FUNCTIONS = {
    "rosenbrock": lambda x, y: (1 - x) ** 2 + 100 * (y - x ** 2) ** 2,
    "himmelblau": lambda x, y: (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2,
    "rastrigin": lambda x, y: 20 + x ** 2 - 10 * math.cos(2 * math.pi * x) + y ** 2 - 10 * math.cos(2 * math.pi * y),
    "sphere": lambda x, y: x ** 2 + y ** 2,
    "beale": lambda x, y: (1.5 - x + x * y) ** 2 + (2.25 - x + x * y ** 2) ** 2 + (2.625 - x + x * y ** 3) ** 2,
    "booth": lambda x, y: (x + 2 * y - 7) ** 2 + (2 * x + y - 5) ** 2,
}

GRADIENTS = {
    "rosenbrock": lambda x, y: np.array([-2 * (1 - x) - 400 * x * (y - x ** 2), 200 * (y - x ** 2)]),
    "himmelblau": lambda x, y: np.array([4 * x * (x ** 2 + y - 11) + 2 * (x + y ** 2 - 7), 2 * (x ** 2 + y - 11) + 4 * y * (x + y ** 2 - 7)]),
    "rastrigin": lambda x, y: np.array([2 * x + 20 * math.pi * math.sin(2 * math.pi * x), 2 * y + 20 * math.pi * math.sin(2 * math.pi * y)]),
    "sphere": lambda x, y: np.array([2 * x, 2 * y]),
    "beale": lambda x, y: np.array([
        2 * (1.5 - x + x * y) * (-1 + y) + 2 * (2.25 - x + x * y ** 2) * (-1 + y ** 2) + 2 * (2.625 - x + x * y ** 3) * (-1 + y ** 3),
        2 * (1.5 - x + x * y) * x + 2 * (2.25 - x + x * y ** 2) * (2 * x * y) + 2 * (2.625 - x + x * y ** 3) * (3 * x * y ** 2)
    ]),
    "booth": lambda x, y: np.array([2 * (x + 2 * y - 7) + 4 * (2 * x + y - 5), 4 * (x + 2 * y - 7) + 2 * (2 * x + y - 5)]),
}

HESSIANS = {
    "rosenbrock": lambda x, y: np.array([
        [2 - 400 * y + 1200 * x ** 2, -400 * x],
        [-400 * x, 200]
    ]),
    "sphere": lambda x, y: np.array([[2, 0], [0, 2]]),
    "booth": lambda x, y: np.array([[10, 8], [8, 10]]),
}

FUNCTION_NAMES = {
    "rosenbrock": "Rosenbrock 香蕉函数",
    "himmelblau": "Himmelblau 函数",
    "rastrigin": "Rastrigin 函数",
    "sphere": "Sphere 球函数",
    "beale": "Beale 函数",
    "booth": "Booth 函数",
}

# 默认值：留空或只填一半时按这组值补全
DEFAULTS = {
    "algorithm": "gradient_descent",
    "functionId": "rosenbrock",
    "x0": -1.5,
    "y0": 2.5,
    "learningRate": 0.001,
    "iterations": 100,
    "momentum": 0.9,
    "temperature": 100.0,
    "coolingRate": 0.95,
}

# 各参数允许的取值范围（互相独立的部分）；组合是否可行由仿真判定
PARAM_LIMITS = {
    "x0": (-10.0, 10.0),
    "y0": (-10.0, 10.0),
    "learningRate": (1e-7, 1.0),
    "iterations": (10, 500),
    "momentum": (0.0, 0.99),
    "temperature": (1.0, 1000.0),
    "coolingRate": (0.5, 0.999),
}

FIELD_LABELS = {
    "x0": "初始X",
    "y0": "初始Y",
    "learningRate": "学习率",
    "iterations": "迭代次数",
    "momentum": "动量",
    "temperature": "初始温度",
    "coolingRate": "降温系数",
}

# 函数值或坐标超过该量级即判定为数值发散
DIVERGE_Z = 1e9
DIVERGE_XY = 1e6


class OptimizationRequest(BaseModel):
    # 全部允许为空：空着或只填一半时由服务端按默认值补全
    algorithm: Optional[str] = None
    functionId: Optional[str] = None
    x0: Optional[float] = None
    y0: Optional[float] = None
    learningRate: Optional[float] = None
    iterations: Optional[int] = None
    momentum: Optional[float] = None
    temperature: Optional[float] = None
    coolingRate: Optional[float] = None


def normalize_params(req: OptimizationRequest) -> dict:
    """把未填写（None）或非法（NaN/inf）的字段替换为默认值。"""
    raw = req.model_dump()
    params = {}
    for key, default in DEFAULTS.items():
        value = raw.get(key)
        if value is None:
            params[key] = default
        elif isinstance(value, float) and not math.isfinite(value):
            params[key] = default
        else:
            params[key] = value
    params["iterations"] = int(params["iterations"])
    return params


def validate_ranges(params: dict) -> list:
    """校验各参数独立取值范围，返回问题列表。"""
    issues = []
    for field, (lo, hi) in PARAM_LIMITS.items():
        value = params[field]
        if not (lo <= value <= hi):
            issues.append({
                "field": field,
                "reason": f"{FIELD_LABELS[field]} {value} 超出允许范围，推荐范围为 [{lo}, {hi}]。",
            })
    return issues


def simulate(params: dict, lr_override: Optional[float] = None):
    """按给定参数完整跑一遍优化。

    返回 (path, diverge_step, diverge_value)；未发散时 diverge_step 为 None。
    """
    fn = FUNCTIONS.get(params["functionId"], FUNCTIONS["rosenbrock"])
    grad_fn = GRADIENTS.get(params["functionId"])

    def g_fn(x, y):
        if grad_fn:
            return grad_fn(x, y)
        return np.array([
            (fn(x + 1e-5, y) - fn(x - 1e-5, y)) / 2e-5,
            (fn(x, y + 1e-5) - fn(x, y - 1e-5)) / 2e-5
        ])

    lr = params["learningRate"] if lr_override is None else lr_override
    iterations = params["iterations"]
    x, y = params["x0"], params["y0"]

    def safe_z(x, y):
        try:
            z = float(fn(x, y))
        except (OverflowError, FloatingPointError):
            return math.inf
        return z

    def diverged(step, x, y):
        z = safe_z(x, y)
        if not math.isfinite(z) or abs(z) > DIVERGE_Z or abs(x) > DIVERGE_XY or abs(y) > DIVERGE_XY:
            return step, z
        return None

    path = [{"step": 0, "x": x, "y": y, "z": safe_z(x, y)}]
    with np.errstate(all="ignore"):
        if params["algorithm"] == "gradient_descent":
            vx, vy = 0.0, 0.0
            for i in range(iterations):
                try:
                    g = g_fn(x, y)
                    vx = params["momentum"] * vx - lr * float(g[0])
                    vy = params["momentum"] * vy - lr * float(g[1])
                except (OverflowError, FloatingPointError):
                    return path, i + 1, math.inf
                x += vx
                y += vy
                path.append({"step": i + 1, "x": x, "y": y, "z": safe_z(x, y)})
                d = diverged(i + 1, x, y)
                if d:
                    return path, d[0], d[1]

        elif params["algorithm"] == "newton":
            hess_fn = HESSIANS.get(params["functionId"])
            for i in range(iterations):
                try:
                    g = g_fn(x, y)
                    if hess_fn is None:
                        dx = -g * lr
                    else:
                        try:
                            dx = np.linalg.solve(hess_fn(x, y), -g)
                        except np.linalg.LinAlgError:
                            dx = -g * lr
                    dx = np.nan_to_num(dx, nan=0.0, posinf=math.inf, neginf=-math.inf)
                    x += float(dx[0])
                    y += float(dx[1])
                except (OverflowError, FloatingPointError):
                    return path, i + 1, math.inf
                path.append({"step": i + 1, "x": x, "y": y, "z": safe_z(x, y)})
                d = diverged(i + 1, x, y)
                if d:
                    return path, d[0], d[1]

        elif params["algorithm"] == "conjugate_gradient":
            try:
                g = g_fn(x, y)
            except (OverflowError, FloatingPointError):
                return path, 1, math.inf
            d = -g.copy()
            for i in range(iterations):
                try:
                    x_new = x + lr * float(d[0])
                    y_new = y + lr * float(d[1])
                    g_new = g_fn(x_new, y_new)
                    beta = max(0.0, float(g_new @ g_new) / (float(g @ g) + 1e-10))
                    d = -g_new + beta * d
                except (OverflowError, FloatingPointError):
                    return path, i + 1, math.inf
                x, y, g = x_new, y_new, g_new
                path.append({"step": i + 1, "x": x, "y": y, "z": safe_z(x, y)})
                dv = diverged(i + 1, x, y)
                if dv:
                    return path, dv[0], dv[1]

        elif params["algorithm"] == "simulated_annealing":
            T = params["temperature"]
            best_x, best_y = x, y
            best_z = safe_z(x, y)
            for i in range(iterations):
                try:
                    nx = x + random.gauss(0, T / params["temperature"] * 2)
                    ny = y + random.gauss(0, T / params["temperature"] * 2)
                    nz = safe_z(nx, ny)
                    delta = nz - safe_z(x, y)
                    if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-5)):
                        x, y = nx, ny
                        if safe_z(x, y) < best_z:
                            best_x, best_y = x, y
                            best_z = safe_z(x, y)
                except (OverflowError, FloatingPointError):
                    return path, i + 1, math.inf
                T *= params["coolingRate"]
                path.append({"step": i + 1, "x": x, "y": y, "z": safe_z(x, y)})
                dv = diverged(i + 1, x, y)
                if dv:
                    return path, dv[0], dv[1]

    return path, None, None


def find_lr_upper_bound(params: dict) -> Optional[float]:
    """在其余参数不变的前提下，二分搜索不发散的最大学习率。"""
    lo, hi = PARAM_LIMITS["learningRate"]

    def stable(lr):
        _, step, _ = simulate(params, lr_override=lr)
        return step is None

    if stable(hi):
        return None  # 上限即允许的最大值，不算组合问题
    if not stable(lo):
        return None  # 最小学习率都发散，问题不在学习率（通常是起点），不给学习率建议
    for _ in range(40):
        mid = math.sqrt(lo * hi)  # 几何二分，学习率跨数量级更均匀
        if stable(mid):
            lo = mid
        else:
            hi = mid
    return lo


def fmt_num(v: float) -> str:
    if v == 0:
        return "0"
    if abs(v) >= 1e4 or abs(v) < 1e-3:
        return f"{v:.3e}"
    return f"{v:g}"


def error_response(code: str, message: str, issues: list, normalized: dict,
                   upper_bounds: Optional[dict] = None, recommended: Optional[dict] = None) -> JSONResponse:
    detail = {
        "code": code,
        "message": message,
        "issues": issues,
        "limits": {k: [v[0], v[1]] for k, v in PARAM_LIMITS.items()},
        "normalized": normalized,
    }
    if upper_bounds:
        detail["upperBounds"] = upper_bounds
    if recommended:
        detail["recommendedParams"] = recommended
    return JSONResponse(status_code=400, content={"detail": detail})


@app.post("/api/optimize")
def optimize(req: OptimizationRequest):
    params = normalize_params(req)

    if params["functionId"] not in FUNCTIONS:
        return error_response(
            "INVALID_PARAMS",
            f"未知的测试函数 {params['functionId']}，可选：{', '.join(FUNCTIONS)}。",
            [{"field": "functionId", "reason": "不支持的测试函数。"}], params)
    if params["algorithm"] not in ("gradient_descent", "newton", "conjugate_gradient", "simulated_annealing"):
        return error_response(
            "INVALID_PARAMS",
            f"未知的算法 {params['algorithm']}。",
            [{"field": "algorithm", "reason": "不支持的优化算法。"}], params)

    issues = validate_ranges(params)
    if issues:
        message = "；".join(i["reason"] for i in issues)
        return error_response("INVALID_PARAMS", message, issues, params)

    path, div_step, div_value = simulate(params)

    if div_step is not None:
        fname = FUNCTION_NAMES.get(params["functionId"], params["functionId"])
        value_text = "数值溢出（inf）" if not math.isfinite(div_value) else f"函数值 {fmt_num(div_value)}"
        combo_parts = [f"函数 {fname}", f"起点({params['x0']}, {params['y0']})"]
        if params["algorithm"] == "gradient_descent":
            combo_parts.append(f"动量 {params['momentum']}")
        if params["algorithm"] == "simulated_annealing":
            combo_parts.append(f"初始温度 {params['temperature']}")
            combo_parts.append(f"降温系数 {params['coolingRate']}")
        combo_parts.append(f"{params['iterations']} 次迭代")
        combo_desc = "、".join(combo_parts)
        if params["algorithm"] in ("gradient_descent", "conjugate_gradient"):
            ub = find_lr_upper_bound(params)
            if ub is not None:
                recommended_lr = 10 ** math.floor(math.log10(ub))
                message = (f"学习率 {fmt_num(params['learningRate'])} 过大：在{combo_desc}的组合下，"
                           f"第 {div_step} 步数值发散（{value_text}）。"
                           f"学习率、动量、起点、迭代次数互相牵制，该组合学习率上限约为 {fmt_num(round(ub, 6))}，"
                           f"建议学习率 ≤ {fmt_num(recommended_lr)}。")
                return error_response(
                    "INVALID_COMBO", message,
                    [{"field": "learningRate", "reason": message}], params,
                    upper_bounds={"learningRate": round(ub, 6)},
                    recommended={"learningRate": recommended_lr})
        message = (f"当前参数组合在第 {div_step} 步数值发散（{value_text}）：{combo_desc}。"
                   f"学习率、动量、起点、迭代次数互相牵制，建议调整起点或改用梯度下降并降低学习率。")
        return error_response(
            "INVALID_COMBO", message,
            [{"field": "learningRate", "reason": message}], params)

    final = path[-1]
    return {
        "params": params,
        "path": path,
        "finalPoint": [final["x"], final["y"]],
        "finalValue": final["z"],
        "iterations": len(path) - 1,
        "converged": abs(final["z"]) < 1e-3 or len(path) - 1 >= params["iterations"]
    }
