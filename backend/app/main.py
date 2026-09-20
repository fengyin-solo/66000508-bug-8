import math
import warnings
from typing import Any, Dict, Optional

import numpy as np
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

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
    "himmelblau": lambda x, y: np.array([
        4 * x * (x ** 2 + y - 11) + 2 * (x + y ** 2 - 7),
        2 * (x ** 2 + y - 11) + 4 * y * (x + y ** 2 - 7),
    ]),
    "rastrigin": lambda x, y: np.array([
        2 * x + 20 * math.pi * math.sin(2 * math.pi * x),
        2 * y + 20 * math.pi * math.sin(2 * math.pi * y),
    ]),
    "sphere": lambda x, y: np.array([2 * x, 2 * y]),
    "beale": lambda x, y: np.array([
        2 * (1.5 - x + x * y) * (-1 + y)
        + 2 * (2.25 - x + x * y ** 2) * (-1 + y ** 2)
        + 2 * (2.625 - x + x * y ** 3) * (-1 + y ** 3),
        2 * (1.5 - x + x * y) * x
        + 2 * (2.25 - x + x * y ** 2) * (2 * x * y)
        + 2 * (2.625 - x + x * y ** 3) * (3 * x * y ** 2),
    ]),
    "booth": lambda x, y: np.array([
        2 * (x + 2 * y - 7) + 4 * (2 * x + y - 5),
        4 * (x + 2 * y - 7) + 2 * (2 * x + y - 5),
    ]),
}

HESSIANS = {
    "rosenbrock": lambda x, y: np.array([
        [2 - 400 * y + 1200 * x ** 2, -400 * x],
        [-400 * x, 200],
    ]),
    "sphere": lambda x, y: np.array([[2.0, 0.0], [0.0, 2.0]]),
    "booth": lambda x, y: np.array([[10.0, 8.0], [8.0, 10.0]]),
}

DEFAULTS: Dict[str, Any] = {
    "algorithm": "gradient_descent",
    "functionId": "rosenbrock",
    "x0": -1.5,
    "y0": 2.5,
    # 0.01 配合较大动量会在 Rosenbrock 的强梯度起点发散；0.001、0 动量可稳定出图。
    "learningRate": 0.001,
    "iterations": 100,
    "momentum": 0.0,
    "temperature": 100.0,
    "coolingRate": 0.95,
}

MIN_LEARNING_RATE = 0.00001
MAX_LEARNING_RATE = 0.01
MIN_ITERATIONS = 10
MAX_ITERATIONS = 1000
VALUE_LIMIT = 1e12
COORDINATE_LIMIT = 1e8


class OptimizationRequest(BaseModel):
    algorithm: Optional[str] = None
    functionId: Optional[str] = None
    x0: Optional[float] = None
    y0: Optional[float] = None
    learningRate: Optional[float] = None
    iterations: Optional[int] = None
    momentum: Optional[float] = None
    temperature: Optional[float] = None
    coolingRate: Optional[float] = None

    @field_validator("x0", "y0", "learningRate", "momentum", "temperature", "coolingRate")
    @classmethod
    def reject_non_finite_float(cls, value: Optional[float]) -> Optional[float]:
        if value is not None and not math.isfinite(value):
            raise ValueError("不能为 NaN 或无穷大")
        return value

    @field_validator("iterations")
    @classmethod
    def reject_non_finite_int(cls, value: Optional[int]) -> Optional[int]:
        if value is not None and not math.isfinite(value):
            raise ValueError("迭代次数必须是有效整数")
        return value


class OptimizationError(Exception):
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


@app.exception_handler(OptimizationError)
async def optimization_error_handler(request: Request, exc: OptimizationError):
    detail = {"message": exc.message}
    detail.update(exc.details)
    return JSONResponse(status_code=400, content={"detail": detail})


def normalize_request(req: OptimizationRequest) -> Dict[str, Any]:
    supplied = req.model_dump(exclude_unset=True)
    params = DEFAULTS.copy()
    for key, value in supplied.items():
        if value is not None:
            params[key] = value

    if params["algorithm"] not in {"gradient_descent", "newton", "conjugate_gradient", "simulated_annealing"}:
        raise OptimizationError("不支持的优化算法")
    if params["functionId"] not in FUNCTIONS:
        raise OptimizationError("不支持的测试函数")

    numeric_ranges = {
        "x0": (-10.0, 10.0),
        "y0": (-10.0, 10.0),
        "learningRate": (MIN_LEARNING_RATE, MAX_LEARNING_RATE),
        "momentum": (0.0, 0.99),
        "temperature": (1.0, 10000.0),
        "coolingRate": (0.5, 0.999),
    }
    names = {
        "x0": "初始 X",
        "y0": "初始 Y",
        "learningRate": "学习率",
        "momentum": "动量",
        "temperature": "温度",
        "coolingRate": "冷却率",
    }
    for key, (low, high) in numeric_ranges.items():
        value = float(params[key])
        if not low <= value <= high:
            raise OptimizationError(f"{names[key]}必须在 {low:g} 到 {high:g} 之间，当前为 {value:g}")

    iterations = int(params["iterations"])
    if not MIN_ITERATIONS <= iterations <= MAX_ITERATIONS:
        raise OptimizationError(f"迭代次数必须在 {MIN_ITERATIONS} 到 {MAX_ITERATIONS} 之间，当前为 {iterations}")
    params["iterations"] = iterations

    return params


def get_functions(function_id: str):
    fn = FUNCTIONS[function_id]
    analytic_grad = GRADIENTS.get(function_id)

    def numeric_gradient(x: float, y: float) -> np.ndarray:
        epsilon = 1e-6
        return np.array([
            (fn(x + epsilon, y) - fn(x - epsilon, y)) / (2 * epsilon),
            (fn(x, y + epsilon) - fn(x, y - epsilon)) / (2 * epsilon),
        ], dtype=float)

    return fn, analytic_grad or numeric_gradient


def failure_payload(
    params: Dict[str, Any],
    step: int,
    kind: str,
    value: Optional[float] = None,
) -> Dict[str, Any]:
    recommendations = None if params.get("__probe__", False) else build_recommendations(params, step)
    if kind == "non_finite":
        if params["algorithm"] == "gradient_descent":
            reason = (
                f"第 {step} 次迭代得到 NaN/无穷大。学习率 {params['learningRate']:g} 与动量 "
                f"{params['momentum']:g} 会在强梯度处把更新步累积得过大。"
            )
        elif params["algorithm"] == "conjugate_gradient":
            reason = (
                f"第 {step} 次迭代得到 NaN/无穷大。固定步长 {params['learningRate']:g} 已超过该组合的稳定上限。"
            )
        else:
            reason = f"第 {step} 次迭代得到 NaN/无穷大，当前出发点或参数组合不稳定。"
    else:
        reason = (
            f"第 {step} 次迭代后函数值约为 {value:.2e}，超过 {VALUE_LIMIT:.0e} 的安全上限，"
            "继续计算会变成无穷大。"
        )

    return {
        "valid": False,
        "params": params,
        "failedStep": step,
        "reason": reason,
        "recommendations": recommendations,
    }


def evaluate(fn, x: float, y: float, step: int, params: Dict[str, Any]) -> float:
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        z = float(fn(x, y))
    if not math.isfinite(z):
        raise OptimizationError("数值变为 NaN 或无穷大", failure_payload(params, step, "non_finite"))
    if abs(z) > VALUE_LIMIT or abs(x) > COORDINATE_LIMIT or abs(y) > COORDINATE_LIMIT:
        raise OptimizationError(
            "数值超过安全上限",
            failure_payload(params, step, "overflow", z if math.isfinite(z) else VALUE_LIMIT),
        )
    return z


def gradient_at(grad_fn, x: float, y: float, step: int, params: Dict[str, Any]) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        gradient = np.asarray(grad_fn(x, y), dtype=float)
    if not np.all(np.isfinite(gradient)):
        raise OptimizationError("梯度变为 NaN 或无穷大", failure_payload(params, step, "non_finite"))
    return gradient


def build_path(params: Dict[str, Any]) -> list:
    fn, grad_fn = get_functions(params["functionId"])
    x, y = float(params["x0"]), float(params["y0"])
    iterations = int(params["iterations"])
    path = [{"step": 0, "x": x, "y": y, "z": evaluate(fn, x, y, 0, params)}]

    def add_point(next_x: float, next_y: float, step: int):
        next_x = float(next_x)
        next_y = float(next_y)
        if not math.isfinite(next_x) or not math.isfinite(next_y):
            raise OptimizationError("坐标变为 NaN 或无穷大", failure_payload(params, step, "non_finite"))
        z = evaluate(fn, next_x, next_y, step, params)
        point = {"step": step, "x": next_x, "y": next_y, "z": z}
        path.append(point)
        return next_x, next_y, z

    if params["algorithm"] == "gradient_descent":
        vx = vy = 0.0
        for step in range(1, iterations + 1):
            gradient = gradient_at(grad_fn, x, y, step, params)
            vx = params["momentum"] * vx - params["learningRate"] * gradient[0]
            vy = params["momentum"] * vy - params["learningRate"] * gradient[1]
            x, y, _ = add_point(x + vx, y + vy, step)

    elif params["algorithm"] == "newton":
        hess_fn = HESSIANS.get(params["functionId"])
        if hess_fn is None:
            for step in range(1, iterations + 1):
                gradient = gradient_at(grad_fn, x, y, step, params)
                x, y, _ = add_point(
                    x - params["learningRate"] * gradient[0],
                    y - params["learningRate"] * gradient[1],
                    step,
                )
        else:
            for step in range(1, iterations + 1):
                gradient = gradient_at(grad_fn, x, y, step, params)
                hessian = np.asarray(hess_fn(x, y), dtype=float)
                try:
                    if not np.all(np.isfinite(hessian)):
                        raise np.linalg.LinAlgError()
                    delta = np.linalg.solve(hessian, -gradient)
                    if not np.all(np.isfinite(delta)):
                        raise np.linalg.LinAlgError()
                except np.linalg.LinAlgError:
                    delta = -params["learningRate"] * gradient
                x, y, _ = add_point(x + delta[0], y + delta[1], step)

    elif params["algorithm"] == "conjugate_gradient":
        gradient = gradient_at(grad_fn, x, y, 1, params)
        direction = -gradient.copy()
        for step in range(1, iterations + 1):
            if np.linalg.norm(gradient) < 1e-10:
                break
            next_x = x + params["learningRate"] * direction[0]
            next_y = y + params["learningRate"] * direction[1]
            x, y, _ = add_point(next_x, next_y, step)
            next_gradient = gradient_at(grad_fn, x, y, step, params)
            denominator = float(gradient @ gradient)
            beta = max(0.0, float(next_gradient @ next_gradient) / denominator) if denominator > 1e-20 else 0.0
            direction = -next_gradient + beta * direction
            gradient = next_gradient

    elif params["algorithm"] == "simulated_annealing":
        rng = np.random.default_rng(42)
        temperature = float(params["temperature"])
        initial_temperature = temperature
        for step in range(1, iterations + 1):
            sigma = 2.0 * max(temperature, 1e-8) / initial_temperature
            next_x = x + rng.normal(0.0, sigma)
            next_y = y + rng.normal(0.0, sigma)
            next_z = evaluate(fn, next_x, next_y, step, params)
            current_z = path[-1]["z"]
            delta = next_z - current_z
            if delta <= 0:
                accept = True
            elif delta / max(temperature, 1e-8) > 1000:
                accept = False
            else:
                accept = rng.random() < math.exp(-delta / max(temperature, 1e-8))
            if accept:
                x, y = next_x, next_y
                path.append({"step": step, "x": x, "y": y, "z": next_z})
            else:
                path.append({"step": step, "x": x, "y": y, "z": current_z})
            temperature *= params["coolingRate"]

    return path


def safe_run(params: Dict[str, Any]) -> bool:
    trial = dict(params)
    trial["__probe__"] = True
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            build_path(trial)
        return True
    except OptimizationError:
        return False


def search_contiguous_upper(params: Dict[str, Any], key: str, low: float, high: float) -> Optional[float]:
    trial = dict(params)

    def safe_at(value: float) -> bool:
        trial[key] = value
        return safe_run(trial)

    if not safe_at(low):
        return None

    last_safe = low
    candidates = np.linspace(low, high, 48, dtype=float)
    first_failure: Optional[float] = None
    for candidate in candidates[1:]:
        if safe_at(float(candidate)):
            last_safe = float(candidate)
        else:
            first_failure = float(candidate)
            break
    else:
        return last_safe

    # 在最后一个连续安全点和首个失败点之间再做一轮细扫。
    refined = np.linspace(last_safe, first_failure, 14, dtype=float)
    for candidate in refined[1:-1]:
        value = float(candidate)
        if safe_at(value):
            last_safe = value
        else:
            break
    return last_safe


def build_recommendations(params: Dict[str, Any], failed_step: int) -> Optional[Dict[str, Any]]:
    current_safe_iterations = max(0, failed_step - 1)
    if params["algorithm"] == "gradient_descent":
        # 推荐上限必须按允许的最大迭代次数验证，避免只在当前较短迭代中“暂时安全”。
        probe_params = dict(params)
        probe_params["iterations"] = MAX_ITERATIONS
        momentum = probe_params["momentum"]
        max_lr = search_contiguous_upper(probe_params, "learningRate", MIN_LEARNING_RATE, MAX_LEARNING_RATE)
        if max_lr is None:
            max_momentum_floor = search_contiguous_upper(probe_params, "momentum", 0.0, 0.99)
            momentum = max_momentum_floor if max_momentum_floor is not None else 0.0
            probe_params["momentum"] = momentum
            max_lr = search_contiguous_upper(probe_params, "learningRate", MIN_LEARNING_RATE, MAX_LEARNING_RATE)

        if max_lr is None:
            return None

        recommended_lr = max(MIN_LEARNING_RATE, max_lr * 0.8)
        # 控件有 5 位小数精度，向上舍入后仍保留安全余量。
        max_lr_ceiling = math.floor(max_lr * 0.9 * 100000) / 100000
        momentum_params = dict(probe_params)
        momentum_params["learningRate"] = recommended_lr
        max_momentum = search_contiguous_upper(momentum_params, "momentum", 0.0, 0.99)
        momentum_ceiling = math.floor((max_momentum or 0.0) * 100) / 100
        recommended_momentum = min(momentum, momentum_ceiling)
        return {
            "learningRate": recommended_lr,
            "maxLearningRate": max(MIN_LEARNING_RATE, max_lr_ceiling),
            "momentum": recommended_momentum,
            "maxMomentum": momentum_ceiling,
            "iterations": MAX_ITERATIONS,
            "maxIterations": MAX_ITERATIONS,
            "currentSafeIterations": current_safe_iterations,
        }

    if params["algorithm"] == "conjugate_gradient":
        probe_params = dict(params)
        probe_params["iterations"] = MAX_ITERATIONS
        max_lr = search_contiguous_upper(probe_params, "learningRate", MIN_LEARNING_RATE, MAX_LEARNING_RATE)
        if max_lr is None:
            return None
        recommended_lr = max(MIN_LEARNING_RATE, max_lr * 0.8)
        max_lr_ceiling = math.floor(max_lr * 0.9 * 100000) / 100000
        return {
            "learningRate": recommended_lr,
            "maxLearningRate": max(MIN_LEARNING_RATE, max_lr_ceiling),
            "iterations": MAX_ITERATIONS,
            "maxIterations": MAX_ITERATIONS,
            "currentSafeIterations": current_safe_iterations,
        }

    return {
        "iterations": max(MIN_ITERATIONS, current_safe_iterations)
        if current_safe_iterations >= MIN_ITERATIONS
        else None,
        "maxIterations": MAX_ITERATIONS,
        "currentSafeIterations": current_safe_iterations,
    }


def validate_params(params: Dict[str, Any]) -> Dict[str, Any]:
    try:
        build_path(params)
    except OptimizationError as exc:
        return exc.details
    return {"valid": True, "params": params, "failedStep": None, "reason": None, "recommendations": None}


@app.post("/api/validate")
def validate(req: OptimizationRequest):
    try:
        params = normalize_request(req)
    except OptimizationError as exc:
        return {
            "valid": False,
            "params": DEFAULTS,
            "failedStep": None,
            "reason": exc.message,
            "recommendations": None,
        }
    return validate_params(params)


@app.post("/api/optimize")
def optimize(req: OptimizationRequest):
    params = normalize_request(req)
    path = build_path(params)
    final = path[-1]
    _, grad_fn = get_functions(params["functionId"])
    final_gradient = grad_fn(final["x"], final["y"])
    converged = abs(final["z"]) < 1e-3 or (
        np.all(np.isfinite(final_gradient)) and np.linalg.norm(final_gradient) < 1e-6
    )

    return {
        "params": params,
        "path": path,
        "finalPoint": [final["x"], final["y"]],
        "finalValue": final["z"],
        "iterations": len(path) - 1,
        "converged": bool(converged),
    }
