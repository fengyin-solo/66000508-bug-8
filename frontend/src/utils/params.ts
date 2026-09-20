import type { OptimizationParams, ParamIssue } from '../types'

/** 默认参数：留空或只填一半时按这组值补全（保证是一组可直接跑通的组合） */
export const DEFAULT_PARAMS: OptimizationParams = {
  algorithm: 'gradient_descent',
  functionId: 'rosenbrock',
  x0: -1.5,
  y0: 2.5,
  learningRate: 0.001,
  iterations: 100,
  momentum: 0.9,
  temperature: 100,
  coolingRate: 0.95
}

/** 各参数允许的取值范围（与后端 PARAM_LIMITS 一致，用于提交前本地校验） */
export const PARAM_LIMITS: Record<string, [number, number]> = {
  x0: [-10, 10],
  y0: [-10, 10],
  learningRate: [1e-7, 1],
  iterations: [10, 500],
  momentum: [0, 0.99],
  temperature: [1, 1000],
  coolingRate: [0.5, 0.999]
}

export const FIELD_LABELS: Record<string, string> = {
  x0: '初始X',
  y0: '初始Y',
  learningRate: '学习率',
  iterations: '迭代次数',
  momentum: '动量',
  temperature: '初始温度',
  coolingRate: '降温系数'
}

const LAST_GOOD_KEY = 'optviz:lastGoodParams'

export type ParamsForm = { [K in keyof OptimizationParams]: OptimizationParams[K] | null }

function isBlank(v: unknown): boolean {
  return v === null || v === undefined || (typeof v === 'number' && !Number.isFinite(v)) || v === ''
}

/** 空着或只填一半时按默认值补全；返回补全后的参数和被补全的字段名 */
export function normalizeParams(raw: Partial<ParamsForm>): { params: OptimizationParams; filled: string[] } {
  const params = { ...DEFAULT_PARAMS } as OptimizationParams
  const filled: string[] = []
  for (const key of Object.keys(DEFAULT_PARAMS) as (keyof OptimizationParams)[]) {
    const v = raw[key]
    if (isBlank(v)) {
      if (key in raw) filled.push(key)
    } else {
      ;(params as any)[key] = v
    }
  }
  params.iterations = Math.round(params.iterations)
  return { params, filled }
}

/** 提交前的本地范围校验；组合是否可行由后端仿真判定 */
export function validateParams(params: OptimizationParams): ParamIssue[] {
  const issues: ParamIssue[] = []
  for (const [field, [lo, hi]] of Object.entries(PARAM_LIMITS)) {
    const v = params[field as keyof OptimizationParams] as number
    if (v < lo || v > hi) {
      issues.push({ field, reason: `${FIELD_LABELS[field]} ${v} 超出允许范围，推荐范围为 [${lo}, ${hi}]。` })
    }
  }
  return issues
}

export function loadLastGoodParams(): OptimizationParams | null {
  try {
    const raw = localStorage.getItem(LAST_GOOD_KEY)
    if (!raw) return null
    return normalizeParams(JSON.parse(raw)).params
  } catch {
    return null
  }
}

export function saveLastGoodParams(params: OptimizationParams): void {
  try {
    localStorage.setItem(LAST_GOOD_KEY, JSON.stringify(params))
  } catch {
    /* localStorage 不可用时静默降级 */
  }
}
