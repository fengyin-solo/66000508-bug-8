<template>
  <div class="control-card">
    <el-alert
      v-if="validationInvalid"
      class="param-alert"
      type="warning"
      show-icon
      :closable="false"
      title="当前参数组合不可用"
    >
      <template #default>
        <p>{{ validation?.reason || '该学习率、动量和迭代次数组合可能产生 NaN 或无穷大。' }}</p>
        <ul v-if="recommendationLines.length" class="recommend-list">
          <li v-for="line in recommendationLines" :key="line">{{ line }}</li>
        </ul>
        <div class="alert-actions">
          <el-button size="small" type="primary" :disabled="!hasRecommendation" @click="applyRecommendation">
            应用推荐参数
          </el-button>
          <el-button size="small" :disabled="!lastWorkingParams" @click="restoreLastWorking">
            回到上次可用参数
          </el-button>
        </div>
      </template>
    </el-alert>

    <el-alert
      v-else-if="store.error"
      class="param-alert"
      type="error"
      show-icon
      :closable="false"
      :title="store.error.message"
    >
      <template #default v-if="store.error.recommendations">
        <div class="alert-actions">
          <el-button size="small" type="primary" @click="applyStoreRecommendation">应用推荐参数</el-button>
          <el-button size="small" :disabled="!lastWorkingParams" @click="restoreLastWorking">回到上次可用参数</el-button>
        </div>
      </template>
    </el-alert>

    <el-form :model="form" inline>
      <el-form-item label="测试函数">
        <el-select v-model="form.functionId" style="width:180px">
          <el-option v-for="f in TEST_FUNCTIONS" :key="f.id" :label="f.name" :value="f.id" />
        </el-select>
      </el-form-item>
      <el-form-item label="算法">
        <el-select v-model="form.algorithm" style="width:150px">
          <el-option v-for="a in ALGORITHMS" :key="a.id" :label="a.name" :value="a.id" />
        </el-select>
      </el-form-item>
      <el-form-item label="初始X">
        <el-input-number v-model="form.x0" :min="-10" :max="10" :step="0.5" size="small" controls-position="right" />
      </el-form-item>
      <el-form-item label="初始Y">
        <el-input-number v-model="form.y0" :min="-10" :max="10" :step="0.5" size="small" controls-position="right" />
      </el-form-item>
      <el-form-item v-if="form.algorithm !== 'simulated_annealing'" label="学习率">
        <el-input-number
          v-model="form.learningRate"
          :min="0.00001"
          :max="0.01"
          :step="0.0001"
          :precision="5"
          size="small"
          controls-position="right"
        />
        <div class="field-hint">
          全局 0.00001–0.01
          <template v-if="learningRateHint">；{{ learningRateHint }}</template>
          <template v-else>；留空时按 {{ DEFAULT_PARAMS.learningRate }}</template>
        </div>
      </el-form-item>
      <el-form-item label="迭代">
        <el-input-number
          v-model="form.iterations"
          :min="10"
          :max="1000"
          :step="10"
          size="small"
          controls-position="right"
        />
        <div class="field-hint">范围 10–1000；留空时按 {{ DEFAULT_PARAMS.iterations }}</div>
      </el-form-item>
      <el-form-item v-if="form.algorithm==='gradient_descent'" label="动量">
        <el-input-number
          v-model="form.momentum"
          :min="0"
          :max="0.99"
          :step="0.1"
          :precision="2"
          size="small"
          controls-position="right"
        />
        <div class="field-hint">
          全局 0–0.99
          <template v-if="momentumHint">；{{ momentumHint }}</template>
          <template v-else>；留空时按 {{ DEFAULT_PARAMS.momentum }}</template>
        </div>
      </el-form-item>
      <el-form-item v-if="form.algorithm==='simulated_annealing'" label="温度">
        <el-input-number
          v-model="form.temperature"
          :min="1"
          :max="10000"
          :step="10"
          size="small"
          controls-position="right"
        />
        <div class="field-hint">范围 1–10000；留空时按 {{ DEFAULT_PARAMS.temperature }}</div>
      </el-form-item>
      <el-form-item class="action-item">
        <el-button type="primary" @click="run" :loading="store.loading" :disabled="startDisabled">
          {{ validationLoading ? '校验中…' : '🚀 开始优化' }}
        </el-button>
        <el-button @click="resetDefaults">恢复默认</el-button>
        <el-button :disabled="!lastWorkingParams" @click="restoreLastWorking">
          上次可用{{ lastWorkingSummary }}
        </el-button>
      </el-form-item>
    </el-form>

    <div v-if="validationMessage" class="validation-status" :class="validationLoading ? 'is-loading' : 'is-warning'">
      {{ validationMessage }}
    </div>

    <div class="animation-bar" v-if="store.result">
      <div class="anim-controls">
        <el-button size="small" @click="store.playAnimation" :disabled="store.isPlaying">▶ 播放</el-button>
        <el-button size="small" @click="store.pauseAnimation" :disabled="!store.isPlaying">⏸ 暂停</el-button>
        <el-button size="small" @click="store.resetAnimation">⏹ 重置</el-button>
      </div>
      <el-slider v-model="animStep" :min="0" :max="maxStep" @input="onSlider" style="flex:1;margin:0 20px" />
      <span class="step-text">步 {{ animStep }}/{{ maxStep }}</span>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, onUnmounted, reactive, ref, watch } from 'vue'
import axios from 'axios'
import { useOptimizationStore } from '../store/optimization'
import { TEST_FUNCTIONS, ALGORITHMS } from '../types'
import type { OptimizationParams, ParameterRecommendations, ValidationResponse } from '../types'

type FormValues = {
  algorithm: string
  functionId: string
  x0: number | null
  y0: number | null
  learningRate: number | null
  iterations: number | null
  momentum: number | null
  temperature: number | null
  coolingRate: number | null
}

const DEFAULT_PARAMS: OptimizationParams = {
  algorithm: 'gradient_descent',
  functionId: 'rosenbrock',
  x0: -1.5,
  y0: 2.5,
  learningRate: 0.001,
  iterations: 100,
  momentum: 0,
  temperature: 100,
  coolingRate: 0.95,
}

const STORAGE_KEY = 'optimization-visualizer:last-working-params'
const STORAGE_VERSION = 1
const store = useOptimizationStore()

function createForm(source: OptimizationParams): FormValues {
  return {
    algorithm: source.algorithm,
    functionId: source.functionId,
    x0: source.x0,
    y0: source.y0,
    learningRate: source.learningRate,
    iterations: source.iterations,
    momentum: source.momentum,
    temperature: source.temperature,
    coolingRate: source.coolingRate,
  }
}

function loadLastWorking(): OptimizationParams | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return null
    const parsed = JSON.parse(raw)
    if (
      typeof parsed !== 'object' ||
      parsed === null ||
      parsed.version !== STORAGE_VERSION ||
      !parsed.params ||
      !parsed.params.algorithm ||
      !parsed.params.functionId
    ) return null
    return { ...DEFAULT_PARAMS, ...parsed.params }
  } catch {
    return null
  }
}

const lastWorkingParams = ref<OptimizationParams | null>(loadLastWorking())
const form = reactive<FormValues>(createForm(lastWorkingParams.value || DEFAULT_PARAMS))
const validation = ref<ValidationResponse | null>(null)
const validationLoading = ref(false)
const validationMessage = ref('')

const animStep = ref(0)
const maxStep = computed(() => Math.max(0, (store.result?.path.length || 1) - 1))
const validationInvalid = computed(() => validation.value?.valid === false)
const recommendations = computed<ParameterRecommendations | null>(() => validation.value?.recommendations || null)
const hasRecommendation = computed(() =>
  recommendations.value?.learningRate !== undefined || recommendations.value?.iterations !== undefined
)
const startDisabled = computed(() => validationLoading.value || validationInvalid.value)

watch(() => store.animationStep, (v) => { animStep.value = v })

function numberOr(value: number | null | undefined, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback
}

function normalizedParams(): OptimizationParams {
  return {
    algorithm: form.algorithm || DEFAULT_PARAMS.algorithm,
    functionId: form.functionId || DEFAULT_PARAMS.functionId,
    x0: numberOr(form.x0, DEFAULT_PARAMS.x0),
    y0: numberOr(form.y0, DEFAULT_PARAMS.y0),
    learningRate: numberOr(form.learningRate, DEFAULT_PARAMS.learningRate),
    iterations: Math.round(numberOr(form.iterations, DEFAULT_PARAMS.iterations)),
    momentum: numberOr(form.momentum, DEFAULT_PARAMS.momentum),
    temperature: numberOr(form.temperature, DEFAULT_PARAMS.temperature),
    coolingRate: numberOr(form.coolingRate, DEFAULT_PARAMS.coolingRate),
  }
}

function formatNumber(value?: number | null): string {
  if (value === undefined || value === null) return ''
  return String(Number(value.toPrecision(6)))
}

const learningRateHint = computed(() => {
  const rec = recommendations.value
  if (!rec?.maxLearningRate) return ''
  return `配 ${formatNumber(rec.momentum)} 动量时建议 ≤ ${formatNumber(rec.maxLearningRate)}，推荐 ${formatNumber(rec.learningRate)}`
})

const momentumHint = computed(() => {
  const rec = recommendations.value
  if (rec?.maxMomentum === undefined) return ''
  return `配 ${formatNumber(rec.learningRate)} 学习率时建议 ≤ ${formatNumber(rec.maxMomentum)}，推荐 ${formatNumber(rec.momentum)}`
})

const recommendationLines = computed(() => {
  const rec = recommendations.value
  if (!rec) return []
  const lines: string[] = []
  if (rec.maxLearningRate !== undefined) {
    lines.push(`学习率推荐 ${formatNumber(rec.learningRate)}；与 ${formatNumber(rec.momentum)} 动量搭配时，学习率上限约 ${formatNumber(rec.maxLearningRate)}。`)
  }
  if (rec.maxMomentum !== undefined) {
    lines.push(`动量推荐 ${formatNumber(rec.momentum)}；与 ${formatNumber(rec.learningRate)} 学习率搭配时，动量上限约 ${formatNumber(rec.maxMomentum)}。`)
  }
  if (rec.iterations !== undefined && rec.iterations !== null) {
    lines.push(`推荐迭代次数：${rec.iterations} 次；硬上限 ${rec.maxIterations} 次。`)
  } else if (rec.currentSafeIterations !== undefined) {
    lines.push(`当前组合只能稳定计算到第 ${rec.currentSafeIterations} 次迭代，请先降低学习率或动量。`)
  }
  return lines
})

const lastWorkingSummary = computed(() => {
  const p = lastWorkingParams.value
  if (!p) return ''
  const lr = p.algorithm === 'simulated_annealing' ? '' : `，lr=${formatNumber(p.learningRate)}`
  return `（${p.functionId}${lr}，${p.iterations}次）`
})

let debounceTimer: ReturnType<typeof setTimeout> | null = null
let activeController: AbortController | null = null

function cancelValidation() {
  if (debounceTimer) {
    clearTimeout(debounceTimer)
    debounceTimer = null
  }
  activeController?.abort()
  activeController = null
}

async function validateCurrent(immediate = false): Promise<boolean> {
  cancelValidation()
  if (immediate) return requestValidation()

  validationLoading.value = true
  validationMessage.value = '正在校验参数组合…'
  validation.value = null
  return new Promise((resolve) => {
    debounceTimer = setTimeout(async () => {
      resolve(await requestValidation())
    }, 350)
  })
}

async function requestValidation(): Promise<boolean> {
  const controller = new AbortController()
  activeController = controller
  validationLoading.value = true
  try {
    const { data } = await axios.post<ValidationResponse>(
      '/api/validate',
      normalizedParams(),
      { signal: controller.signal }
    )
    validation.value = data
    validationMessage.value = data.valid ? '' : '请先调整为推荐范围，或恢复上一次可运行参数。'
    return data.valid
  } catch (err) {
    if (axios.isCancel(err)) return false
    if (axios.isAxiosError(err) && err.response?.status === 400 && err.response.data?.detail) {
      const detail = err.response.data.detail
      validation.value = {
        valid: false,
        params: normalizedParams(),
        failedStep: detail.failedStep ?? null,
        reason: detail.message || '参数超出允许范围。',
        recommendations: detail.recommendations ?? null,
      }
      validationMessage.value = '请先调整为允许范围。'
      return false
    }
    validation.value = null
    validationMessage.value = axios.isAxiosError(err) && err.code === 'ERR_NETWORK'
      ? '参数预检服务未连接；仍可尝试开始，若失败请检查后端服务。'
      : '暂时无法完成参数预检。'
    return true
  } finally {
    if (activeController === controller) {
      activeController = null
      validationLoading.value = false
    }
  }
}

watch(
  () => ({ ...form }),
  () => {
    store.clearError()
    void validateCurrent()
  },
  { deep: true }
)

async function run() {
  const valid = await requestValidation()
  if (!valid) return
  const params = normalizedParams()
  Object.assign(form, createForm(params))
  const success = await store.runOptimization(params)
  if (success && store.result) {
    const saved = { ...params, ...store.result.params }
    lastWorkingParams.value = saved
    Object.assign(form, createForm(saved))
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ version: STORAGE_VERSION, params: saved }))
  }
}

function applyRecommendationValues(rec: ParameterRecommendations | null | undefined) {
  if (!rec) return
  if (rec.learningRate !== undefined) form.learningRate = rec.learningRate
  if (rec.momentum !== undefined) form.momentum = rec.momentum
  if (rec.iterations !== undefined && rec.iterations !== null) form.iterations = rec.iterations
}

function applyRecommendation() {
  applyRecommendationValues(recommendations.value)
}

function applyStoreRecommendation() {
  applyRecommendationValues(store.error?.recommendations)
  store.clearError()
}

function restoreLastWorking() {
  if (!lastWorkingParams.value) return
  Object.assign(form, createForm(lastWorkingParams.value))
  store.clearError()
}

function resetDefaults() {
  Object.assign(form, createForm(DEFAULT_PARAMS))
  store.clearError()
}

function onSlider(v: number) {
  store.setStep(v)
  store.pauseAnimation()
}

void validateCurrent(true)
onUnmounted(cancelValidation)
</script>

<style scoped>
.control-card { background:#fff; border-radius:8px; padding:16px 20px; box-shadow:0 2px 8px rgba(0,0,0,.06); margin-bottom:16px }
.param-alert { margin-bottom:14px; text-align:left }
.param-alert :p { margin:0 0 6px; line-height:1.5 }
.recommend-list { margin:4px 0 0 18px; padding:0; color:#606266; font-size:13px; line-height:1.6 }
.alert-actions { display:flex; gap:8px; margin-top:8px }
.field-hint { width:170px; margin-top:4px; color:#909399; font-size:12px; line-height:1.4 }
.action-item { margin-right:0 }
.validation-status { margin:-4px 0 4px; font-size:13px }
.validation-status.is-warning { color:#b88230 }
.validation-status.is-loading { color:#909399 }
.animation-bar { display:flex; align-items:center; margin-top:12px; padding-top:12px; border-top:1px solid #eee }
.anim-controls { display:flex; gap:6px }
.step-text { font-size:13px; color:#666; white-space:nowrap }
</style>
