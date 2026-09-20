import { defineStore } from 'pinia'
import { ref } from 'vue'
import axios from 'axios'
import type { OptimizationParams, OptimizationResult, IterationPoint, ParameterRecommendations } from '@/types'

export interface OptimizationErrorState {
  message: string
  failedStep?: number | null
  recommendations?: ParameterRecommendations | null
}

export const useOptimizationStore = defineStore('optimization', () => {
  const loading = ref(false)
  const result = ref<OptimizationResult | null>(null)
  const error = ref<OptimizationErrorState | null>(null)
  const animationStep = ref(0)
  const isPlaying = ref(false)
  let playTimer: ReturnType<typeof setInterval> | null = null

  function toErrorState(err: unknown): OptimizationErrorState {
    if (axios.isAxiosError(err)) {
      const detail = err.response?.data?.detail
      if (typeof detail === 'object' && detail !== null) {
        return {
          message: detail.message || '这组参数无法完成计算，请按推荐范围调整。',
          failedStep: detail.failedStep ?? null,
          recommendations: detail.recommendations ?? null,
        }
      }
      if (typeof detail === 'string' && detail) return { message: detail }
      if (err.code === 'ERR_NETWORK') return { message: '无法连接优化服务，请确认后端已在 8000 端口启动。' }
      return { message: '优化服务暂时不可用，请稍后重试。' }
    }
    return { message: '发生未知错误，请检查参数后重试。' }
  }

  async function runOptimization(params: OptimizationParams): Promise<boolean> {
    loading.value = true
    error.value = null
    stopAnimation()
    try {
      const { data } = await axios.post<OptimizationResult>('/api/optimize', params)
      result.value = data
      animationStep.value = 0
      return true
    } catch (err) {
      error.value = toErrorState(err)
      return false
    } finally {
      loading.value = false
    }
  }

  const currentPath = (): IterationPoint[] => {
    if (!result.value) return []
    return result.value.path.slice(0, animationStep.value + 1)
  }

  function playAnimation() {
    if (!result.value) return
    isPlaying.value = true
    playTimer = setInterval(() => {
      if (animationStep.value < (result.value?.path.length || 0) - 1) {
        animationStep.value++
      } else {
        stopAnimation()
      }
    }, 80)
  }

  function pauseAnimation() { stopAnimation() }
  function stopAnimation() {
    isPlaying.value = false
    if (playTimer) { clearInterval(playTimer); playTimer = null }
  }

  function resetAnimation() { stopAnimation(); animationStep.value = 0 }
  function setStep(step: number) { animationStep.value = step }
  function clearError() { error.value = null }

  return {
    loading, result, error, animationStep, isPlaying, currentPath,
    runOptimization, playAnimation, pauseAnimation, stopAnimation, resetAnimation, setStep, clearError
  }
})
