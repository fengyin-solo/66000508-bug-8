import { defineStore } from 'pinia'
import { ref } from 'vue'
import axios from 'axios'
import type { OptimizationParams, OptimizationResult, OptimizationErrorDetail } from '@/types'
import { loadLastGoodParams, saveLastGoodParams } from '../utils/params'

export const useOptimizationStore = defineStore('optimization', () => {
  const loading = ref(false)
  const result = ref<OptimizationResult | null>(null)
  const error = ref<OptimizationErrorDetail | null>(null)
  const lastGoodParams = ref<OptimizationParams | null>(loadLastGoodParams())
  const animationStep = ref(0)
  const isPlaying = ref(false)
  let playTimer: ReturnType<typeof setInterval> | null = null

  async function runOptimization(params: OptimizationParams) {
    loading.value = true
    error.value = null
    stopAnimation()
    try {
      const { data } = await axios.post('/api/optimize', params)
      result.value = data
      animationStep.value = 0
      // 跑通的一组记下来：用于"恢复上次可用参数"和刷新后回填
      const used = (data.params ?? params) as OptimizationParams
      lastGoodParams.value = used
      saveLastGoodParams(used)
    } catch (e) {
      error.value = toErrorDetail(e)
      // 保留上一次成功的可视化结果，页面不会空白
    } finally { loading.value = false }
  }

  function toErrorDetail(e: unknown): OptimizationErrorDetail {
    if (axios.isAxiosError(e)) {
      const detail = e.response?.data?.detail
      if (detail && typeof detail === 'object' && detail.message) return detail as OptimizationErrorDetail
      if (typeof detail === 'string') return { code: 'SERVER_ERROR', message: detail, issues: [] }
      if (!e.response) return { code: 'NETWORK_ERROR', message: '无法连接后端服务，请确认服务已启动后重试。', issues: [] }
      return { code: 'SERVER_ERROR', message: `服务器错误（HTTP ${e.response.status}），请稍后重试。`, issues: [] }
    }
    return { code: 'UNKNOWN', message: '发生未知错误，请稍后重试。', issues: [] }
  }

  const currentPath = () => {
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

  return {
    loading, result, error, lastGoodParams, animationStep, isPlaying, currentPath,
    runOptimization, playAnimation, pauseAnimation, stopAnimation, resetAnimation, setStep
  }
})
