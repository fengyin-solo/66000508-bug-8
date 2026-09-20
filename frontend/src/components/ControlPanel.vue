<template>
  <div class="control-card">
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
        <el-input-number v-model="form.x0" :min="-10" :max="10" :step="0.5" size="small" />
      </el-form-item>
      <el-form-item label="初始Y">
        <el-input-number v-model="form.y0" :min="-10" :max="10" :step="0.5" size="small" />
      </el-form-item>
      <el-form-item label="学习率">
        <el-input-number v-model="form.learningRate" :min="0.001" :max="1" :step="0.01" :precision="3" size="small" />
      </el-form-item>
      <el-form-item label="迭代">
        <el-input-number v-model="form.iterations" :min="10" :max="500" :step="10" size="small" />
      </el-form-item>
      <el-form-item label="动量" v-if="form.algorithm==='gradient_descent'">
        <el-input-number v-model="form.momentum" :min="0" :max="0.99" :step="0.1" :precision="1" size="small" />
      </el-form-item>
      <el-form-item label="温度" v-if="form.algorithm==='simulated_annealing'">
        <el-input-number v-model="form.temperature" :min="1" :max="1000" :step="10" size="small" />
      </el-form-item>
      <el-form-item>
        <el-button type="primary" @click="run" :loading="store.loading">🚀 开始优化</el-button>
        <el-button v-if="store.lastGoodParams" @click="restoreLastGood" size="default">↩ 恢复上次可用参数</el-button>
      </el-form-item>
    </el-form>

    <el-alert v-if="clientIssues.length" type="warning" :closable="false" class="param-alert" title="参数超出允许范围，未提交">
      <ul class="issue-list">
        <li v-for="(issue, i) in clientIssues" :key="i">{{ issue.reason }}</li>
      </ul>
    </el-alert>

    <el-alert v-if="store.error" type="error" :closable="false" class="param-alert" :title="store.error.message">
      <ul v-if="store.error.issues?.length > 1" class="issue-list">
        <li v-for="(issue, i) in store.error.issues" :key="i">{{ issue.reason }}</li>
      </ul>
      <div v-if="limitEntries.length" class="limits">
        <span class="limits-title">推荐范围：</span>
        <el-tag v-for="[field, range] in limitEntries" :key="field" size="small" class="limit-tag">
          {{ labelOf(field) }} [{{ range[0] }}, {{ range[1] }}]
        </el-tag>
        <el-tag v-for="(ub, field) in store.error.upperBounds" :key="'ub-'+field" size="small" type="danger" class="limit-tag">
          {{ labelOf(field) }}上限 ≈ {{ ub }}
        </el-tag>
      </div>
      <div class="alert-actions">
        <el-button v-if="store.error.recommendedParams" size="small" type="primary" @click="applyRecommended">
          ✨ 使用推荐参数并重新运行
        </el-button>
        <el-button v-if="store.lastGoodParams" size="small" @click="restoreLastGood">
          ↩ 恢复上次可用参数
        </el-button>
      </div>
    </el-alert>

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
import { reactive, ref, computed, watch } from 'vue'
import { ElMessage } from 'element-plus'
import { useOptimizationStore } from '../store/optimization'
import { TEST_FUNCTIONS, ALGORITHMS } from '../types'
import type { OptimizationParams, ParamIssue } from '../types'
import {
  DEFAULT_PARAMS, FIELD_LABELS, normalizeParams, validateParams,
  type ParamsForm
} from '../utils/params'

const store = useOptimizationStore()

// 刷新后回到面板：优先回填上次跑通的那组，否则用默认值
const form = reactive<ParamsForm>(
  normalizeParams(store.lastGoodParams ?? { ...DEFAULT_PARAMS }).params as ParamsForm
)

const clientIssues = ref<ParamIssue[]>([])
const animStep = ref(0)
const maxStep = computed(() => Math.max(0, (store.result?.path.length || 1) - 1))
const limitEntries = computed(() => Object.entries(store.error?.limits ?? {}))

watch(() => store.animationStep, (v) => { animStep.value = v })

function labelOf(field: string) { return FIELD_LABELS[field] ?? field }

function run() {
  // 空着或只填一半时按默认值补全，并回显到面板上
  const { params, filled } = normalizeParams(form)
  Object.assign(form, params)
  if (filled.length) {
    ElMessage.info(`已按默认值补全：${filled.map(labelOf).join('、')}`)
  }
  clientIssues.value = validateParams(params)
  if (clientIssues.value.length) return
  store.runOptimization(params)
}

function applyRecommended() {
  const rec = store.error?.recommendedParams
  if (!rec) return
  Object.assign(form, rec)
  clientIssues.value = []
  run()
}

function restoreLastGood() {
  if (!store.lastGoodParams) return
  Object.assign(form, store.lastGoodParams as OptimizationParams)
  clientIssues.value = []
  store.error = null
  ElMessage.success('已恢复上次跑通的参数')
}

function onSlider(v: number) { store.setStep(v); store.pauseAnimation() }
</script>

<style scoped>
.control-card { background:#fff; border-radius:8px; padding:16px 20px; box-shadow:0 2px 8px rgba(0,0,0,.06); margin-bottom:16px }
.animation-bar { display:flex; align-items:center; margin-top:12px; padding-top:12px; border-top:1px solid #eee }
.anim-controls { display:flex; gap:6px }
.step-text { font-size:13px; color:#666; white-space:nowrap }
.param-alert { margin-top:12px }
.issue-list { margin:4px 0 0; padding-left:18px }
.limits { margin-top:8px; display:flex; flex-wrap:wrap; align-items:center; gap:6px }
.limits-title { font-size:12px; color:#666 }
.limit-tag { font-family:monospace }
.alert-actions { margin-top:10px; display:flex; gap:8px }
</style>
