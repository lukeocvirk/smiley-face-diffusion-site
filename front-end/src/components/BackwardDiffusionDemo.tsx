import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

type InitResp = { session_id: string; T: number; n: number }
type StepResp = { t: number; n: number; points: number[][]; bounds: { cx: number; cy: number; half: number } }

function drawPoints(canvas: HTMLCanvasElement, data: StepResp, color = '#1f77b4', alpha = 0.8) {
  const ctx = canvas.getContext('2d')!
  const { width, height } = canvas
  ctx.clearRect(0, 0, width, height)

  // Transform from data bounds to canvas coords (keep square aspect)
  const { cx, cy, half } = data.bounds
  const side = Math.max(half * 2, 1e-3)
  const scale = Math.min(width, height) / side
  const ox = width / 2 - (cx * scale)
  const oy = height / 2 + (cy * scale) // flip Y for screen coords

  ctx.globalAlpha = alpha
  ctx.fillStyle = color
  const r = 1.2
  for (let i = 0; i < data.points.length; i++) {
    const [x, y] = data.points[i]
    const px = ox + x * scale
    const py = oy - y * scale
    ctx.beginPath()
    ctx.arc(px, py, r, 0, Math.PI * 2)
    ctx.fill()
  }
}

export default function BackwardDiffusionDemo() {
  const MAX_T = 100
  const [session, setSession] = useState<string | null>(null)
  const [T, setT] = useState(0)
  const [t, setTVal] = useState(0)
  const [n, setN] = useState(10000)
  const [loading, setLoading] = useState(false)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)

  const displayMax = useMemo(() => Math.max(Math.min(T - 1, MAX_T), 0), [T])

  const fetchStep = useCallback(async (sid: string, step: number) => {
    const res = await fetch(`${API_BASE}/api/step?session_id=${sid}&t=${step}`)
    if (!res.ok) throw new Error('step fetch failed')
    const data: StepResp = await res.json()
    const cvs = canvasRef.current
    if (cvs) drawPoints(cvs, data)
  }, [])

  const init = useCallback(async () => {
    setLoading(true)
    try {
      const res = await fetch(`${API_BASE}/api/init`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ n }),
      })
      if (!res.ok) throw new Error('init failed')
      const data: InitResp = await res.json()
      setSession(data.session_id)
      setT(data.T)
      const t0 = Math.max(0, Math.min(data.T - 1, MAX_T))
      setTVal(t0)
      await fetchStep(data.session_id, t0)
    } finally {
      setLoading(false)
    }
  }, [n, fetchStep])

  useEffect(() => {
    init()
  }, [])

  const onScrub = async (val: number) => {
    const v = Math.max(0, Math.min(val, displayMax))
    setTVal(v)
    if (session) await fetchStep(session, v)
  }

  return (
    <div className="card">
      <h2 className="section-title">Try it Yourself</h2>
      <p>Here is a simple, scrubbable example of diffusion I created for you to try. The image starts with completely random noise at <i>t</i> = 100. As you scrub the slider to the left, the model denoises the image until it reaches <i>t</i> = 0 where the image has been sufficiently denoised by the model. You can also adjust the number of points if you like, though it may become very slow if you add too many.</p>
      <div className="controls">
        <button className="btn" onClick={init} disabled={loading}>
          {loading ? 'Preparing…' : 'Reset'}
        </button>
        <label className="label">Points</label>
        <input className="input" type="number" min={500} max={20000} step={500} value={n} onChange={e => setN(parseInt(e.target.value || '0', 10))} />
        
        <div style={{ flex: 1 }} />
        <span className="label"><i>t</i></span>
        <input
          type="range"
          min={0}
          max={displayMax}
          step={1}
          value={t}
          onChange={e => onScrub(parseInt(e.target.value, 10))}
          style={{ width: 300 }}
        />
        <span>{t} / {displayMax}</span>
      </div>
      <div className="canvas-wrap">
        <canvas ref={canvasRef} width={600} height={600} />
      </div>
    </div>
  )
}
