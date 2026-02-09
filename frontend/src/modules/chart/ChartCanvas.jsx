import { useEffect, useRef } from 'react'
import { createChart } from 'lightweight-charts'
import { getEventColor } from './chartUtils'

export default function ChartCanvas({ 
  data, 
  eventIndex, 
  chartType = 'line',
  showVolume = false,
  height = 200 
}) {
  const containerRef = useRef(null)
  const chartRef = useRef(null)

  useEffect(() => {
    if (!containerRef.current || !data?.candles?.length) return

    // Chart erstellen
    const chart = createChart(containerRef.current, {
      width: containerRef.current.clientWidth,
      height: height,
      layout: {
        background: { color: 'transparent' },
        textColor: '#9ca3af',
      },
      grid: {
        vertLines: { color: '#374151' },
        horzLines: { color: '#374151' },
      },
      crosshair: {
        mode: 0, // Normal
      },
      rightPriceScale: {
        borderColor: '#374151',
      },
      timeScale: {
        borderColor: '#374151',
        timeVisible: true,
      },
    })

    chartRef.current = chart

    // Candlestick oder Line
    if (chartType === 'candle') {
      const candleSeries = chart.addCandlestickSeries({
        upColor: '#22c55e',
        downColor: '#ef4444',
        borderUpColor: '#22c55e',
        borderDownColor: '#ef4444',
        wickUpColor: '#22c55e',
        wickDownColor: '#ef4444',
      })
      
      const candleData = data.candles.map(c => ({
        time: c.relativeTime,
        open: c.open,
        high: c.high,
        low: c.low,
        close: c.close,
      }))
      candleSeries.setData(candleData)
    } else {
      const lineSeries = chart.addLineSeries({
        color: getEventColor(eventIndex),
        lineWidth: 2,
      })
      
      const lineData = data.candles.map(c => ({
        time: c.relativeTime,
        value: c.close,
      }))
      lineSeries.setData(lineData)
    }

    // Volume Histogram
    if (showVolume) {
      const volumeSeries = chart.addHistogramSeries({
        color: '#3b82f6',
        priceFormat: { type: 'volume' },
        priceScaleId: 'volume',
      })
      
      chart.priceScale('volume').applyOptions({
        scaleMargins: { top: 0.8, bottom: 0 },
      })

      const volumeData = data.candles.map(c => ({
        time: c.relativeTime,
        value: c.volume,
        color: c.close >= c.open ? '#22c55e50' : '#ef444450',
      }))
      volumeSeries.setData(volumeData)
    }

    // Event-Start Markierung
    if (data.eventStartTime) {
      chart.timeScale().setVisibleRange({
        from: data.candles[0]?.relativeTime || 0,
        to: data.candles[data.candles.length - 1]?.relativeTime || 100,
      })
    }

    // Resize Handler
    const handleResize = () => {
      if (containerRef.current) {
        chart.applyOptions({ width: containerRef.current.clientWidth })
      }
    }
    window.addEventListener('resize', handleResize)

    return () => {
      window.removeEventListener('resize', handleResize)
      chart.remove()
    }
  }, [data, eventIndex, chartType, showVolume, height])

  return (
    <div ref={containerRef} className="w-full" style={{ height }} />
  )
}
