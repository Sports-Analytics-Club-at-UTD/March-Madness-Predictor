"use client";

import { useEffect, useId, useRef, useState, type PointerEvent, type ReactNode } from "react";

// Small, dependency-free SVG charts for the write-up. Colors come from the site's
// chart tokens so both themes follow globals.css; the gray series is always a
// reference/baseline and is also dashed, so identity never relies on color alone.

export type SeriesColor = "primary" | "accent" | "reference";

const STROKE: Record<SeriesColor, string> = {
  primary: "var(--chart-1)",
  accent: "var(--chart-2)",
  reference: "var(--muted-foreground)",
};

function useWidth(fallback: number) {
  const ref = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(fallback);
  useEffect(() => {
    const element = ref.current;
    if (!element) return;
    const observer = new ResizeObserver(([entry]) => setWidth(Math.max(200, Math.round(entry.contentRect.width))));
    observer.observe(element);
    return () => observer.disconnect();
  }, []);
  return [ref, width] as const;
}

function niceTicks(min: number, max: number, count = 4) {
  const raw = (max - min) / count;
  const magnitude = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map(m => m * magnitude).find(s => s >= raw) ?? raw;
  const start = Math.floor(min / step) * step;
  const end = Math.ceil(max / step) * step;
  const ticks: number[] = [];
  for (let value = start; value <= end + step / 2; value += step) ticks.push(Number(value.toPrecision(12)));
  return ticks;
}

function decimalsFor(ticks: number[]) {
  const step = ticks.length > 1 ? Math.abs(ticks[1] - ticks[0]) : 0.01;
  let decimals = Math.max(0, Math.ceil(-Math.log10(step) - 1e-9));
  while (decimals < 6 && Math.abs(Math.round(step * 10 ** decimals) - step * 10 ** decimals) > 1e-9) decimals++;
  return decimals;
}

function Legend({ items }: { items: { label: string; color: SeriesColor }[] }) {
  return (
    <ul className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
      {items.map(item => (
        <li key={item.label} className="flex items-center gap-1.5">
          <svg width="18" height="8" aria-hidden="true">
            <line x1="1" x2="17" y1="4" y2="4" stroke={STROKE[item.color]} strokeWidth="2" strokeLinecap="round" strokeDasharray={item.color === "reference" ? "3 3" : undefined} />
          </svg>
          {item.label}
        </li>
      ))}
    </ul>
  );
}

function Tooltip({ x, y, width, children }: { x: number; y: number; width: number; children: ReactNode }) {
  const left = Math.min(Math.max(x, 90), width - 90);
  return (
    <div role="status" className="pointer-events-none absolute z-10 w-max max-w-56 -translate-x-1/2 -translate-y-full rounded-lg border border-border bg-popover px-2.5 py-1.5 text-xs text-popover-foreground shadow-md" style={{ left, top: y - 10 }}>
      {children}
    </div>
  );
}

export function ChartFrame({ title, subtitle, children, table, footnote }: { title: string; subtitle?: ReactNode; children: ReactNode; table?: ReactNode; footnote?: ReactNode }) {
  return (
    <figure className="not-prose my-8 rounded-xl border border-border bg-card p-4 shadow-sm sm:p-5">
      <figcaption className="mb-3 space-y-1">
        <div className="text-sm font-medium text-card-foreground">{title}</div>
        {subtitle && <div className="text-xs leading-relaxed text-muted-foreground">{subtitle}</div>}
      </figcaption>
      {children}
      {footnote && <p className="mt-3 text-xs leading-relaxed text-muted-foreground">{footnote}</p>}
      {table && (
        <details className="group mt-3 text-xs">
          <summary className="cursor-pointer select-none text-muted-foreground hover:text-foreground">Show data table</summary>
          <div className="mt-2 overflow-x-auto">{table}</div>
        </details>
      )}
    </figure>
  );
}

export function DataTable({ columns, rows }: { columns: string[]; rows: (string | number)[][] }) {
  return (
    <table className="w-full border-collapse text-left font-mono text-[0.7rem]">
      <thead>
        <tr>{columns.map((column, index) => <th key={index} className="border-b border-border px-2 py-1 font-sans font-medium text-muted-foreground">{column}</th>)}</tr>
      </thead>
      <tbody>
        {rows.map((row, index) => (
          <tr key={index} className="border-b border-border/60 last:border-0">
            {row.map((cell, cellIndex) => <td key={cellIndex} className="px-2 py-1 tabular-nums">{cell}</td>)}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export type LineSeries = { key: string; label: string; color: SeriesColor; values: number[] };

export function LineChart({ categories, series, height = 240, ariaLabel, yLabel = "Brier ↓", markers = [], highlightMin = false, format = (value: number) => value.toFixed(4) }: {
  categories: (string | number)[];
  series: LineSeries[];
  height?: number;
  ariaLabel: string;
  yLabel?: string;
  markers?: { index: number; label: string }[];
  highlightMin?: boolean;
  format?: (value: number) => string;
}) {
  const [ref, width] = useWidth(640);
  const [active, setActive] = useState<number | null>(null);
  const clipId = useId();
  const all = series.flatMap(s => s.values);
  const ticks = niceTicks(Math.min(...all), Math.max(...all));
  const decimals = decimalsFor(ticks);
  const directLabels = series.length > 1 && series.length <= 4 && width >= 520;
  const margin = { top: 16, right: directLabels ? 92 : 16, bottom: 28, left: 52 };
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;
  const x = (index: number) => margin.left + (categories.length === 1 ? innerWidth / 2 : (index / (categories.length - 1)) * innerWidth);
  const yMin = ticks[0];
  const yMax = ticks[ticks.length - 1];
  const y = (value: number) => margin.top + (1 - (value - yMin) / (yMax - yMin)) * innerHeight;
  const labelEvery = Math.ceil(categories.length / Math.max(2, Math.floor(innerWidth / 44)));

  function onMove(event: PointerEvent<SVGRectElement>) {
    const bounds = event.currentTarget.getBoundingClientRect();
    const relative = (event.clientX - bounds.left) / bounds.width;
    setActive(Math.round(relative * (categories.length - 1)));
  }

  // Spread end labels so they never overlap.
  const ends = series.map(s => ({ key: s.key, label: s.label, y: y(s.values[s.values.length - 1]) })).sort((a, b) => a.y - b.y);
  for (let index = 1; index < ends.length; index++) ends[index].y = Math.max(ends[index].y, ends[index - 1].y + 13);

  return (
    <div className="space-y-2">
      {series.length > 1 && <Legend items={series} />}
      <div ref={ref} className="relative">
        <svg width={width} height={height} role="img" aria-label={ariaLabel} className="block overflow-visible">
          <defs>
            <clipPath id={clipId}><rect x={margin.left - 6} y={margin.top - 6} width={innerWidth + 12} height={innerHeight + 12} /></clipPath>
          </defs>
          {ticks.map(tick => (
            <g key={tick}>
              <line x1={margin.left} x2={width - margin.right} y1={y(tick)} y2={y(tick)} stroke="var(--border)" strokeWidth="1" strokeDasharray={tick === yMin ? undefined : "2 4"} />
              <text x={margin.left - 8} y={y(tick)} dy="0.32em" textAnchor="end" className="fill-muted-foreground font-mono text-[10px] tabular-nums">{tick.toFixed(decimals)}</text>
            </g>
          ))}
          <text x={margin.left - 8} y={margin.top - 6} textAnchor="end" className="fill-muted-foreground text-[10px]">{yLabel}</text>
          {categories.map((category, index) => (index % labelEvery === 0 || index === categories.length - 1) && (
            <text key={category} x={x(index)} y={height - 8} textAnchor="middle" className="fill-muted-foreground font-mono text-[10px] tabular-nums">{category}</text>
          ))}
          {markers.map(marker => (
            <g key={marker.label}>
              <line x1={x(marker.index)} x2={x(marker.index)} y1={margin.top} y2={margin.top + innerHeight} stroke="var(--foreground)" strokeOpacity="0.35" strokeDasharray="4 3" />
              <text x={x(marker.index) + 4} y={margin.top + 8} className="fill-muted-foreground text-[10px]">{marker.label}</text>
            </g>
          ))}
          {active !== null && <line x1={x(active)} x2={x(active)} y1={margin.top} y2={margin.top + innerHeight} stroke="var(--foreground)" strokeOpacity="0.2" />}
          <g clipPath={`url(#${clipId})`}>
            {series.map(s => (
              <polyline key={s.key} fill="none" stroke={STROKE[s.color]} strokeWidth="2" strokeLinejoin="round" strokeLinecap="round" strokeDasharray={s.color === "reference" ? "5 4" : undefined}
                points={s.values.map((value, index) => `${x(index)},${y(value)}`).join(" ")} />
            ))}
            {series.map(s => s.values.map((value, index) => {
              const isMin = highlightMin && value === Math.min(...s.values);
              const show = active === index || isMin || categories.length <= 8;
              return show ? <circle key={`${s.key}-${index}`} cx={x(index)} cy={y(value)} r={isMin || active === index ? 4.5 : 3} fill={STROKE[s.color]} stroke="var(--card)" strokeWidth="2" /> : null;
            }))}
          </g>
          {directLabels && ends.map(end => (
            <text key={end.key} x={width - margin.right + 8} y={end.y} dy="0.32em" className="fill-muted-foreground text-[10px]">{end.label}</text>
          ))}
          <rect x={margin.left} y={margin.top} width={innerWidth} height={innerHeight} fill="transparent" onPointerMove={onMove} onPointerLeave={() => setActive(null)} />
        </svg>
        {active !== null && (
          <Tooltip x={x(active)} y={Math.min(...series.map(s => y(s.values[active])))} width={width}>
            <div className="mb-0.5 font-medium">{categories[active]}</div>
            {series.map(s => (
              <div key={s.key} className="flex items-center justify-between gap-3">
                <span className="flex items-center gap-1.5 text-muted-foreground"><span className="inline-block size-2 rounded-full" style={{ background: STROKE[s.color] }} />{s.label}</span>
                <span className="font-mono tabular-nums">{format(s.values[active])}</span>
              </div>
            ))}
          </Tooltip>
        )}
      </div>
    </div>
  );
}

export type DotRow = { key: string; label: string; value: number; color: SeriesColor; detail?: string };

export function DotPlot({ rows, references = [], ariaLabel, domain, format = (value: number) => value.toFixed(4) }: {
  rows: DotRow[];
  references?: { value: number; label: string }[];
  ariaLabel: string;
  domain?: [number, number];
  format?: (value: number) => string;
}) {
  const [ref, width] = useWidth(640);
  const [active, setActive] = useState<string | null>(null);
  const values = [...rows.map(r => r.value), ...references.map(r => r.value)];
  const ticks = niceTicks(domain?.[0] ?? Math.min(...values), domain?.[1] ?? Math.max(...values), width < 420 ? 3 : 5);
  const decimals = decimalsFor(ticks);
  const labelWidth = width < 480 ? 0 : 196;
  const rowHeight = width < 480 ? 46 : 34;
  const margin = { top: references.length ? 34 : 10, right: 20, bottom: 26, left: labelWidth + 12 };
  const height = margin.top + rows.length * rowHeight + margin.bottom;
  const innerWidth = width - margin.left - margin.right;
  const min = ticks[0];
  const max = ticks[ticks.length - 1];
  const x = (value: number) => margin.left + ((value - min) / (max - min)) * innerWidth;
  const rowY = (index: number) => margin.top + index * rowHeight + rowHeight / 2 + (labelWidth ? 0 : 7);
  const activeRow = rows.find(row => row.key === active);
  const activeIndex = rows.findIndex(row => row.key === active);

  return (
    <div ref={ref} className="relative">
      <svg width={width} height={height} role="img" aria-label={ariaLabel} className="block overflow-visible">
        {ticks.map(tick => (
          <g key={tick}>
            <line x1={x(tick)} x2={x(tick)} y1={margin.top} y2={height - margin.bottom} stroke="var(--border)" strokeDasharray="2 4" />
            <text x={x(tick)} y={height - 8} textAnchor="middle" className="fill-muted-foreground font-mono text-[10px] tabular-nums">{tick.toFixed(decimals)}</text>
          </g>
        ))}
        {references.map((reference, index) => (
          <g key={reference.label}>
            <line x1={x(reference.value)} x2={x(reference.value)} y1={margin.top - 4} y2={height - margin.bottom} stroke="var(--foreground)" strokeOpacity="0.45" strokeDasharray="5 3" />
            <text x={x(reference.value)} y={margin.top - (index % 2 ? 8 : 20)} textAnchor="middle" className="fill-muted-foreground text-[10px]">{reference.label}</text>
          </g>
        ))}
        {rows.map((row, index) => (
          <g key={row.key} onPointerEnter={() => setActive(row.key)} onPointerLeave={() => setActive(null)}>
            <rect x={0} y={rowY(index) - rowHeight / 2 - (labelWidth ? 0 : 7)} width={width} height={rowHeight} fill="transparent" />
            {labelWidth
              ? <text x={labelWidth} y={rowY(index)} dy="0.32em" textAnchor="end" className="fill-foreground text-[11px]">{row.label}</text>
              : <text x={margin.left} y={rowY(index) - 13} className="fill-foreground text-[11px]">{row.label}</text>}
            <line x1={margin.left} x2={width - margin.right} y1={rowY(index)} y2={rowY(index)} stroke="var(--border)" strokeOpacity="0.6" />
            <circle cx={x(row.value)} cy={rowY(index)} r={active === row.key ? 6.5 : 5.5} fill={STROKE[row.color]} stroke="var(--card)" strokeWidth="2" />
          </g>
        ))}
      </svg>
      {activeRow && (
        <Tooltip x={x(activeRow.value)} y={rowY(activeIndex) - 6} width={width}>
          <div className="font-medium">{activeRow.label}</div>
          <div className="font-mono tabular-nums">{format(activeRow.value)}</div>
          {activeRow.detail && <div className="text-muted-foreground">{activeRow.detail}</div>}
        </Tooltip>
      )}
    </div>
  );
}
