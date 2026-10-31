"use client";

import { useId, useState, type PointerEvent } from "react";
import type { KalshiChart } from "../../convex/kalshiChartModel";
import { oddsDomain, oddsPath, priceAtTime } from "@/lib/odds-chart";

const timeFormat = new Intl.DateTimeFormat("en-US", { hour: "numeric", minute: "2-digit", timeZone: "America/New_York" });
const dateFormat = new Intl.DateTimeFormat("en-US", { month: "short", day: "numeric", year: "numeric", timeZone: "America/New_York" });
const formatPrice = (price: number | undefined) => price === undefined ? "—" : `${Math.round(price * 1000) / 10}%`;

export function KalshiOddsChart({ chart }: { chart: KalshiChart }) {
  const [hoverTime, setHoverTime] = useState<number | null>(null);
  const titleId = useId();
  const domain = oddsDomain(chart.series);
  const series = chart.series.map((item, index) => ({ ...item, color: ["var(--chart-1)", "var(--chart-5)"][index % 2] }));
  const values = series.map(item => hoverTime === null ? item.price : priceAtTime(item.points, hoverTime)?.price);
  const labelYs = values.map(price => Math.max(34, Math.min(178, 20 + (1 - (price ?? 0.5)) * 190 - 8)));
  if (labelYs.length === 2 && Math.abs(labelYs[0] - labelYs[1]) < 40) {
    const upper = labelYs[0] <= labelYs[1] ? 0 : 1;
    labelYs[upper] = Math.max(34, Math.min(138, labelYs[upper] - 20));
    labelYs[1 - upper] = labelYs[upper] + 42;
  }

  function scrub(event: PointerEvent<SVGSVGElement>) {
    if (!domain) return;
    const bounds = event.currentTarget.getBoundingClientRect();
    const x = (event.clientX - bounds.left) / bounds.width * 560;
    setHoverTime(x >= 8 && x <= 420 ? domain.start + (x - 8) / 412 * (domain.end - domain.start) : null);
  }

  return (
    <div>
      <div className="flex items-center justify-between border-b border-border px-4 py-3">
        <div className="flex items-center gap-2 text-xs font-medium text-card-foreground">
          <svg aria-hidden="true" viewBox="0 0 16 16" className="size-3.5 fill-current"><path d="M2 8h3v6H2zm5-5h3v11H7zm5 3h3v8h-3z" /></svg>
          Stats
          <span className={`ml-1 size-1.5 rounded-full ${chart.status === "live" ? "bg-primary" : "bg-muted-foreground"}`} />
          <span className="text-muted-foreground">{chart.status === "live" ? "Live" : chart.status === "settled" ? "Settled" : "Market closed"}</span>
        </div>
        <span className="text-lg font-bold tracking-tight text-muted-foreground">Kalshi</span>
      </div>
      <svg
        viewBox="0 0 560 250"
        role="img"
        aria-labelledby={titleId}
        tabIndex={0}
        className="w-full touch-pan-y outline-none focus-visible:ring-1 focus-visible:ring-inset focus-visible:ring-ring"
        onPointerMove={scrub}
        onPointerLeave={() => setHoverTime(null)}
        onBlur={() => setHoverTime(null)}
        onKeyDown={event => {
          if (!domain || !["ArrowLeft", "ArrowRight", "Home", "End", "Escape"].includes(event.key)) return;
          event.preventDefault();
          if (event.key === "Escape") setHoverTime(null);
          else if (event.key === "Home") setHoverTime(domain.start);
          else if (event.key === "End") setHoverTime(domain.end);
          else setHoverTime(Math.max(domain.start, Math.min(domain.end, (hoverTime ?? domain.end) + (event.key === "ArrowLeft" ? -60 : 60))));
        }}
      >
        <title id={titleId}>{series.map((item, index) => `${item.teamName}: ${formatPrice(values[index])}`).join("; ")}. Move over the chart or use arrow keys to inspect historical prices.</title>
        {[1, 0.75, 0.5, 0.25, 0].map(price => (
          <g key={price}>
            <line x1={8} x2={530} y1={20 + (1 - price) * 190} y2={20 + (1 - price) * 190} stroke="var(--border)" strokeDasharray="1 4" />
            <text x={554} y={24 + (1 - price) * 190} textAnchor="end" fill="var(--muted-foreground)" fontSize={10}>{price * 100}%</text>
          </g>
        ))}
        {domain && series.map((item, index) => {
          const point = hoverTime === null ? item.points.at(-1) : priceAtTime(item.points, hoverTime);
          return (
            <g key={item.ticker}>
              <path d={oddsPath(item.points, domain)} fill="none" stroke={item.color} strokeWidth={1.8} strokeLinejoin="round" />
              {point && <circle cx={8 + (point.timestamp - domain.start) / (domain.end - domain.start) * 412} cy={20 + (1 - point.price) * 190} r={4} fill="var(--card)" stroke={item.color} strokeWidth={2} />}
              <g transform={`translate(440 ${labelYs[index]})`} fill={item.color}>
                <text fontSize={11} fontWeight={600}>{item.ticker.split("-").at(-1)}</text>
                <text y={23} fontSize={22} fontWeight={600}>{formatPrice(values[index])}</text>
              </g>
            </g>
          );
        })}
        {!domain && <text x={280} y={120} textAnchor="middle" fill="var(--muted-foreground)" fontSize={13}>No price history available yet</text>}
        {domain && [0, 1, 2, 3, 4].map(index => (
          <text key={index} x={8 + index * 103} y={238} textAnchor={index === 0 ? "start" : index === 4 ? "end" : "middle"} fill="var(--muted-foreground)" fontSize={10}>
            {timeFormat.format(new Date((domain.start + (domain.end - domain.start) * index / 4) * 1000))}
          </text>
        ))}
        {domain && hoverTime !== null && (
          <g>
            <line x1={8 + (hoverTime - domain.start) / (domain.end - domain.start) * 412} x2={8 + (hoverTime - domain.start) / (domain.end - domain.start) * 412} y1={20} y2={210} stroke="var(--muted-foreground)" strokeDasharray="3 3" />
            <rect x={Math.max(8, Math.min(340, 8 + (hoverTime - domain.start) / (domain.end - domain.start) * 412 - 36))} y={223} width={80} height={23} rx={4} fill="var(--muted)" />
            <text x={Math.max(48, Math.min(380, 8 + (hoverTime - domain.start) / (domain.end - domain.start) * 412 + 4))} y={238} textAnchor="middle" fill="var(--foreground)" fontSize={10}>{timeFormat.format(new Date(hoverTime * 1000))}</text>
          </g>
        )}
      </svg>
      <div className="flex flex-wrap items-center justify-between gap-2 border-t border-border px-4 py-2.5 text-[10px] text-muted-foreground">
        <span>{dateFormat.format(new Date((domain?.end ?? chart.fetchedAt / 1000) * 1000))} · ET{chart.status === "live" ? " · Updates every 15s" : ""}</span>
        <span>{chart.marketKind === "championship" ? "Tournament winner" : "Game winner"} · {hoverTime === null && chart.status === "settled" ? "Final outcome" : "Implied probability"}</span>
      </div>
      <div className="sr-only">{series.map((item, index) => <p key={item.ticker}>{item.teamName}: {formatPrice(values[index])}</p>)}</div>
    </div>
  );
}
