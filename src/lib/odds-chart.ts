export type OddsPoint = { timestamp: number; price: number };

export function oddsDomain(series: { points: OddsPoint[] }[]) {
  const timestamps = series.flatMap(item => item.points.map(point => point.timestamp));
  if (!timestamps.length) return null;
  const start = Math.min(...timestamps);
  return { start, end: Math.max(start + 60, ...timestamps) };
}

export function priceAtTime(points: OddsPoint[], timestamp: number) {
  return points.findLast(point => point.timestamp <= timestamp);
}

export function oddsPath(points: OddsPoint[], domain: { start: number; end: number }) {
  return points.map((point, index) => {
    const x = 8 + (point.timestamp - domain.start) / (domain.end - domain.start) * 412;
    const y = 20 + (1 - point.price) * 190;
    return `${index === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`;
  }).join(" ");
}
