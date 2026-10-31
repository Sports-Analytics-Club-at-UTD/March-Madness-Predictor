import { flushSync } from "react-dom";

export async function toggleThemeWithReveal(
  event: { clientX: number; clientY: number; detail: number; currentTarget: HTMLButtonElement },
  nextTheme: "light" | "dark",
  setTheme: (theme: string) => void,
) {
  const root = document.documentElement;
  if (root.dataset.themeReveal) return;
  let applied = false;
  const updateTheme = () => {
    if (applied) return;
    applied = true;
    flushSync(() => setTheme(nextTheme));
  };
  if (!document.startViewTransition || window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
    updateTheme();
    return;
  }

  const bounds = event.currentTarget.getBoundingClientRect();
  const x = event.detail === 0 ? bounds.left + bounds.width / 2 : event.clientX;
  const y = event.detail === 0 ? bounds.top + bounds.height / 2 : event.clientY;
  const radius = Math.hypot(Math.max(x, window.innerWidth - x), Math.max(y, window.innerHeight - y));
  root.dataset.themeReveal = "true";
  let transition: ViewTransition | undefined;
  try {
    transition = document.startViewTransition(updateTheme);
    await transition.ready;
    await root.animate(
      { clipPath: [`circle(0px at ${x}px ${y}px)`, `circle(${radius}px at ${x}px ${y}px)`] },
      { duration: 720, easing: "cubic-bezier(.76, 0, .24, 1)", pseudoElement: "::view-transition-new(root)" },
    ).finished;
  } catch {
    transition?.skipTransition();
    updateTheme();
  } finally {
    await transition?.finished.catch(() => {});
    delete root.dataset.themeReveal;
  }
}
