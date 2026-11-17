"use client";

import type { MouseEvent } from "react";
import { Moon, Sun } from "lucide-react";
import { useTheme } from "@/components/theme-provider";
import { Button } from "@/components/ui/button";
import { toggleThemeWithReveal } from "@/lib/theme-transition";

export function ThemeToggleButton({ className }: { className?: string }) {
  const { resolvedTheme, setTheme } = useTheme();
  function toggle(event: MouseEvent<HTMLButtonElement>) {
    void toggleThemeWithReveal(event, resolvedTheme === "dark" ? "light" : "dark", setTheme);
  }
  return (
    <Button variant="ghost" size="icon" className={className} onClick={toggle} aria-label="Toggle light and dark theme" title="Toggle light and dark theme">
      <Moon className="dark:hidden" />
      <Sun className="hidden dark:block" />
    </Button>
  );
}
