"use client";

import { ThemeProvider as NextThemesProvider, useTheme as useNextTheme } from "next-themes";
import { useCallback, type ComponentProps } from "react";

export function ThemeProvider({ children, ...props }: ComponentProps<typeof NextThemesProvider>) {
  return (
    <NextThemesProvider attribute="class" defaultTheme="system" storageKey="ui-theme" enableSystem disableTransitionOnChange {...props}>
      {children}
    </NextThemesProvider>
  );
}

// karppa.dev applies the class synchronously so view-transition snapshots see the new theme.
export function useTheme() {
  const theme = useNextTheme();
  const { setTheme: saveTheme } = theme;
  const setTheme = useCallback((nextTheme: string) => {
    const resolved = nextTheme === "system"
      ? window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"
      : nextTheme;
    document.documentElement.classList.remove("light", "dark");
    document.documentElement.classList.add(resolved);
    document.documentElement.style.colorScheme = resolved;
    saveTheme(nextTheme);
  }, [saveTheme]);
  return { ...theme, setTheme };
}
