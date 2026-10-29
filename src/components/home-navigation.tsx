"use client";

import Link from "next/link";
import {
  NavigationMenu,
  NavigationMenuContent,
  NavigationMenuItem,
  NavigationMenuLink,
  NavigationMenuList,
  NavigationMenuTrigger,
  navigationMenuTriggerStyle,
} from "@/components/ui/navigation-menu";

const projects = [
  {
    title: "March Madness Predictor",
    href: "/projects/mmp",
    description: "Explore our college basketball predictions and models.",
  },
];

export function HomeNavigation() {
  return (
    <header className="sticky top-6 z-50 mx-auto mt-6 w-fit max-w-[calc(100%-2rem)]">
      <NavigationMenu
        aria-label="Main navigation"
        align="center"
        className="rounded-full border border-border/60 bg-background/70 p-1.5 shadow-sm backdrop-blur-xl"
      >
        <NavigationMenuList className="gap-1">
          <NavigationMenuItem>
            <NavigationMenuTrigger className="rounded-full px-4">
              Projects
            </NavigationMenuTrigger>
            <NavigationMenuContent>
              <ul className="grid w-72 max-w-[calc(100vw-2rem)] gap-1 p-1">
                {projects.map((project) => (
                  <li key={project.href}>
                    <NavigationMenuLink
                      render={<Link href={project.href} />}
                      className="items-start p-3"
                    >
                      <div className="space-y-1">
                        <div className="font-medium">{project.title}</div>
                        <p className="text-sm leading-relaxed text-muted-foreground">
                          {project.description}
                        </p>
                      </div>
                    </NavigationMenuLink>
                  </li>
                ))}
              </ul>
            </NavigationMenuContent>
          </NavigationMenuItem>
          <NavigationMenuItem>
            <NavigationMenuLink
              render={<Link href="/blog" />}
              className={`${navigationMenuTriggerStyle()} rounded-full px-4`}
            >
              Blog
            </NavigationMenuLink>
          </NavigationMenuItem>
        </NavigationMenuList>
      </NavigationMenu>
    </header>
  );
}
