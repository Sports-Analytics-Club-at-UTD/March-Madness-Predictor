import type { ReactNode } from "react";
import { HomeNavigation } from "@/components/home-navigation";

export default function HomeLayout({ children }: { children: ReactNode }) {
  return (
    <div className="min-h-screen">
      <HomeNavigation />
      {children}
    </div>
  );
}
