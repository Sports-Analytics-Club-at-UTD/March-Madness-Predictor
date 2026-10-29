"use client";

import { createContext, useContext, useState, type ReactNode } from "react";
import { useQuery } from "convex/react";
import type { FunctionReturnType } from "convex/server";
import { api } from "../../convex/_generated/api";
import type { Id } from "../../convex/_generated/dataModel";

type BracketWorkspace = {
  brackets: FunctionReturnType<typeof api.brackets.list> | undefined;
  selectedId: Id<"brackets"> | null;
  selectBracket: (id: Id<"brackets">) => void;
};
const WorkspaceContext = createContext<BracketWorkspace | null>(null);

export function BracketWorkspaceProvider({ children }: { children: ReactNode }) {
  const brackets = useQuery(api.brackets.list);
  const [selection, selectBracket] = useState<Id<"brackets"> | null>(null);
  const selectedId = brackets?.find(bracket => bracket._id === selection)?._id ?? brackets?.[0]?._id ?? null;
  return <WorkspaceContext.Provider value={{ brackets, selectedId, selectBracket }}>{children}</WorkspaceContext.Provider>;
}

export function useBracketWorkspace() {
  const workspace = useContext(WorkspaceContext);
  if (!workspace) throw new Error("Bracket workspace provider is missing.");
  return workspace;
}
