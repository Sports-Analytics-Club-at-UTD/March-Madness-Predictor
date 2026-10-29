import { execFileSync } from "node:child_process";
import { getTeams } from "cbbd";
import { selectLargeLogo } from "./lib/team-logos.mjs";

function run(name: string, args: object = {}) {
  return JSON.parse(execFileSync("pnpm", ["exec", "convex", "run", name, JSON.stringify(args)], {
    encoding: "utf8", stdio: ["ignore", "pipe", "inherit"],
  }).trim());
}

const apiKey = process.env.CFBD_API_KEY;
if (!apiKey) throw new Error("Set CFBD_API_KEY in .env.local before importing logos.");
// Pass credentials per request so other SDK calls don't inherit a global client key.
const { data: teams } = await getTeams({
  headers: { Authorization: `Bearer ${apiKey}` }, throwOnError: true,
});
if (!teams?.length) throw new Error("The basketball API returned no teams.");

const stored: { teamId: number; sourceUrl: string }[] = run("teamLogos:listStored");
const existing = new Map(stored.map(team => [team.teamId, team.sourceUrl]));
const missing = teams.filter(team => !selectLargeLogo(team.logos));
const pending = teams.flatMap(team => {
  const sourceUrl = selectLargeLogo(team.logos);
  if (!sourceUrl || existing.get(team.id) === sourceUrl) return [];
  return [{ teamId: team.id, teamName: team.school, displayName: team.displayName ?? team.school, sourceUrl }];
});
console.log(`${teams.length} teams; ${missing.length} without a large logo; ${pending.length} logos to import.`);
let imported = 0;
const failures: { teamName: string; reason: string }[] = [];
for (let offset = 0; offset < pending.length; offset += 20) {
  const result = run("teamLogos:importBatch", { teams: pending.slice(offset, offset + 20) });
  imported += result.imported;
  failures.push(...result.failures);
  console.log(`${Math.min(offset + 20, pending.length)}/${pending.length} processed; ${imported} stored.`);
}
console.log(`Finished: ${imported} imported, ${failures.length} failed, ${missing.length} unavailable from the API.`);
if (failures.length) {
  console.error(failures);
  process.exitCode = 1;
}
