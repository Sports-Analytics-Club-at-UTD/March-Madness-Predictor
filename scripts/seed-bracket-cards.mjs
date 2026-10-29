import { execFileSync } from "node:child_process";
import { readFile } from "node:fs/promises";

function run(name, args = {}) {
  const output = execFileSync("npx", ["convex", "run", name, JSON.stringify(args)], { encoding: "utf8", stdio: ["ignore", "pipe", "inherit"] }).trim();
  return output ? JSON.parse(output) : null;
}
let storageId = run("brackets:getSeedLogo");
if (!storageId) {
  const uploadUrl = run("brackets:generateUploadUrl");
  const response = await fetch(uploadUrl, {
    method: "POST",
    headers: { "Content-Type": "image/x-icon" },
    body: await readFile(new URL("../public/next-favicon.ico", import.meta.url)),
  });
  if (!response.ok) throw new Error(`Favicon upload failed: ${response.status}`);
  ({ storageId } = await response.json());
}
const bracketIds = run("brackets:seedExamples", { logoStorageId: storageId });
console.log(`Default favicon stored in Convex; ${bracketIds.length} bracket cards ready.`);
