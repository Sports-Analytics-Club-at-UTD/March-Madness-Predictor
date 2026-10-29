import assert from "node:assert/strict";
import { test } from "node:test";
import { selectLargeLogo } from "../scripts/lib/team-logos.mjs";

test("selects the large standard logo regardless of API ordering", () => {
  const large = "https://cdn.collegefootballdata.com/cbb-logos/500/45.png";
  assert.equal(selectLargeLogo([
    "https://cdn.collegefootballdata.com/cbb-logos/48/45.png",
    "https://cdn.collegefootballdata.com/cbb-logos-dark/500/45.png",
    large,
  ]), large);
  assert.equal(selectLargeLogo(null), null);
  assert.equal(selectLargeLogo([]), null);
  assert.equal(selectLargeLogo(["https://example.com/cbb-logos/500/45.png"]), null);
});
