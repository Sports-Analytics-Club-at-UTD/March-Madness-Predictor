/* eslint-disable @typescript-eslint/no-require-imports -- Use the existing TypeScript compiler to test the browser integration. */
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const { test } = require("node:test");
const ts = require("typescript");

const compiled = ts.transpileModule(fs.readFileSync("src/lib/theme-transition.ts", "utf8"), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
}).outputText;
const providerCompiled = ts.transpileModule(fs.readFileSync("src/components/theme-provider.tsx", "utf8"), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, jsx: ts.JsxEmit.ReactJSX },
}).outputText;

function browser({ supported = true, reduced = false, skipped = false, animation = Promise.resolve(), osDark = false } = {}) {
  const classes = new Set(["font-token", "light"]);
  const preferences = [];
  const snapshots = [];
  const animations = [];
  const root = {
    dataset: {}, style: { colorScheme: "light" },
    classList: {
      contains: name => classes.has(name),
      remove: (...names) => names.forEach(name => classes.delete(name)),
      add: name => classes.add(name),
    },
    animate: (frames, options) => { animations.push({ frames, options }); return { finished: animation }; },
  };
  const document = { documentElement: root };
  if (supported) document.startViewTransition = update => {
    update();
    snapshots.push({ dark: classes.has("dark"), light: classes.has("light"), colorScheme: root.style.colorScheme });
    return { ready: skipped ? Promise.reject(new Error("Transition skipped")) : Promise.resolve(),
      finished: Promise.resolve(), skipTransition() {} };
  };
  const window = { innerWidth: 1000, innerHeight: 800, matchMedia: query => ({ matches: query.includes("reduced-motion") ? reduced : osDark }) };
  let theme = "light";
  const provider = {};
  vm.runInNewContext(providerCompiled, { exports: provider, document, window, require: name => {
    if (name === "next-themes") return { useTheme: () => ({ resolvedTheme: theme === "system" ? osDark ? "dark" : "light" : theme,
      setTheme: next => { theme = next; preferences.push(next); } }) };
    if (name === "react") return { useCallback: callback => callback };
    if (name === "react/jsx-runtime") return {};
    throw new Error(`Unexpected import: ${name}`);
  } });
  const exports = {};
  vm.runInNewContext(compiled, { exports, document, window, require: name => {
    assert.equal(name, "react-dom");
    return { flushSync: callback => callback() };
  } });
  return { root, classes, preferences, snapshots, animations,
    useTheme: provider.useTheme,
    toggle: (event = { clientX: 125, clientY: 235, detail: 1 }) => {
      const { resolvedTheme, setTheme } = provider.useTheme();
      return exports.toggleThemeWithReveal({ ...event, currentTarget: { getBoundingClientRect: () => ({ left: 100, top: 200, width: 20, height: 20 }) } },
        resolvedTheme === "dark" ? "light" : "dark", setTheme);
    },
  };
}

test("the new theme is in the DOM at snapshot time without waiting for React effects", async () => {
  const app = browser();
  await app.toggle();
  assert.deepEqual(app.snapshots, [{ dark: true, light: false, colorScheme: "dark" }]);
  assert.deepEqual(app.preferences, ["dark"]);
  assert.ok(app.classes.has("font-token"));
  assert.equal(app.animations[0].frames.clipPath[0], "circle(0px at 125px 235px)");
  assert.equal(app.animations[0].options.duration, 720);
  assert.equal(app.animations[0].options.easing, "cubic-bezier(.76, 0, .24, 1)");
  assert.equal(app.animations[0].options.pseudoElement, "::view-transition-new(root)");
  assert.equal(app.root.dataset.themeReveal, undefined);
  await app.toggle();
  assert.deepEqual(app.preferences, ["dark", "light"]);
  assert.equal(app.root.style.colorScheme, "light");
});

test("keyboard activation reveals from the button rather than the top-left corner", async () => {
  const app = browser();
  await app.toggle({ clientX: 0, clientY: 0, detail: 0 });
  assert.equal(app.animations[0].frames.clipPath[0], "circle(0px at 110px 210px)");
});

test("the synchronous provider hook retains system-theme support", () => {
  const app = browser({ osDark: true });
  app.useTheme().setTheme("system");
  assert.ok(app.classes.has("dark"));
  assert.equal(app.root.style.colorScheme, "dark");
  assert.deepEqual(app.preferences, ["system"]);
});

test("reduced motion and unsupported browsers still change and save the theme", async () => {
  for (const settings of [{ reduced: true }, { supported: false }]) {
    const app = browser(settings);
    await app.toggle();
    assert.deepEqual(app.preferences, ["dark"]);
    assert.ok(app.classes.has("dark"));
    assert.equal(app.animations.length, 0);
  }
});

test("a skipped transition doesn't toggle back or leave the reveal locked", async () => {
  const app = browser({ skipped: true });
  await app.toggle();
  assert.deepEqual(app.preferences, ["dark"]);
  assert.equal(app.root.dataset.themeReveal, undefined);
  assert.equal(app.animations.length, 0);
});

test("rapid clicks don't start competing snapshots", async () => {
  let finish;
  const animation = new Promise(resolve => { finish = resolve; });
  const app = browser({ animation });
  const first = app.toggle();
  await app.toggle();
  assert.deepEqual(app.preferences, ["dark"]);
  assert.equal(app.snapshots.length, 1);
  finish();
  await first;
  assert.equal(app.root.dataset.themeReveal, undefined);
});
