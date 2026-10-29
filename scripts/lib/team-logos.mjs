/** @param {string[] | null} logos */
export function selectLargeLogo(logos) {
  return logos?.find(logo => {
    const url = new URL(logo);
    return url.origin === "https://cdn.collegefootballdata.com" && url.pathname.startsWith("/cbb-logos/500/");
  }) ?? null;
}
