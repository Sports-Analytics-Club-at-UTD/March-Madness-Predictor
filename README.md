This is a [Next.js](https://nextjs.org) project bootstrapped with [`create-next-app`](https://nextjs.org/docs/app/api-reference/cli/create-next-app).

## Getting Started

Install dependencies and configure your Convex development deployment:

```bash
npm install
npm run convex:dev
```

On the first run, Convex prompts you to log in and select or create a project.
It writes deployment settings, including `NEXT_PUBLIC_CONVEX_URL`, to `.env.local`
and generates typed helpers in `convex/_generated/`. Keep this command running
to sync backend changes. Add tables to `convex/schema.ts` and backend functions
under `convex/`.

In a second terminal, run the Next.js development server:

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

You can start editing the homepage in `src/app/(Home)/page.tsx`. The page auto-updates as you edit the file.

The root layout provides Convex to client components through
`src/app/ConvexClientProvider.tsx`. Use `useQuery` and `useMutation` from
`convex/react` with the generated API to access your backend.

For production, run `npm run convex:deploy` and set `NEXT_PUBLIC_CONVEX_URL` in
your hosting environment to the production deployment URL before building.

This project uses [`next/font`](https://nextjs.org/docs/app/building-your-application/optimizing/fonts) to automatically optimize and load [Geist](https://vercel.com/font), a new font family for Vercel.

## Tournament bracket

`/projects/mmp` loads the selected prediction bracket from Convex. To populate
a new development deployment, run:

```bash
npx convex dev --once --run tournaments:seed2026
npx convex dev --once
node scripts/seed-bracket-cards.mjs
```

The seed inserts the example only if its slug is absent; rerunning it preserves
existing data. It includes all 68 teams and 67 completed games, with results from
the [official NCAA bracket](https://www.ncaa.com/march-madness-live/bracket).
The card seed uploads `public/next-favicon.ico` to Convex file storage and adds
The UT Upheaval, A&M Avenue, and The Real Deal. The first two are illustrative
predictions that advance Texas or Texas A&M to the championship; they are not
model output. The Real Deal contains historical results. Rerunning the script
reuses the uploaded logo and existing cards.

`brackets` stores each card's title, description, storage logo ID, and tournament
reference. `PredictionBracketCard` accepts `imageUrl`, `title`, `bracketId`, and
`onSelect` props. The sidebar loads the selected card's tournament in the main
canvas and refits it when switching cards.

`TournamentBracket` accepts a `tournament` prop with teams, game results, and
references to the games that feed each matchup. It generates locked cards and
connections in mirrored halves meeting at the championship; pan, zoom,
fit-to-content, and theme controls remain available.
Scores are optional and hidden by default. Model predictions need only a
`winnerTeamId` for each game. Pass `showScores` to `TournamentBracket` or
`TournamentBracketCard` to display scores when supplied.
`convex/tournamentModel.ts` defines the data shape and validates advancement,
scores, and the champion. The seed is internal; the browser has read access only.

Run the bracket checks with:

```bash
node --test tests/tournament-bracket.test.cjs
```

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.
