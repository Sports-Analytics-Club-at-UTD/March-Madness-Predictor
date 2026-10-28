# March Madness Predictor

Vite + React frontend using React Router in declarative mode, Convex, Tailwind CSS, and shadcn/ui. Uses pnpm.

## Development

```bash
pnpm install
pnpm dev:all
```

`dev:all` syncs Convex functions and starts Vite at http://localhost:5173.
You can also run `pnpm dev:convex` and `pnpm dev` in separate terminals.

The first Convex run prompts you to select or create a project and writes
`CONVEX_DEPLOYMENT`, `VITE_CONVEX_URL`, and `VITE_CONVEX_SITE_URL`
to `.env.local`. This file is ignored by Git. This checkout is configured with
a cloud development deployment; other developers should run `pnpm dev:convex`
to configure their own deployment.

## Backend

Add tables to `convex/schema.ts` and query/mutation/action functions in `convex/`.
Convex generates typed helpers in `convex/_generated/`; commit these generated files.
The root render includes `ConvexProvider`, so components can use `useQuery` and
`useMutation` from `convex/react` with the generated API from
`convex/_generated/api`.

The schema starts empty. No sample data or public write functions are installed.

- [Convex dashboard](https://dashboard.convex.dev/t/abel-sam/march-madness-predictor-69042)
- [Convex React guide](https://docs.convex.dev/client/react/quickstart-create-react-app)

## Checks

```bash
pnpm lint
pnpm typecheck
pnpm build
```

## Production

Run `pnpm exec convex deploy` to deploy the backend. Configure the frontend host's
`VITE_CONVEX_URL` with the production deployment URL before building Vite.
Development and production deployments have separate data. Keep deployment keys
in the hosting provider's secret settings.
