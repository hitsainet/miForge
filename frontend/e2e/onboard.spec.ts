/**
 * The seven-step authoring flow, walked end to end (F7 task 12.11; FR-007.17 - FR-007.20).
 *
 * All seven steps are authored in one pass, with the **dry run at step 4** and the **commit at
 * step 7** — which is what task 12.11 asks for and also the one path no unit test covers: each
 * panel has its own RTL spec, and a flow is exactly the thing a per-component test cannot see.
 *
 * **The API is stubbed at the network, not mocked in the app.** `vite preview` serves the built
 * bundle with no backend behind it (see `playwright.config.ts`), so the alternative to a network
 * stub is no flow test at all. Stubbing the transport keeps the real client, the real state
 * transitions and the real paint in the measurement, and it is the same rule the backend follows
 * for siblings: mock the HTTP surface, never the code (ADR-001, ADR-016).
 *
 * **The stub is stateful on purpose.** `OnboardPage` re-reads the draft after every commit —
 * *"the server is the truth and there is no optimistic validity"* — so a stub that returned a
 * fixed draft would let a screen that never advanced pass. The handler below advances
 * `current_step`, records which steps were PUT, and marks the earlier ones `done`, so the step
 * list's markers are the server's answer and the assertions are about the flow rather than about
 * a counter the page keeps.
 *
 * Three things are asserted that "it worked" would not be:
 *
 * - **the PUT sequence**, so a flow that skipped a step or committed one twice fails;
 * - **the commit's path and call count** — ADR-016 #1: a test that only checks "was called"
 *   passes against a call sending the wrong arguments, and `…/commit` against `…/steps/7` is
 *   exactly that mistake;
 * - **that going back to step 3 does not un-finish steps 4 to 6**, which is the property
 *   `StepList` exists for and the one a less-than comparison on the step number would break.
 */
import { expect, test, type Page, type Route } from '@playwright/test';

const DRAFT_ID = 'draft-e2e-1';
const STEP_COUNT = 7;

/** The seven titles, from `StepList.tsx`'s own table. */
const TITLES = [
  'Describe the domain',
  'Choose the prompts',
  'Choose the answers',
  'Choose how judgment works',
  'Choose what improvement means',
  'Choose what must not regress',
  'Review and save',
];

interface Calls {
  readonly steps: number[];
  readonly dryRuns: number;
  readonly commits: string[];
  readonly validityReads: number;
}

/**
 * Install the stub and return the call record.
 *
 * One catch-all handler dispatching on path and method, rather than a handler per route: with
 * several `page.route` patterns the one that answers is decided by registration order, and a
 * flow test whose stub depends on that is a test about Playwright.
 */
async function stubDraftFlow(page: Page): Promise<Calls> {
  const calls: Calls = { steps: [], dryRuns: 0, commits: [], validityReads: 0 };
  let current = 1;
  const reached = new Set<number>();

  const draft = () => ({
    draft_id: DRAFT_ID,
    name: reached.has(1) ? 'notary-md' : null,
    version: '1',
    domain: reached.has(1) ? 'notary' : null,
    current_step: current,
    step_count: STEP_COUNT,
    // The server's own markers. `done` for every step that has been PUT, whatever the
    // current cursor is - which is the behaviour `StepList` documents and the reason it does
    // not compare step numbers.
    step_states: Object.fromEntries(
      Array.from({ length: STEP_COUNT }, (_, index) => {
        const number = index + 1;
        const state = reached.has(number) ? 'done' : number === current ? 'current' : 'todo';
        return [String(number), state];
      }),
    ),
    slots: {
      prompt_set: reached.has(2)
        ? { contract_reference: { type: 'prompts', name: 'core', version: '3' }, external: null }
        : null,
      corpus: null,
      preference_pairs: null,
      selector_config: reached.has(4)
        ? {
            contract_reference: { type: 'selectors', name: 'judge-statute-cite', version: '2' },
            external: null,
          }
        : null,
      grader_config: null,
      test_set: reached.has(5)
        ? { contract_reference: { type: 'tests', name: 'holdout', version: '3' }, external: null }
        : null,
      retention_set: reached.has(6)
        ? {
            contract_reference: { type: 'tests', name: 'general', version: '1' },
            external: null,
            floor: 0.78,
          }
        : null,
    },
    created_at: '2026-10-14T00:00:00Z',
    updated_at: '2026-10-14T00:10:00Z',
    subtitle: `step ${current} of ${STEP_COUNT} · progress is saved as you go`,
  });

  const dryRun = () => ({
    provider: { name: 'in-process', model: 'LFM2-1.2B' },
    judge: { name: 'judge-statute-cite@2' },
    config_sha256: 'c'.repeat(64),
    config_kind: 'selector_config',
    hash_provenance: 'contract',
    health: {
      n: 5,
      accepted: 3,
      rejected: 1,
      undecided: 1,
      acceptance_rate: 0.6,
      reason_histogram: { cites_statute: 3, no_citation: 1, pending_review: 1 },
      grader_stats: null,
      signals: [],
    },
    verdicts: [
      {
        prompt: 'Who may witness a notarial act in Maryland?',
        completion: 'Any disinterested adult, per Md. Code § 18-214.',
        verdict: 'accept',
        reason_code: 'cites_statute',
        reason: 'names the controlling section',
        score: 0.91,
        admitted: true,
      },
      {
        prompt: 'How long must a notary keep a journal?',
        completion: 'Indefinitely.',
        verdict: 'reject',
        reason_code: 'no_citation',
        reason: 'no statute named',
        score: 0.12,
        admitted: false,
      },
    ],
    asynchronous_note: null,
    advisory: true,
  });

  await page.route('**/api/**', (route: Route) => {
    const request = route.request();
    const path = new URL(request.url()).pathname;
    const method = request.method();
    const json = (body: unknown, status = 200) =>
      route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(body) });

    if (path === '/api/health') {
      return json({ status: 'ok', dependencies: [], integrations: [] });
    }
    if (path === '/api/v1/pack-drafts' && method === 'POST') return json(draft(), 201);
    if (path === `/api/v1/pack-drafts/${DRAFT_ID}` && method === 'GET') return json(draft());

    const step = /^\/api\/v1\/pack-drafts\/[^/]+\/steps\/(\d+)$/.exec(path);
    if (step && method === 'PUT') {
      const number = Number(step[1]);
      calls.steps.push(number);
      reached.add(number);
      // The flow decides what follows a commit, and it does not go past the last step.
      current = Math.min(number + 1, STEP_COUNT);
      return json(draft());
    }
    if (path.endsWith('/dry-run') && method === 'POST') {
      calls.dryRuns += 1;
      return json(dryRun());
    }
    if (path.endsWith('/validity') && method === 'GET') {
      calls.validityReads += 1;
      return json({
        state: 'valid',
        missing_slots: [],
        unresolved: [],
        service_unavailable: [],
        retention_floor_missing: false,
      });
    }
    if (path.endsWith('/commit') && method === 'POST') {
      calls.commits.push(path);
      return json(
        {
          pack_id: 'notary-md',
          version: '1',
          name: 'notary-md',
          domain: 'notary',
          parts_filled: 4,
          parts_total: 7,
          prompt_count: 640,
          runs_using: [],
          validity: {
            state: 'valid',
            missing_slots: [],
            unresolved: [],
            service_unavailable: [],
            retention_floor_missing: false,
          },
          derived_from: null,
          is_example: false,
          definition_sha256: 'd'.repeat(64),
          created_at: '2026-10-14T00:20:00Z',
        },
        201,
      );
    }
    // The registry, which the screen lands on after a successful commit.
    if (path === '/api/v1/packs' && method === 'GET') return json({ packs: [] });

    // Anything else is a route this flow does not use. A 404 with the real envelope rather
    // than an empty 200: a stub that answered everything would hide a call nobody meant to
    // make.
    return json(
      { error: { code: 'not_stubbed', message: `${method} ${path} is not part of this flow` } },
      404,
    );
  });

  return calls;
}

async function openOnboard(page: Page): Promise<void> {
  await page.goto('/');
  await page
    .getByRole('navigation', { name: 'Screens' })
    .getByRole('button', { name: 'Onboard a domain' })
    .click();
  await expect(page.getByTestId('onboard-panel')).toBeVisible();
}

test('lists all seven steps with their titles', async ({ page }) => {
  await stubDraftFlow(page);
  await openOnboard(page);

  const list = page.getByTestId('step-list');
  for (const [index, title] of TITLES.entries()) {
    await expect(
      list.getByRole('button', { name: new RegExp(`${index + 1}\\. ${title}`) }),
    ).toBeVisible();
  }
  // Seven, not "at least seven": a flow that grew an eighth step without the backend's
  // `step_count` saying so would be a screen inventing a step.
  await expect(list.getByRole('listitem')).toHaveCount(STEP_COUNT);
  await expect(page.getByTestId('shell-page-head').locator('p')).toContainText(
    `step 1 of ${STEP_COUNT} · progress is saved as you go`,
  );
});

test('authors all seven steps, dry-runs at step 4, and commits at step 7', async ({ page }) => {
  const calls = await stubDraftFlow(page);
  await openOnboard(page);

  // Step 1 - the only step with free-text answers. `domain` is given rather than asserted
  // absent: null means "not domain-scoped", which is a different fact from unanswered.
  await expect(page.getByTestId('onboard-panel')).toContainText('1. Describe the domain');
  await page.getByTestId('draft-name').fill('notary-md');
  await page.getByTestId('draft-domain').fill('notary');
  await page.getByRole('button', { name: 'Save and continue' }).click();

  // Steps 2 and 3.
  await expect(page.getByTestId('onboard-panel')).toContainText('2. Choose the prompts');
  await page.getByRole('button', { name: 'Save and continue' }).click();
  await expect(page.getByTestId('onboard-panel')).toContainText('3. Choose the answers');
  await page.getByRole('button', { name: 'Save and continue' }).click();

  // Step 4 - THE DRY RUN. Advisory, and never an input to the pack's validity.
  await expect(page.getByTestId('onboard-panel')).toContainText('4. Choose how judgment works');
  await page.getByTestId('run-dry-run').click();
  await expect(page.getByTestId('dry-run-panel')).toBeVisible();
  // All three counts and their denominator, from F3's record - never a bare percentage.
  await expect(page.getByTestId('dry-run-counts')).toHaveText(
    'Accepted 3, rejected 1, undecided 1 of 5',
  );
  // A verdict arrives WITH ITS REASON, which is the whole content of AC11.
  await expect(page.getByTestId('dry-run-row-0')).toContainText('names the controlling section');
  await expect(page.getByTestId('dry-run-row-1')).toContainText('no statute named');
  await expect(page.getByTestId('dry-run-panel')).toContainText('never an input');
  expect(calls.dryRuns).toBe(1);
  await page.getByRole('button', { name: 'Save and continue' }).click();

  // Steps 5 and 6.
  await expect(page.getByTestId('onboard-panel')).toContainText('5. Choose what improvement means');
  await page.getByRole('button', { name: 'Save and continue' }).click();
  await expect(page.getByTestId('onboard-panel')).toContainText('6. Choose what must not regress');
  await page.getByTestId('draft-floor').fill('0.78');
  await page.getByRole('button', { name: 'Save and continue' }).click();

  // Step 7 - review, then THE COMMIT.
  await expect(page.getByTestId('onboard-panel')).toContainText('7. Review and save');
  await page.getByRole('button', { name: 'Check what is missing' }).click();
  await expect(page.getByTestId('onboard-validity')).toContainText('Valid');
  expect(calls.validityReads).toBe(1);

  await page.getByTestId('commit-draft').click();

  // The payload and the call count, not "was called": `…/steps/7` would also have been a
  // call, and it would have saved a step instead of committing a pack.
  expect(calls.commits).toEqual([`/api/v1/pack-drafts/${DRAFT_ID}/commit`]);
  expect(calls.steps).toEqual([1, 2, 3, 4, 5, 6]);

  // A successful commit lands on the registry, which is how an author sees the pack exist.
  await expect(page.getByRole('heading', { level: 1 })).toHaveText('Domain packs');
});

test('going back to step 3 leaves steps 4 to 6 done', async ({ page }) => {
  const calls = await stubDraftFlow(page);
  await openOnboard(page);

  for (let step = 1; step <= 6; step += 1) {
    await page.getByRole('button', { name: 'Save and continue' }).click();
  }
  expect(calls.steps).toEqual([1, 2, 3, 4, 5, 6]);

  await page
    .getByTestId('step-list')
    .getByRole('button', { name: /3\. Choose the answers/ })
    .click();
  await expect(page.getByTestId('onboard-panel')).toContainText('3. Choose the answers');

  // The markers are the server's `step_states`. A screen that derived them from "is this
  // number below the cursor" would show 4, 5 and 6 as todo and make finished work look lost.
  for (const step of [4, 5, 6]) {
    await expect(
      page.getByTestId('step-list').locator(`button[data-step="${step}"]`),
    ).toHaveAttribute('data-step-state', 'done');
  }
  // And nothing was re-committed by going back.
  expect(calls.steps).toEqual([1, 2, 3, 4, 5, 6]);
});

test('says a draft is not a pack, and keeps it out of the registry', async ({ page }) => {
  await stubDraftFlow(page);
  await openOnboard(page);

  await expect(page.getByTestId('shell-content')).toContainText('A draft is not a pack');
  await expect(page.getByTestId('shell-content')).toContainText(
    'Later steps are not discarded when you go back to an earlier one.',
  );
});
