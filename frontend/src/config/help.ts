/**
 * The Help topics (ENH-004, 2026-10-20): short summaries, each linking to its manual section.
 *
 * Origin: miDataworks (Onegaishimas/miDataworks) `frontend/src/config/help.ts` @ 32cf406.
 * Mode: adapt (`docs/REUSE.md`). Kept: `HelpTopic {id, title, body, steps?, manual}`, a flat
 * ordered list, `HELP_TOPIC_IDS`, `helpTopic()`, a per-step map. Changed: miForge's topics; a
 * per-SCREEN map, so the page head links every screen's topic from one place; the topic is a
 * URL parameter (`#/help/<topic>`), not a store.
 *
 * **Help is a summary; the manual is the reference.** Nothing here is generated from or copied
 * into the manual. `manual` is `page#heading-slug` relative to the manual's base, and
 * `test/manualLinks.test.tsx` checks every page and anchor against the manual's own sources.
 *
 * **Plain words** (`test/plainLanguage.test.ts`, ENH-004 D8): no internal identifiers - the
 * manual's own publish rules are applied - and every acronym spelled out in parentheses at its
 * first use in a topic.
 */
import type { ScreenId } from '@/contexts/NavigationContext';

export interface HelpTopic {
  /** The URL's `<topic>` in `#/help/<topic>`, and the card's `help-<id>` anchor. */
  readonly id: string;
  readonly title: string;
  /** Paragraphs. */
  readonly body: readonly string[];
  /** An ordered list, when the topic is a procedure. */
  readonly steps?: readonly string[];
  /** `page#heading-slug`, relative to the manual's base URL. */
  readonly manual: string;
}

export const HELP_TOPICS: readonly HelpTopic[] = [
  {
    id: 'getting-started',
    title: 'Getting started',
    body: [
      'miForge changes the weights of a language model, a little at a time, toward a domain you describe. Every saved step is a model you can load, compare or branch from.',
      'It works on its own. miServer and miDataworks are optional: when one is not set up, the features that need it say so by name.',
    ],
    steps: [
      'Make a model and some data available: a model folder or a Hugging Face repository, and parts (prompts, answers or text).',
      'Describe your domain as a pack in Onboard a domain.',
      'Define and start a run in New run, with at least one stop condition.',
      'Watch the run: its loss, its test and retention scores, its saves and its log.',
      'Find the results in Checkpoints, where any saved step can be continued or branched.',
    ],
    manual: 'getting-started/first-run',
  },
  {
    id: 'key-ideas',
    title: 'Key ideas',
    body: [
      'A step is one update to the weights. A checkpoint is the model as it was after a step, saved as an ordinary model folder.',
      'A run is one job that takes steps until a stop condition fires. A pack describes a domain: its prompts, its answers, how answers are judged and what must not get worse.',
      'Test scores say whether the model got better at the domain; retention scores say whether it kept what it already knew. The floor is how far retention may fall before the run stops.',
    ],
    manual: 'getting-started/concepts',
  },
  {
    id: 'runs',
    title: 'Runs',
    body: [
      'Every run, newest first, with its status, type and last step. Filter by status or type, open one to watch it, and archive finished runs you no longer want in the list.',
      'An archived run is hidden, not deleted. Show archived runs to see them again, or to remove one for good.',
    ],
    manual: 'guides/runs#the-screens',
  },
  {
    id: 'run',
    title: 'A run',
    body: [
      'One run as it happens: the loss curve, the test and retention scores at each evaluation, the saves and the run log.',
      'Liveness comes from the database, not from the live connection: the screen shows how long ago the run last reported and the threshold after which it would be treated as lost.',
      'A stop, asked for through the application programming interface (API) or by an agent, takes effect at the next step boundary and is followed by a save; the screen shows when it was asked for. A run that ended can be continued from its last save.',
    ],
    manual: 'guides/scoring#reading-scores',
  },
  {
    id: 'new-run',
    title: 'New run',
    body: [
      'One definition holds everything a run needs: the model, the pack, where the training signal comes from, the configuration, how often to test, check retention and save, and at least one stop condition.',
      'Check it first: the check reports every problem at once and estimates whether the run fits on the chosen graphics card. Nothing starts until you start it.',
    ],
    manual: 'guides/runs#the-run-definition',
  },
  {
    id: 'checkpoints',
    title: 'Checkpoints',
    body: [
      'Every saved step of every run, as a tree: a branch shows which checkpoint a run started from.',
      'Continue a run from its latest save, or branch a new run from any checkpoint. Each checkpoint is a standard model folder you can also load anywhere else.',
    ],
    manual: 'guides/checkpoints#the-checkpoints-screen',
  },
  {
    id: 'packs',
    title: 'Domain packs',
    body: [
      'A pack is everything miForge needs to know about a domain: the prompts, the answers or text to learn from, how an answer is judged, what counts as better and what must not regress.',
      'Packs are versioned. A run records the exact version it used, so a result can always be traced back to its pack.',
    ],
    manual: 'guides/packs#the-domain-packs-screen',
  },
  {
    id: 'onboard',
    title: 'Onboard a domain',
    body: [
      'Seven steps that build a pack. Progress is saved as you go, a draft is never a pack until you save it, and going back to an earlier step does not discard the later ones.',
    ],
    steps: [
      'Describe the domain in a sentence or two.',
      'Choose the prompts.',
      'Choose the answers or text to learn from.',
      'Choose how an answer is judged.',
      'Choose what improvement means.',
      'Choose what must not regress.',
      'Review and save the pack.',
    ],
    manual: 'guides/packs#authoring-a-pack-onboard-a-domain',
  },
  {
    id: 'judging',
    title: 'Judging and improvement',
    body: [
      'A selector decides which answers are good enough to learn from; a grader scores an answer. Rules, a reference answer, a judge model or a human reviewer can each do it.',
      'The model being trained never judges its own answers, and an answer nobody could decide on stays undecided rather than being counted as good or bad.',
    ],
    manual: 'guides/signals-and-judging#selectors-and-graders',
  },
  {
    id: 'regression',
    title: 'What must not regress',
    body: [
      'Retention is a held-out set of things the model already did well. It is scored during the run, and the floor is how far that score may fall before the run stops itself.',
    ],
    manual: 'guides/scoring#test-and-retention',
  },
  {
    id: 'approvals',
    title: 'Approvals',
    // ENH-008 §2.6: the same four points in all four suite apps.
    body: [
      'When an artificial intelligence (AI) agent asks miForge to do something that needs you (start work on a graphics card, attach to a model miServer is serving, delete something or change a setting), the request waits. The Approvals menu shows the number, and a banner shows it on every screen.',
      'Approve carries it out once, exactly as shown. Reject means it never runs. A request you do not decide expires after 24 hours. When two or more settings changes wait, Approve all settings changes approves the ones it lists in one step; each is still carried out and recorded on its own.',
      'Each app decides its own requests: a request to miForge is decided here, in miForge.',
      'History keeps every decision: who decided, when, the reason and what happened.',
    ],
    manual: 'operators/approvals#the-approvals-screen',
  },
  {
    id: 'wsp-runs',
    title: 'Weight-space programming runs',
    body: [
      'A weight-space programming (WSP) run gives an editor model a probe’s weights and a task; it reads them, changes them through tools and submits a candidate. An agent asks for one and it waits in Approvals until you approve it.',
      'WSP runs lists every one, newest first: its state, why it stopped, what it used and when it ended. Open one to see what it produced, its verdict when one is stored, and the editor’s whole conversation, shown as text.',
      'Both screens keep themselves current: every 5 seconds while a run is live, and the run’s screen stops once it ends. A live run shows its elapsed time, its editor calls and tokens, the worker’s last heartbeat and each budget spent; its Editor calls list shows each call’s outcome, status, time and tokens, never its text. A request still waiting in Approvals is listed above the runs.',
      'A budget the request leaves out takes a default from the editor’s endpoint. For a local editor the defaults are safety stops that catch a runaway: 5,000,000 tokens, 100 steps and 2 hours. For an external one they are 100,000 tokens, 50 steps and 1 hour, plus the cost cap. Forward passes are 0 unless the request names them. A budget the request names always wins.',
    ],
    manual: 'guides/runs#weight-space-programming-runs',
  },
  {
    id: 'wsp-endpoints',
    title: 'Editor endpoints',
    body: [
      'An editor endpoint is a model server a weight-space programming session uses as its editor: miServer in the cluster, another local server, or an external provider. A session names one by its id.',
      'Add one with every field the registration takes. miForge works out whether the host is local or external; an external endpoint needs https, an application programming interface (API) key and input and output prices. The key is sent once, sealed for the gateway, and never shown again: the list shows only whether a key is sealed and its fingerprint.',
      'Change lets you set the action protocol, the context window, thinking, the output cap and paging; any other field is bound to the key, so you remove the endpoint and register it again. Removing one stops any run that names it from starting. Run protocol test queues a short test run and links it.',
      'Thinking is off by default: the engine asks the editor not to reason at length. On, which needs an OpenAI-compatible miServer endpoint, gives each turn a reasoning allowance (4,096 tokens unless you set one) beside its action; the reasoning is kept in the run’s transcript and never run. A turn with too little context left for a useful thought is sent with thinking off, and the transcript says so. Default sends nothing and leaves it to the server. A run can override it, and the run’s screen shows the value it used.',
      'Output cap is on by default: each request names a per-turn limit on the reply. Off names none, so a reply can use the room the context window has left; the run’s budgets still apply.',
      'Paging is on by default, for the suite’s own engine (miwsp_engine) only. A bulky tool result, such as a read of a whole weight, stays in the conversation for three turns and is then replaced by a short note naming its page, which the editor can recall; the engine also keeps a ledger of every measurement and edit, built from the tool results, at the end of the conversation. Off keeps every result in the conversation, as before. Paging switches itself off for an editor whose context window is under 16,384 tokens, and the run says so. A run can override it, and the run’s screen shows the value it used.',
      'An agent can propose a local endpoint with no key and no prices. It waits in Approvals, showing the id, the model, the server’s uniform resource locator (URL) and the protocol, and appears here once you approve it. Keys, prices, external endpoints, changes and removals are yours alone.',
    ],
    manual: 'guides/runs#editor-endpoints',
  },
  {
    id: 'settings',
    title: 'Settings',
    body: [
      'Where each setting comes from, what it does and when a change takes effect. A few can be changed while miForge runs; the rest come from the environment and need a restart of the pod named on them.',
      'Every change is recorded with who made it and an optional reason. Recent changes lists them, newest first.',
    ],
    manual: 'operators/settings#the-settings-screen',
  },
  {
    id: 'hot-update',
    title: 'Changing weights while miServer serves them',
    body: [
      'Hot update is optional. A run attaches to the model miServer is serving and writes each step into it in place, so the served model changes as it trains.',
      'Before the first step, both sides score the same prompt to prove miForge rebuilt the served model exactly. Each update is bracketed so a reader never sees half-written weights.',
    ],
    manual: 'operators/hot-update',
  },
  {
    id: 'glossary',
    title: 'Words used here',
    body: [
      'Attach: start a run on the weights miServer is serving. Branch: start a new run from any checkpoint. Floor: the lowest retention score a run may reach before it stops.',
      'Part: one set of prompts, answers or text a pack uses. Lease: a run’s hold on a graphics card. Override: a setting changed while miForge runs, replacing the environment’s value until it is cleared.',
      'The manual’s glossary defines every word miForge uses.',
    ],
    manual: 'glossary',
  },
];

export const HELP_TOPIC_IDS: readonly string[] = HELP_TOPICS.map((topic) => topic.id);

export function helpTopic(id: string | undefined): HelpTopic | undefined {
  return id === undefined ? undefined : HELP_TOPICS.find((topic) => topic.id === id);
}

/**
 * Each screen's topic: the page head links it (ENH-004 D4). `null` - the Help screen itself.
 * Typed over every `ScreenId`, so a new screen does not compile until it names its topic.
 */
export const HELP_FOR_SCREEN: Readonly<Record<ScreenId, string | null>> = {
  runs: 'runs',
  run: 'run',
  'new-run': 'new-run',
  checkpoints: 'checkpoints',
  packs: 'packs',
  onboard: 'onboard',
  approvals: 'approvals',
  'wsp-runs': 'wsp-runs',
  'wsp-run': 'wsp-runs',
  'wsp-endpoints': 'wsp-endpoints',
  settings: 'settings',
  help: null,
};

/** Onboard a domain's per-step "Help with this step" (steps 4-6 have their own pages). */
export const HELP_FOR_ONBOARD_STEP: Readonly<Record<number, string>> = {
  1: 'onboard',
  2: 'onboard',
  3: 'onboard',
  4: 'judging',
  5: 'judging',
  6: 'regression',
  7: 'onboard',
};
