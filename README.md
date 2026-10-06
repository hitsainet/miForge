# XCC Framework

## Quick Setup: From Template to Your Project

### Prerequisites
- Git installed
- GitHub account
- **VS Code with Claude Code extension** (Primary development environment)
- Node.js (optional, for custom automation scripts)

> **Windows users:** The setup commands below (`rm -rf`, `mv`, `mkdir -p`, etc.) use bash syntax. Run them in **Git Bash** (installed alongside Git for Windows), or in PowerShell using the equivalents: `Remove-Item -Recurse -Force` for `rm -rf`, `Move-Item`/`Rename-Item` for `mv`, and `New-Item -ItemType Directory -Force` for `mkdir -p`.

---

## Features

### 🗂️ Organized Structure
All framework files are organized under the `0xcc/` directory:
- Separation between framework and project files
- Navigation with `0` prefix sorting to top of file explorer
- Portable framework structure

### 🏠 Project Memory
`AGENT.md` tracks project status and a Document Inventory of what's done and what's pending — enough to resume work in a new session. Claude Code's own conversation history and `/compact` handle everything else.

---

## Step-by-Step Setup

### Option A: Use the GitHub Template (recommended)

1. Go to [github.com/hitsainet/xcc_open](https://github.com/hitsainet/xcc_open)
2. Click **"Use this template"** → **"Create a new repository"**
3. Name it, choose Public or Private, and click **"Create repository"** — you get a fresh repo with no template history
4. Clone your new repository and enter it:
```bash
git clone https://github.com/yourusername/your-project-name.git
cd your-project-name
```

### Option B: Manual Clone (non-GitHub remotes or offline)

```bash
# 1. Clone the template and flush its history
git clone https://github.com/hitsainet/xcc_open.git your-project-name
cd your-project-name
rm -rf .git

# 2. Start a fresh repository
git init
git add .
git commit -m "Initial commit: XCC Framework with 0xcc organization"
git branch -M main

# 3. Connect to your remote and push
git remote add origin https://your-git-host/your-project-name.git
git push -u origin main
```

---

## Claude Code Integration

### 1. Open Project in VS Code
```bash
code .
```

### 2. Start Claude Code Session
- Open Command Palette: `Ctrl+Shift+P` (Windows/Linux) or `Cmd+Shift+P` (Mac)
- Type: **"Claude Code: Start Chat"**
- Or use the Claude Code icon in the Activity Bar

### 3. Initialize XCC Framework Context
```bash
# In Claude Code chat:
@AGENT.md

@0xcc/instruct/001_generate-brd.md

# Provide a free-form transcript of your project idea and requirements,
# plus any supporting research/documents. The framework will ask clarifying
# questions, then generate a BRD in 0xcc/prds/.

@0xcc/instruct/002_create-project-prd.md

# The framework will use the BRD as context and guide you through
# strategic clarifying questions to produce the Project PRD.
```

### 4. Resuming a Session
```bash
# Standard session start sequence
@AGENT.md
# Check the Document Inventory section for what's done and what's pending

# Load current work area based on phase
@0xcc/prds/     # For PRD work
@0xcc/tdds/     # For TDD work
@0xcc/tids/     # For TID work
@0xcc/tasks/    # For task execution
```

---

## Workflow Process

### Phase 0: Business Requirements (once per project)
```bash
# Right after cloning, before anything else
@0xcc/instruct/001_generate-brd.md
# Provide: free-form transcript + supporting research/documents
# Output: 0xcc/prds/000_BRD|[project-name].md
```

### Phase 1: Project Foundation
```bash
# Session 1: Project Vision
@0xcc/instruct/002_create-project-prd.md
@0xcc/prds/000_BRD|[project-name].md   # if it exists
# Output: 0xcc/prds/000_PPRD|[project-name].md

# Session 2: Technical Foundation
@0xcc/instruct/003_create-adr.md
@0xcc/prds/000_PPRD|[project-name].md
# Output: 0xcc/adrs/000_PADR|[project-name].md
# Action: Copy Project Standards section to AGENT.md

# Session 3: Foundation Backlog
@0xcc/instruct/007_generate-tasks.md
@0xcc/prds/000_PPRD|[project-name].md
@0xcc/adrs/000_PADR|[project-name].md
# Output: 0xcc/tasks/000_FTASKS|Project_Foundation.md
# (scaffolding, CI/CD, database, auth, tooling — work owned by no single feature)
```

### Phase 2: Feature Development (For each feature)
```bash
# Feature Requirements
@0xcc/instruct/004_create-feature-prd.md
@0xcc/prds/000_PPRD|[project-name].md
@0xcc/adrs/000_PADR|[project-name].md
# Output: 0xcc/prds/[###]_FPRD|[feature-name].md

# Technical Design
@0xcc/instruct/005_create-tdd.md
@0xcc/prds/[###]_FPRD|[feature-name].md
# Output: 0xcc/tdds/[###]_FTDD|[feature-name].md

# Implementation Planning
@0xcc/instruct/006_create-tid.md
@0xcc/prds/[###]_FPRD|[feature-name].md
@0xcc/tdds/[###]_FTDD|[feature-name].md
# Output: 0xcc/tids/[###]_FTID|[feature-name].md

# Task Generation
@0xcc/instruct/007_generate-tasks.md
@0xcc/prds/[###]_FPRD|[feature-name].md
@0xcc/tdds/[###]_FTDD|[feature-name].md
@0xcc/tids/[###]_FTID|[feature-name].md
# Output: 0xcc/tasks/[###]_FTASKS|[feature-name].md

# Implementation with Progress Tracking
@0xcc/instruct/008_process-task-list.md
@0xcc/tasks/[###]_FTASKS|[feature-name].md
# Execute tasks with progress tracking
```

---

## What You Get

### XCC Framework Project Structure
```
your-project-name/
├── 0xcc/                           # Core XCC Framework
│   ├── adrs/                       # Architecture Decision Records
│   │   └── 000_PADR|Project_Name.md
│   ├── docs/                       # Additional framework documentation
│   ├── instruct/                   # XCC Framework instruction files
│   │   ├── 000_README.md
│   │   ├── 001_generate-brd.md
│   │   ├── 002_create-project-prd.md
│   │   ├── 003_create-adr.md
│   │   ├── 004_create-feature-prd.md
│   │   ├── 005_create-tdd.md
│   │   ├── 006_create-tid.md
│   │   ├── 007_generate-tasks.md
│   │   └── 008_process-task-list.md
│   ├── prds/                       # Product Requirements Documents
│   │   ├── 000_BRD|Project_Name.md
│   │   ├── 000_PPRD|Project_Name.md
│   │   ├── 001_FPRD|Feature_A.md
│   │   └── 002_FPRD|Feature_B.md
│   ├── tasks/                      # Task Lists with progress tracking
│   │   ├── 000_FTASKS|Project_Foundation.md
│   │   ├── 001_FTASKS|Feature_A.md
│   │   └── 002_FTASKS|Feature_B.md
│   ├── tdds/                       # Technical Design Documents
│   │   ├── 001_FTDD|Feature_A.md
│   │   └── 002_FTDD|Feature_B.md
│   ├── tids/                       # Technical Implementation Documents
│   │   ├── 001_FTID|Feature_A.md
│   │   └── 002_FTID|Feature_B.md
│   └── scripts/                    # Optional automation scripts
├── .claude/                        # Claude Extensions System
│   ├── commands/                   # Command definitions
│   │   ├── analyze.md              # /analyze command
│   │   ├── collaborate.md          # /collaborate command
│   │   ├── feature.md              # /feature command
│   │   ├── health.md               # /health command
│   │   ├── review.md               # /review command
│   │   ├── smart-clear.md          # /smart-clear command
│   │   └── update_repo.md          # /update_repo command
│   ├── context/                    # Context management
│   │   ├── agents/                 # Expert agent contexts
│   │   │   ├── product_engineer.md
│   │   │   ├── qa_engineer.md
│   │   │   ├── architect.md
│   │   │   └── test_engineer.md
│   │   ├── health/                 # Health monitoring
│   │   │   ├── dashboard.md
│   │   │   └── assessment_template.md
│   │   └── sessions/               # Session tracking
│   │       └── session_template.md
│   ├── prompts/                    # Analysis templates
│   │   ├── analysis/               # Expert analysis prompts
│   │   │   ├── architecture/       # Architecture assessment
│   │   │   ├── integration/        # Integration review
│   │   │   ├── product/            # Product context analysis
│   │   │   ├── quality/            # Code quality review
│   │   │   └── testing/            # Testing strategy analysis
│   │   └── usage_guide.md          # Analysis templates guide
│   └── quick_reference.md          # Extensions usage guide
├── AGENT.md                        # Project memory system (canonical instructions)
├── CLAUDE.md                       # Stub → @AGENT.md (for Claude Code)
├── GEMINI.md                       # Stub → @AGENT.md (for Gemini)
├── src/                            # Your actual project code
├── tests/                          # Your project tests
├── package.json                    # Your project dependencies (if applicable)
└── README.md                       # This file
```

### Key Benefits

#### 🗂️ **Organization**
- Framework isolation: All XCC files in `0xcc/` directory
- Navigation: `0` prefix sorts framework to top in file explorers
- Clear boundaries: Framework vs project code separation
- Portable updates: Framework can be updated independently

#### 🏠 **Project Memory**
- `AGENT.md` tracks current phase, active document, and a Document Inventory of what's done and pending
- Resuming work is just loading `@AGENT.md` and checking that inventory

#### 📊 **Productivity Features**
- Decision consistency through documented standards
- Team collaboration through comprehensive project documentation

---

## Example Usage Patterns

### Resuming Work
```bash
# Claude Code chat
@AGENT.md

# Claude will:
# - Read the Document Inventory to see what's done and pending
# - Show progress and next actions
# - Present any blockers
```

---

## Additional Features

### Team Collaboration
- Context handoffs with complete decision history
- Knowledge preservation across team changes and project pauses
- Onboarding acceleration through comprehensive project documentation
- Decision transparency for stakeholders and team alignment

### Framework Portability
- Easy updates: Replace the `0xcc/instruct/` directory to update the framework (never replace `0xcc/` wholesale — it contains your project's PRDs, TDDs, TIDs, and task lists)
- Project templates: Copy the `0xcc/` folder structure to new projects
- Custom extensions: Add organization-specific instructions
- Version control: Track framework evolution separately from project code

---

## Troubleshooting

### If Context Gets Lost
```bash
# Reload the core project files
@AGENT.md
@0xcc/prds/000_PPRD|[project-name].md
@0xcc/adrs/000_PADR|[project-name].md

# Then ask: "Please help me understand where I am in the workflow"
```

### If You Need to Start a Phase Over
```bash
git log --oneline -10
# Ask: "Please help me restart the [current phase] with a clean approach"
```

---

## Migration from Old Structure

If you have an existing XCC project without the `0xcc/` structure, see
[`0xcc/docs/migration.md`](0xcc/docs/migration.md) for a migration script and
file-reference update guide.

---

**Setup Time:** ~5 minutes
**Features:** Organized structure, BRD-to-backlog workflow
**Ready to Code:** Start with the BRD (@0xcc/instruct/001_generate-brd.md)!

The XCC Framework provides an organized, automatically managed development experience within Claude Code and VS Code. The `0xcc/` structure ensures framework files are organized, updatable, and separated from project code.
