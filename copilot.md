# Copilot.md — Coding Standards & Best Practices

> Written from a Software Engineering perspective.
> This file defines how code should be written, structured, and maintained across any project.
> GitHub Copilot is the AI assistant used in this project. Model selection is handled manually per task.

---

## Core Philosophy

- Clarity over cleverness — code is read far more than it is written. Optimize for the next person reading it (including future you).
- Simple and explainable — if you cannot explain what a block of code does in one sentence, it needs to be simplified.
- Modular by default — every piece of logic should have one job, one place, and be independently testable.
- Consistency is king — a consistent codebase is easier to navigate, debug, and extend than a clever one.
- Explicit over implicit — never assume. Make the code say exactly what it does.

---

## GitHub Copilot Usage Guidelines

Copilot is a tool to assist, not to replace engineering judgment. Treat every Copilot suggestion as a draft that needs review.

```
Model selection guidance:
- Use a reasoning/advanced model for architecture decisions, complex logic, refactoring, and debugging hard problems.
- Use a faster model for boilerplate generation, simple completions, and repetitive patterns.
- Never accept a suggestion without reading it. Copilot does not know your domain, your constraints, or your intent.
```

Rules when working with Copilot:

```
- Always review generated code line by line before accepting it.
- Copilot suggestions must follow every rule in this file — they are not exempt.
- If Copilot generates something clever but hard to read, simplify it before using it.
- Do not let Copilot name your variables, functions, or files without verifying they follow naming conventions.
- Write your comment or function signature first — this guides Copilot toward better completions.
- Use Copilot for acceleration, not for understanding. If you do not understand what was generated, do not use it.
```

---

## Project Structure

```
project/
├── src/               # All source code lives here
│   ├── modules/       # Self-contained feature modules
│   ├── utils/         # Shared helper functions (pure, reusable)
│   ├── services/      # External integrations (API, DB, third-party)
│   ├── config/        # App-wide configuration and constants
│   ├── middleware/    # Request/response pipeline handlers
│   ├── models/        # Data models and schema definitions
│   └── types/         # Shared type definitions and interfaces
├── tests/             # Mirror of src/ structure for test files
├── docs/              # Documentation and architectural notes
├── scripts/           # One-off or automation scripts (not app logic)
├── .env.example       # Template for environment variables (never commit .env)
├── CLAUDE.md          # This file
└── README.md          # Project overview and setup guide
```

> Rule: Every folder should be self-explanatory. A new developer should understand the project layout within 5 minutes.

---

## Naming Conventions

```js
// Variables: camelCase, descriptive nouns — never abbreviations
const userAccountBalance = 0;            // correct
const uab = 0;                           // wrong — cryptic, meaningless outside your head

// Functions: camelCase, start with a verb — describe what they DO
function calculateTotalPrice() {}        // correct — verb + noun
function total() {}                      // wrong — too vague

// Classes: PascalCase, singular noun
class InvoiceGenerator {}                // correct
class invoices {}                        // wrong

// Constants: SCREAMING_SNAKE_CASE
const MAX_RETRY_ATTEMPTS = 3;            // correct
const maxRetry = 3;                      // wrong — looks like a mutable variable

// Files: kebab-case for utilities, PascalCase for components/classes
// user-profile.js      correct
// UserProfile.jsx      correct for React/component files
// userprofile.js       wrong

// Booleans: always prefix with is / has / can / should / was
const isLoading = true;                  // correct
const loading = true;                    // wrong — ambiguous, could be anything

// Arrays: always plural
const userList = [];                     // correct
const user = [];                         // wrong — implies a single item
```

---

## Comments — Comment the WHY, Not the WHAT

> Code shows what is happening. Comments explain why it was done this way.

```js
// Correct — explains the reasoning behind the decision
// Cap retries at 3 to avoid overwhelming the upstream service during outages
const MAX_RETRY_ATTEMPTS = 3;

// Wrong — just restates the code, adds zero value
// Set max retry to 3
const MAX_RETRY_ATTEMPTS = 3;
```

### Function Documentation

```js
// Every function gets a JSDoc block explaining its purpose, parameters, and return value
/**
 * Calculates the final price after applying a discount percentage.
 * @param {number} basePrice  - The original price before discount
 * @param {number} discount   - Discount as a percentage between 0 and 100
 * @returns {number}          - Final price rounded to 2 decimal places
 */
function applyDiscount(basePrice, discount) {
  // Reject invalid discount ranges before doing any math
  if (discount < 0 || discount > 100) throw new Error("Discount must be between 0 and 100");

  // Subtract the calculated discount amount from the original price
  const discountAmount = (basePrice * discount) / 100;
  return parseFloat((basePrice - discountAmount).toFixed(2));
}
```

### Inline Comment Rules

```js
// Comment non-obvious logic — do not narrate obvious steps
function parseUserToken(token) {
  // JWT is structured as header.payload.signature — extract the middle section
  const parts = token.split(".");

  // Payload is Base64-encoded; decode it to get the raw JSON string
  const decoded = atob(parts[1]);

  return JSON.parse(decoded);
}

// Use markers for known issues and decisions
// TODO:   something that needs to be done later
// FIXME:  something broken that needs a fix
// HACK:   a workaround that should be replaced — explain why it exists
// NOTE:   important context that future developers must know
// REVIEW: this section needs a second pair of eyes before shipping
```

### What Not to Do in Comments

```js
// Do not leave commented-out dead code
// const oldFunction = () => { ... }     // wrong — delete it, Git history exists

// Do not write separator lines anywhere in the codebase
// ============================================================   wrong
// -----------------------------------------------------------    wrong
// ************************************************************   wrong

// Do not use print-style debug separators
console.log("=".repeat(60));    // wrong — never do this
console.log("--- START ---");   // wrong
console.log("**********");      // wrong
```

---

## Logging — Use Sparingly and with Purpose

Logs exist to help diagnose real problems in production and development. They are not a debugging scratchpad and not a way to trace execution flow during development.

```js
// Wrong — noise that pollutes output and gets accidentally left in
console.log("here");
console.log("data", data);
console.log("=".repeat(60));
console.log("--- entering function ---");
console.log(user);
console.log("step 1");

// Correct — a structured logger used consistently across the project
import logger from "@/utils/logger";

// Log only what is necessary to diagnose a real failure
logger.error("Payment processing failed", { orderId, reason: error.message });

// Log meaningful state transitions at the appropriate level
logger.info("User session started", { userId });
logger.warn("Rate limit approaching threshold", { userId, requestCount });
```

### Logging Levels — Use the Right One

```
logger.error()   — something broke and needs immediate attention
logger.warn()    — something unusual happened but the system kept running
logger.info()    — meaningful system event (startup, shutdown, key state transitions)
logger.debug()   — detailed context for diagnosing specific issues (disabled in production)
```

### Logging Rules

```
- Never log inside a loop — log before or after with a summary count or result
- Never log sensitive data — passwords, tokens, credit cards, personal identifiers
- Always include context — log WITH relevant IDs and metadata, not just a message string
- Remove all temporary debug logs before finalizing any code
- Do not leave console.log anywhere in production-bound code
- One log per error occurrence — do not log the same error at multiple levels
- If you are logging more than 2-3 times in a single function, you are logging too much
```

---

## Modularity — One Thing, One Place

Every function, class, or module does one thing and does it well (Single Responsibility Principle).

```js
// Wrong — one function carrying too many responsibilities
function handleUser(user) {
  if (!user.email) throw new Error("Missing email");
  user.name = user.name.trim();
  db.save(user);
  emailService.sendWelcome(user.email);
}

// Correct — each function has exactly one job
function validateUser(user) {
  // Ensure required fields are present before any processing occurs
  if (!user.email) throw new Error("Missing email");
  if (!user.name) throw new Error("Missing name");
}

function formatUser(user) {
  // Normalize fields to remove whitespace and enforce lowercase email
  return { ...user, name: user.name.trim(), email: user.email.toLowerCase() };
}

function saveUser(user) {
  // Persist the validated and formatted user record to the database
  return db.save(user);
}

function notifyUser(email) {
  // Trigger the welcome email after a successful registration
  return emailService.sendWelcome(email);
}

// Orchestrate the individual steps in a clear, sequential flow
async function registerUser(rawUser) {
  validateUser(rawUser);             // Step 1: reject bad input early
  const user = formatUser(rawUser);  // Step 2: normalize data
  await saveUser(user);              // Step 3: persist to database
  await notifyUser(user.email);      // Step 4: send confirmation email
}
```

---

## Functions — Rules to Follow

```js
// Rule 1: Functions should be short — aim for under 20-30 lines
// If a function grows beyond that, it is doing too much. Extract sub-functions.

// Rule 2: Maximum 3 parameters — use an options object for anything more complex
// Wrong — hard to read at the call site, easy to mix up argument order
function createUser(name, email, age, role, isAdmin, isVerified) {}

// Correct — self-documenting at the call site
function createUser({ name, email, age, role, isAdmin, isVerified }) {}
createUser({ name: "Alice", role: "admin", isAdmin: true, isVerified: false });

// Rule 3: Return predictable values — avoid hidden side effects
// A function that returns a computed value is always easier to test than one that mutates state

// Rule 4: Guard clauses — exit early to avoid deep nesting
// Wrong — deeply nested, hard to trace the logic
function getDiscount(user) {
  if (user) {
    if (user.isPremium) {
      if (user.yearsActive > 2) {
        return 20;
      }
    }
  }
  return 0;
}

// Correct — guard clauses make each condition explicit and the happy path obvious
function getDiscount(user) {
  if (!user) return 0;                 // No user, no discount
  if (!user.isPremium) return 0;       // Non-premium users are not eligible
  if (user.yearsActive <= 2) return 0; // Minimum loyalty period not reached
  return 20;                           // Loyal premium user qualifies for 20% discount
}

// Rule 5: Avoid boolean parameters — they are a sign the function does two different things
// Wrong — caller has no idea what true or false means here
function fetchUser(id, includeDeleted) {}

// Correct — two explicit functions with clear intent
function fetchActiveUser(id) {}
function fetchDeletedUser(id) {}

// Rule 6: Pure functions where possible — same input always produces same output
// Easier to test, easier to reason about, no hidden state dependencies
function formatCurrency(amount, currency) {
  // Format a number as a localized currency string without touching any external state
  return new Intl.NumberFormat("en-US", { style: "currency", currency }).format(amount);
}
```

---

## Error Handling

```js
// Never silently swallow errors
// Wrong — hides failures and makes debugging impossible
try {
  riskyOperation();
} catch (e) {}

// Correct — handle with intention, log context, rethrow or recover explicitly
try {
  await riskyOperation();
} catch (error) {
  // Log with enough context to reproduce or diagnose the issue
  logger.error("riskyOperation failed", { reason: error.message });
  throw new Error(`Operation failed: ${error.message}`);
}

// Use custom error classes for domain-specific failures
class ValidationError extends Error {
  constructor(message, field) {
    super(message);
    this.name = "ValidationError"; // Identifies the error type in logs and catch blocks
    this.field = field;            // Indicates which input field triggered the failure
  }
}

class NotFoundError extends Error {
  constructor(resource, id) {
    super(`${resource} with id ${id} was not found`);
    this.name = "NotFoundError";
    this.statusCode = 404;         // Carries HTTP context so handlers can respond correctly
  }
}

// Validate inputs at the function boundary — fail fast before touching external systems
function transferFunds(amount, fromAccount, toAccount) {
  if (amount <= 0) throw new ValidationError("Amount must be positive", "amount");
  if (!fromAccount) throw new ValidationError("Source account is required", "fromAccount");
  if (!toAccount) throw new ValidationError("Target account is required", "toAccount");
  if (fromAccount === toAccount) throw new Error("Source and target accounts must differ");
  // All inputs confirmed valid — proceed with transfer logic
}

// Async error handling — always await inside try/catch, never let promises float unhandled
async function loadDashboard(userId) {
  try {
    const user = await fetchUser(userId);
    const stats = await fetchUserStats(userId);
    return buildDashboard(user, stats);
  } catch (error) {
    logger.error("Dashboard load failed", { userId, reason: error.message });
    throw error; // Let the caller decide how to surface this to the user
  }
}
```

---

## Data & State Management

```js
// Immutability — never mutate input arguments
// Wrong — modifies the original object, causes hidden bugs in callers
function applyTax(order) {
  order.total = order.subtotal * 1.2;
  return order;
}

// Correct — return a new object, the original stays untouched
function applyTax(order) {
  // Compute the tax-inclusive total without modifying the original order
  return { ...order, total: order.subtotal * 1.2 };
}

// Avoid global mutable state — it creates invisible coupling between modules
// Wrong
let currentUser = null;          // any module can change this at any time

// Correct — pass state explicitly through function arguments or a scoped store
function renderProfile(user) {   // the dependency is visible and deliberate
  return buildProfileView(user);
}

// Constants belong in config, not scattered across files
// Wrong — magic string buried inside a function
function canAccessAdminPanel(user) {
  return user.role === "super_admin";
}

// Correct — named constant in config, imported where needed
const ROLES = {
  ADMIN: "admin",
  SUPER_ADMIN: "super_admin",
  VIEWER: "viewer",
};

function canAccessAdminPanel(user) {
  // Only super admins can access the admin panel
  return user.role === ROLES.SUPER_ADMIN;
}
```

---

## Async & Concurrency

```js
// Always use async/await — avoid raw .then() chains which are harder to follow
// Wrong — nested .then() chains degrade readability quickly
fetchUser(id)
  .then(user => fetchOrders(user.id))
  .then(orders => processOrders(orders))
  .catch(err => logger.error("Failed", { err }));

// Correct — sequential, readable, and easy to add step-specific error handling
async function loadUserOrders(id) {
  const user = await fetchUser(id);
  const orders = await fetchOrders(user.id);
  return processOrders(orders);
}

// Run independent async operations in parallel — do not serialize what can run concurrently
// Wrong — user and settings are independent, but this waits for one before starting the other
const user = await fetchUser(id);
const settings = await fetchSettings(id);

// Correct — both requests run at the same time
const [user, settings] = await Promise.all([fetchUser(id), fetchSettings(id)]);

// Handle partial failures in parallel calls explicitly
const results = await Promise.allSettled([fetchUser(id), fetchSettings(id)]);
results.forEach(result => {
  if (result.status === "rejected") {
    // Log the partial failure but continue — do not let one failure block the rest
    logger.warn("Parallel fetch partially failed", { reason: result.reason.message });
  }
});
```

---

## Types & Validation (TypeScript)

```ts
// Always type your function signatures — never use implicit any
// Wrong
function processOrder(order: any) {}

// Correct — explicit shape defined once, reused everywhere
interface Order {
  id: string;
  userId: string;
  items: OrderItem[];
  total: number;
  status: "pending" | "processing" | "shipped" | "cancelled";
}

function processOrder(order: Order): ProcessedOrder {}

// Use union types instead of loose strings for controlled values
type UserRole = "admin" | "editor" | "viewer"; // enforced at compile time, not at runtime

// Use unknown instead of any when type is genuinely uncertain — forces type narrowing
function parseApiResponse(data: unknown): User {
  if (!isValidUser(data)) throw new ValidationError("Invalid user shape from API", "response");
  return data as User;
}

// Use readonly to signal data that must not be mutated after creation
interface Config {
  readonly apiUrl: string;
  readonly maxConnections: number;
}

// Avoid type assertions (as SomeType) unless you have confirmed the shape at runtime
// Writing "as X" tells the compiler to trust you — make sure you are right
```

---

## Testing Standards

```js
// Every function should have a corresponding test file
// Test file mirrors the source path:
//   src/utils/price.js   ->   tests/utils/price.test.js

// Structure: Arrange, Act, Assert (AAA)
describe("applyDiscount", () => {

  it("reduces the price by the correct discount percentage", () => {
    // Arrange
    const basePrice = 100;
    const discount = 20;

    // Act
    const result = applyDiscount(basePrice, discount);

    // Assert
    expect(result).toBe(80);
  });

  it("throws when discount exceeds 100", () => {
    // Invalid input must be rejected at the boundary
    expect(() => applyDiscount(100, 150)).toThrow("Discount must be between 0 and 100");
  });

  it("returns the original price unchanged when discount is 0", () => {
    // Zero discount is a valid edge case — price must remain exact
    expect(applyDiscount(50, 0)).toBe(50);
  });

  it("handles floating point prices correctly", () => {
    // Rounding must not introduce cents-level errors
    expect(applyDiscount(99.99, 10)).toBe(89.99);
  });
});

// Testing rules:
// - Tests must be deterministic — same result every run, no random values, no Date.now() unless mocked
// - One assertion per test where possible — a failing test should point to exactly one thing
// - Mock external dependencies (DB, APIs, file system) — unit tests must never hit the network
// - Test the behavior, not the implementation — tests should not break when you refactor internals
// - A test that is hard to write is a signal that the code being tested is too tightly coupled
```

---

## Security Baseline

```js
// Never trust user input — validate and sanitize everything at the boundary
function searchUsers(query) {
  // Reject empty or excessively long queries before they reach the database
  if (!query || query.length > 100) throw new ValidationError("Invalid search query", "query");

  // Use parameterized queries — never concatenate user input into SQL strings
  return db.query("SELECT * FROM users WHERE name ILIKE $1", [`%${query}%`]);
}

// Never expose internal error details to the client
// Wrong — leaks stack traces and implementation details
app.use((error, req, res, next) => {
  res.status(500).json({ error: error.stack });
});

// Correct — log internally, send a safe message externally
app.use((error, req, res, next) => {
  logger.error("Unhandled error", { path: req.path, reason: error.message });
  res.status(500).json({ error: "An unexpected error occurred" });
});

// Environment secrets must never appear in source code or logs
// Wrong
const token = "sk-prod-abc123xyz";    // committed to Git, exposed forever

// Correct
const token = process.env.API_SECRET_KEY; // loaded from environment at runtime

// Sanitize output when rendering user-supplied content to prevent injection attacks
// Use your framework's built-in escaping — never build HTML strings manually from user data
```

---

## Configuration & Environment

```js
// All environment-specific values go in .env — never hardcode them in source
// Wrong
const DB = "postgres://prod-server/real_database";   // exposed, fragile, environment-locked

// Always provide a documented .env.example for onboarding
// .env.example
DATABASE_URL=postgres://localhost:5432/your_db_name
API_SECRET_KEY=replace-with-your-secret
PORT=3000
MAX_CONNECTIONS=10

// Load and validate all config in one central location — fail fast on startup if anything is missing
const config = {
  db: {
    url: process.env.DATABASE_URL,
    maxConnections: parseInt(process.env.MAX_CONNECTIONS, 10) || 10,
  },
  api: {
    secretKey: process.env.API_SECRET_KEY,
    port: parseInt(process.env.PORT, 10) || 3000,
  },
};

// Crash immediately at startup if critical config is absent — better than a silent runtime failure
if (!config.db.url) throw new Error("DATABASE_URL is required but not defined in environment");
if (!config.api.secretKey) throw new Error("API_SECRET_KEY is required but not defined in environment");

module.exports = config;
```

---

## Code Review Checklist

Before marking code as ready for review, verify every item below:

```
  The code runs without errors locally
  All existing tests pass
  New logic has test coverage
  Functions are short, clearly named, and do one thing
  No hardcoded secrets, URLs, or magic numbers
  Comments explain WHY, not WHAT
  No console.log or debug artifacts remain in the code
  No separator lines or print-style debug patterns anywhere
  Environment variables are documented in .env.example
  No dead code or commented-out blocks left behind
  Copilot suggestions have been read, understood, and cleaned up
  The code follows every rule in this document
```

---

## What to Always Avoid

```js
// Magic numbers — every unexplained number must become a named constant
setTimeout(fn, 86400000);                        // wrong — what is this number?
const ONE_DAY_IN_MS = 24 * 60 * 60 * 1000;      // named and self-documenting
setTimeout(fn, ONE_DAY_IN_MS);                   // correct

// These patterns are never acceptable:
console.log("=".repeat(60));          // separator — wrong
console.log("--- debug ---");         // separator — wrong
console.log("step 1");                // execution tracing — wrong
console.log(someObject);             // raw object dump — wrong

// Other things to avoid:
// Deep nesting beyond 2-3 levels — flatten with early returns or extracted functions
// God functions that scroll for pages — break them apart
// Duplicate logic — extract to a shared utility (DRY: Don't Repeat Yourself)
// Global state mutation — keep state changes local and explicit
// Overloaded functions — a function that behaves differently based on argument type is two functions
// Ignoring linter or type errors — warnings today become runtime bugs tomorrow
// Floating promises — every async call must be awaited or explicitly fire-and-forget with a comment
// any type in TypeScript without a justification comment — it defeats the purpose of typing
// Commented-out dead code — delete it, Git history preserves it
// Excessive logging — if you are logging more than 2-3 times in a function, reconsider
```

---

## Key Principles Reference

| Principle             | Rule                                                                 |
|-----------------------|----------------------------------------------------------------------|
| Single Responsibility | One function does one thing                                          |
| DRY                   | Extract repeated logic into shared utilities                         |
| KISS                  | The simplest solution that works is the right one                    |
| Fail Fast             | Validate at the boundary, crash loudly, recover intentionally        |
| Explainability        | Any engineer should understand the code within minutes               |
| Testability           | If it is hard to test, it is too tightly coupled                     |
| Immutability          | Do not mutate inputs — return new values                             |
| Explicit Dependencies | Never rely on hidden global state — pass what you need               |
| Least Privilege       | Only access and expose what is strictly necessary                    |
| Consistency           | Follow the same patterns throughout the entire codebase              |

---

*This document is a living standard. Update it when new patterns are adopted or old ones are retired.*
