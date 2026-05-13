# Publishing gamma-compile-server to npm

End-to-end setup for shipping the daemon as a real npm package so end
users can install it with `npm install -g` and update with one
command. This is a one-time setup plus a small per-release ritual.

The package is configured to publish under the **scoped name**
`@9livezzz/gamma-compile-server`. Scopes are namespaces on npm, like
`@username/package` or `@org/package`. The scope `9livezzz` must
exist on npm before the first publish — see step 2.

---

## 1. Create an npm account

1. Go to https://www.npmjs.com/signup and create an account using
   the same email as your GitHub account (`lpfreiburg@ucsb.edu` works
   here too — they don't need to match, but it's convenient).
2. Verify the email npm sends you. **Account stays unverified until
   you click the link**, and publish will refuse a 403.
3. Pick a strong password. Save it in your password manager — you'll
   need it during `npm login`.

**Username vs. scope.** Your npm *username* is a personal namespace
(e.g. if your username were `lpfreiburg`, you could publish as
`@lpfreiburg/foo` without any extra setup). The `9livezzz` *scope*
in our package.json is different — it's either an npm organization
or an account named exactly `9livezzz`. Either works; see step 2.

## 2. Claim the `9livezzz` scope

You have two options.

### Option A — Use `9livezzz` as your username (simpler)

If you don't already have an npm account, sign up with the username
`9livezzz` directly. The scope `@9livezzz` is then automatically
yours; nothing extra to set up. No org features needed (you're a
single-developer project; org features mostly add billing complexity).

### Option B — Create an npm organization (if `9livezzz` is taken or you want org features)

1. Sign up with whatever username you want (e.g. `lpfreiburg`).
2. Go to https://www.npmjs.com/org/create and create an organization
   named `9livezzz`. **Free tier is fine for public packages** — no
   billing setup needed unless you need private packages later.
3. The org's "Free" plan allows unlimited public packages + up to 5
   members on the team. Plenty for a solo-with-occasional-collaborator
   project.

Either way, after this step `https://www.npmjs.com/~9livezzz` (account)
or `https://www.npmjs.com/org/9livezzz` (org) should resolve to your
profile/org page.

## 3. Enable two-factor auth (REQUIRED for publish)

npm requires 2FA on accounts that publish packages. There are two
modes:
- **`auth-and-writes`** — 2FA prompt on login *and* on every publish.
  Safer. Default.
- **`auth-only`** — 2FA only on login. Lets you publish without a
  prompt; fine for CI but means a compromised token can publish
  silently.

Pick one in the npm web UI under
**Profile → Account → Two-Factor Authentication**. Use an authenticator
app (1Password / Authy / Google Authenticator), NOT SMS. Save the
recovery codes somewhere durable.

For CI auto-publish (step 7) you'll generate a **publish token** that
bypasses the 2FA prompt — that's a separate piece, doesn't conflict
with the human-publish 2FA setting.

## 4. Authenticate locally

```powershell
# From any directory. Will open a browser tab for OAuth-style login,
# OR prompt for username + password + 2FA OTP in the terminal,
# depending on your npm version.
npm login --scope=@9livezzz --auth-type=web
```

On success, `npm whoami` should print your username. If you used
Option B (org), you'll show up as your personal username here, not
as `9livezzz` — that's correct.

**Where the credentials live:** `~/.npmrc` on macOS/Linux,
`%USERPROFILE%\.npmrc` on Windows. Tokens look like
`//registry.npmjs.org/:_authToken=npm_XXXXXXXXX…`. Don't commit this
file (it's already in npm's default gitignore patterns; just don't
manually add it to a git repo).

## 5. Smoke-test what will be published

```powershell
cd C:\Users\lpfre\OneDrive\Documents\gamma-compile-server
npm pack --dry-run
```

Lists every file that *would* be included in the published tarball.
Should include `bin/`, `src/`, `package.json`, `README.md`, `LICENSE`
— matches the `"files"` array in package.json. **It should NOT
include** `node_modules/`, `docs/`, `tests/`, `.git/`, your `.env`,
or any local cache. If something extra shows up, add it to
`.npmignore` (a separate file, similar syntax to `.gitignore`) or
tighten the `files` array.

A typical clean output looks like:
```
npm notice 📦  @9livezzz/gamma-compile-server@0.3.1
npm notice === Tarball Contents ===
npm notice 1.2kB  LICENSE
npm notice 3.4kB  README.md
npm notice 1.0kB  package.json
npm notice 4.5kB  bin/gamma-compile-server.js
npm notice 8.7kB  src/compile.js
npm notice 12.0kB src/osc-bridge.js
npm notice 9.1kB  src/osc-codec.js
npm notice 5.3kB  src/server.js
npm notice 2.8kB  src/setup.js
npm notice ...
```

## 6. First publish

```powershell
cd C:\Users\lpfre\OneDrive\Documents\gamma-compile-server
npm publish --access public
```

The `--access public` flag is REQUIRED for the first publish of a
scoped package on the free tier. Without it, npm defaults scoped
packages to private, which the free tier doesn't allow, and you get
a vague 402 error.

After success, the package is live at
https://www.npmjs.com/package/@9livezzz/gamma-compile-server and
anyone can install it with:

```bash
npx @9livezzz/gamma-compile-server@latest          # one-shot
npm install -g @9livezzz/gamma-compile-server      # persistent
```

Verify with:

```powershell
npm view @9livezzz/gamma-compile-server version
# Should print: 0.3.1
```

## 7. Per-release ritual (manual)

Each time you ship changes:

```powershell
cd C:\Users\lpfre\OneDrive\Documents\gamma-compile-server

# 1. Make your code changes, test them.

# 2. Bump the version in package.json. npm version flips the number
#    AND creates a git commit + tag, in one step:
npm version patch     # 0.3.1 -> 0.3.2  (bug fixes, small additions)
# OR
npm version minor     # 0.3.1 -> 0.4.0  (new features, backward-compatible)
# OR
npm version major     # 0.3.1 -> 1.0.0  (breaking changes)

# 3. Publish to npm. 2FA prompt if you chose auth-and-writes mode.
npm publish

# 4. Push the commit + tag to GitHub so they line up with releases:
git push --follow-tags
```

**Don't manually edit the version in package.json** — let `npm version`
do it. It also rewrites `package-lock.json` correctly, and the git
tag it creates is what step 8 (auto-publish CI) keys on.

## 8. (Recommended) Auto-publish on git tag push

The per-release ritual above works fine for a solo project. If you'd
rather not run `npm publish` from your laptop every time, set up a
GitHub Actions workflow that publishes on every tag push:

### 8.1 Generate an npm publish token

1. https://www.npmjs.com/settings/<your-username>/tokens (or
   `npm token create` from the CLI).
2. **Token type: "Automation"** — these bypass 2FA, which is what CI
   needs. (A "Publish" token would prompt for 2FA on every CI run and
   immediately fail.)
3. Copy the token. You can't view it again.

### 8.2 Store the token as a GitHub Actions secret

1. On https://github.com/9LiveZZZ-Git/gamma-compile-server/settings/secrets/actions
   click **New repository secret**.
2. Name: `NPM_TOKEN`. Value: the token from step 8.1. Save.

### 8.3 Add the workflow

Create `.github/workflows/publish.yml`:

```yaml
name: Publish to npm

on:
  push:
    tags:
      - 'v*'

jobs:
  publish:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      id-token: write   # for npm provenance attestation (optional but nice)
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: '20'
          registry-url: 'https://registry.npmjs.org'
      - run: npm ci
      - run: npm publish --access public --provenance
        env:
          NODE_AUTH_TOKEN: ${{ secrets.NPM_TOKEN }}
```

Then your release flow becomes:

```powershell
npm version patch        # creates the v0.3.2 tag locally
git push --follow-tags   # tag arrives on GitHub; workflow fires; package goes live
```

The `--provenance` flag attaches a signed attestation to the publish
saying "this tarball was built from commit X in repo Y by GitHub
Actions" — visible on the npm page as a green provenance badge. Free
and a nice trust signal for users.

## 9. Update the README

Once published, swap the README's install instructions from the
implicit `npx @9livezzz/gamma-compile-server` (which currently 404s)
to the verified-working version. Suggested edit:

```markdown
## Quick start

Requires **Node 20+** and **git** on your PATH.

```bash
# One-shot (no install):
npx @9livezzz/gamma-compile-server@latest

# Or install once:
npm install -g @9livezzz/gamma-compile-server
gamma-compile-server
```

## Updating

```bash
npm install -g @9livezzz/gamma-compile-server@latest
```

Or just re-run `npx @9livezzz/gamma-compile-server@latest` — npx
re-resolves `@latest` on every invocation.
```

## 10. Common gotchas

**"402 Payment Required"** on first publish — you forgot
`--access public`. Add it. Scoped packages default to private; private
packages need a paid plan.

**"403 Forbidden"** on publish — usually means either (a) your email
isn't verified, (b) the package name is already taken by someone
else, or (c) your 2FA prompt timed out. Re-run with `npm publish`
and watch for the OTP prompt.

**"You cannot publish over the previously published versions"** —
npm doesn't allow overwriting. Bump the version (`npm version patch`)
and try again. (npm DOES let you `unpublish` within 72 hours, but
that's a footgun + breaks anyone who already installed; just bump.)

**"sha integrity check failed"** when users install — usually means
their local npm cache is stale. They can run `npm cache clean
--force` and retry. Almost never our fault.

**Token expired** — automation tokens don't expire by default, but
you can configure expiry on creation. If CI publishes start failing
with 401, regenerate the token + update the GitHub Actions secret.

**Scope name conflict** — if `@9livezzz` is taken and you can't get
it, you can change the scope by editing the `name` field in
package.json (e.g. `@lpfreiburg/gamma-compile-server`). The editor's
README + status pill messaging would need a one-line update to match.

---

## TL;DR

```powershell
# One-time setup
npm login --scope=@9livezzz --auth-type=web

# Each release
npm version patch
npm publish --access public        # first time only needs --access public; npm remembers it after
git push --follow-tags
```

Set up CI per §8 once + you can drop the manual `npm publish` step
forever (just push tags).
