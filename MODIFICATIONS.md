# Modifications

This repository is a **modified copy** of [ColinGPT9/clips-studio](https://github.com/ColinGPT9/clips-studio)
(Clips Kitty), licensed under the GNU Affero General Public License v3.0 or later. See
[LICENSE](LICENSE) and [NOTICE](NOTICE). It is not the official project and is not endorsed by
its authors. The original history is kept unchanged, and the original repository is the
`upstream` remote.

## What was changed (2026-11-20)

- **Added** `llm/providers/spend.py`: a local monthly spending limit for cloud AI requests.
- **Changed** `llm/providers/http.py` and `llm/providers/cloud_backend.py`: a few lines each, to call it.
- **Added** `tests/test_spend_limit.py`: tests for the limit, with the network faked.
- **Added** this file, and a one-line notice at the top of `README.md`.

Nothing else differs from upstream.

## About the spending limit

- It stops sending language-model requests once this month's **estimated** spend would pass a cap
  (default **$5.00**), set in `data/ai_spend_limit.json` (created on first use).
- It is a second layer, not the real limit. The dollar figure is an estimate: tokens counted from
  each response's `usage`, priced from the rates in that file. The default rate is a deliberately
  high placeholder, so put the real per-model rates in `rates`. Set the provider's own spend limit
  as well; that is the real ceiling.
- It only covers language-model requests made through this app (not online transcription), and
  parallel requests can overshoot by about one request each.
- If its files can't be read or written it refuses the request rather than letting it through
  unmetered. `"monthly_cap_usd": null` switches it off.
- `python -m llm.providers.spend data` prints the cap and this month's estimate.

There is no warranty, as the license says.
