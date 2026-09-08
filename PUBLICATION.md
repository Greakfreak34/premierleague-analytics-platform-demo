# Public/private distribution boundary

Only authored implementation files were copied through the explicit allowlist in
`tools/copied_code_allowlist.json`. Public adaptations remove live football/social
collectors and private benchmark checks, require a synthetic model, and identify
the simulated clock in API responses. Source data and private Git metadata were
never copied. This repository starts with one new root commit.

Excluded from this distribution:

- All downloaded HTML and raw third-party CSV/JSON datasets.
- All private processed matches, players, availability archives and row-level reports.
- All private trained parameters, figures, predictions and evaluation artifacts.
- Account configuration, credentials, personal commit email, and private commit history.

The generator writes fictional data locally. Generated directories are ignored;
the public commit consists of code, tests, documentation and dependency manifests.
No statement about real predictive performance is made by synthetic metrics.

The private research explored whether football-specific features improve a simple
chronological baseline and whether more complex model families help. This release
exposes those implementation patterns without claiming to reproduce the private
experiment's data or results. Availability and sentiment remain context only.

Potential future data integrations need independent checks of access rights,
redistribution rights, privacy and attribution. A downloader is not a workaround
for a provider's restrictions. Tests here never require such access.
