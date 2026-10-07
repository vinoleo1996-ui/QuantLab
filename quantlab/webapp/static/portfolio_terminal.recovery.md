# Recovered terminal UI assets

The terminal CSS is a byte-for-byte recovery of the original dark, hard-edge
stylesheet from the local Chrome Simple Cache. The JavaScript is its recovered
base plus narrowly scoped compatibility repairs for the isolated Grok paper
ledger: position fields, truthful quote-state handling, the static 年K control,
and explicit public-market opt-in. `/portfolio` now serves this terminal; it is
not an unused alternative UI.

## Source provenance

| Original recovered source | Cache entry | Source byte range | SHA-256 of recovered source |
| --- | --- | --- | --- |
| `portfolio_terminal.js` | `06efb1dfb575f753_0` | `[107, 40462)` | `ced3428a4c411a8319272dc53f63f7f7d2c9b9fe74bf502707c8903d47c56413` |
| `portfolio_terminal.css` | `17593216ef377cf6_0` | `[452, 32772)` | `91b88b3cbdfcfefda1115c7a2064fbf4c0d0187321f52f548f6121cf8c4af6a0` |
| cached HTML shell | `bfe2ee302eaca6fc_0` | `[88, 7961)` | `eb79b2d83b6e031a0e73239d22ada79668c15222630af8953c4e65a3ac414d58` |

The CSS cache record has a nine-byte Simple Cache trailer
`d8410d97456ffaf401` at `[32772, 32781)`, immediately before its first NUL.
It is cache metadata rather than UTF-8 CSS and was not copied. The recovered
HTML shell is `portfolio_terminal.html`. It retains the original terminal
layout, while its asset references point to `portfolio_terminal.css/js` and it
now declares 年K, the truthful 15-year K-line bound, and the explicit
public-market permission text. The patch transport also adds one harmless final
line terminator because the cached HTML ended without one.

## Original runtime contract

- **Page shell / DOM:** use `portfolio_terminal.html` as the companion shell.
  The script requires the five `.pane` regions
  (`overview-pane`, `holdings-pane`, `signals-pane`, `experts-pane`,
  `run-pane`), tabs with matching `data-pane`, and its named mounts,
  including `chartNav`, `chartDD`, `htable`, `chartK`, `tape`,
  `kpis`, `qSector`, `stable`, `ecards`, and `runKpis`.
- **ECharts:** required. Load a global `window.echarts` before
  `portfolio_terminal.js`; the recovered shell retains
  `/static/echarts.min.js` ahead of the terminal script.
- **Read endpoints expected by the original script:**
  `/api/nav`, `/api/series`, `/api/risk`, `/api/holdings`,
  `/api/intraday?code=<code>&days=1`,
  `/api/kline?code=<code>&freq=<day|week|month|year>&bars=<n>`,
  `/api/indices`, `/api/signals`, and `/api/experts`.
- **One mutating-looking legacy call:** the old “重新收集今日信息并分析” button
  also invokes `/api/experts?refresh=1`. The recovered terminal now presents
  it as a disabled read-only control; it is not wired to any research or
  trading action.
