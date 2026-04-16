# Norse Cache Simulation Report

## Coverage
- Symbols scanned: `236`
- Cache files scanned: `236`
- Thor signals: `20894`
- Baldur signals: `974`
- Freya signals: `11266`

## Agent Outcomes
| Agent | Signals | TP | SL | TIMEOUT | Decided Acc | Weighted PF | Expectancy (ATR) | Avg Hold Bars |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Thor | 20894 | 5519 | 15244 | 131 | 26.58% | 1.001 | +0.0005 | 3.11 |
| Baldur | 974 | 302 | 591 | 81 | 33.82% | 0.909 | -0.0504 | 4.10 |
| Freya | 11266 | 3303 | 7852 | 111 | 29.61% | 0.847 | -0.0642 | 1.66 |

## Baldur Top-Start Study
- Top warnings generated: `3745`
- Confirmed Baldur shorts: `974`
- 1 ATR downside reached within `12` bars: `59.86%`
- Median delay to first sustained downside leg: `3`

## Thor/Freya Interaction
- Freya signals inside active Thor windows: `100.00%`
- Thor context bars: `24`
- Freya max bars: `8`

## Operating Verdict
- Thor remains the only live-ready Norse agent in this pass if its PF stays aligned with the existing Nike v2 baseline.
- Baldur and Freya should remain observe-only until their cache paper stats are reviewed.
- Legacy Greek specialists can stay loaded for attribution while model-driven live execution is limited to `model_live_specialists = 'nike'`.
