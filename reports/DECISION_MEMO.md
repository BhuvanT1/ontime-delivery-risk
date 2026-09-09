# Late deliveries at Olist: what the data supports, and what it does not

**Scope.** 96,470 delivered orders, September 2016 to August 2018, from the public
Olist Brazilian e-commerce dataset. 6.8% arrived after the date promised at
checkout. This memo answers three questions: where lateness concentrates, what
it is associated with, and whether a predictive model can be turned into an
intervention that pays.

**The short version.** The per-order prediction model is not good enough to
justify a proactive intervention programme. The segment analysis is, and points
at two specific, sizeable targets. The recommendation is to act on the segments
and not to deploy the model as a trigger.

---

## 1. Lateness is concentrated, and distance explains most of the map

Late rate rises monotonically with the distance between seller and customer,
from 4.4% under 50 km to 12.0% beyond 2,000 km. Intervals are tight at every
band, so the ordering is real and not an artefact of volume.

That makes raw state-level comparisons misleading. Ceará looks like a problem
at 13.8% late; almost all of that is being 2,187 km from the seller base. To
compare destinations fairly, each state's observed late count is set against
what national rates predict for that state's own mix of distance bands and
half-year periods — an SMR, with Byar intervals.

| State | Orders | Observed late | Expected | SMR (95% CI) | Excess |
|---|---:|---:|---:|---|---:|
| AL | 397 | 85 | 40.1 | 2.12 (1.69–2.62) | +45 |
| **RJ** | **12,350** | **1,495** | **828.0** | **1.81 (1.72–1.90)** | **+667** |
| MA | 717 | 125 | 74.6 | 1.68 (1.40–2.00) | +50 |
| CE | 1,279 | 176 | 132.1 | 1.33 (1.14–1.55) | +44 |

**Rio de Janeiro is the finding.** Alagoas has the worse rate; Rio has the
problem. It is the second-largest destination in the country, it is not remote,
and it still runs 81% above comparable orders elsewhere with an interval nowhere
near 1.0. It accounts for more excess late orders than every other
worse-than-expected state combined.

Eight states, including São Paulo, Minas Gerais and Paraná, run significantly
*better* than comparable orders — so this is not a uniformly stretched network.

## 2. The lost days are mostly on the carrier's side of the handoff

Splitting each late order's delay against the on-time median of its own distance
band:

| Distance | Excess seller-side | Excess carrier-side | Carrier share |
|---|---:|---:|---:|
| <200 km | 6,230 d | 12,622 d | 67% |
| 200–500 km | 7,419 d | 41,017 d | 85% |
| 500–1,000 km | 7,263 d | 39,514 d | 85% |
| 1,000 km+ | 5,455 d | 37,563 d | 87% |

Median transit on a late order is 26.2 days against 7.0 on an on-time order.
The seller-side gap is real but far smaller (3.0 days against 1.8). Sellers do
miss their contractual handoff deadline much more often on late orders (27.8%
against 7.7%), so seller behaviour is a genuine signal — it is just not where
most of the elapsed time goes, except on short routes where it is a third of it.

## 3. Seller reliability separates cleanly, and a small group carries it

Segmenting sellers on volume and on whether their late rate is distinguishable
from the 6.8% base rate:

| Segment | Sellers | Orders | Late rate |
|---|---:|---:|---:|
| High volume, reliably late | 57 | 15,697 (16.3%) | 11.5% |
| High volume, reliably good | 55 | 14,572 (15.1%) | 3.4% |
| High volume, indistinguishable | 306 | 42,391 | 6.2% |
| Low volume, evidence of a problem | 186 | 1,811 | 28.7% |
| Low volume, too few orders to judge | 2,354 | 21,999 | 4.9% |

57 sellers — 1.9% of the seller base — carry one order in six at nearly double
the base rate. That is roughly 730 excess late orders a year concentrated in a
list short enough to work through by hand.

## 4. Lateness is strongly associated with review damage, and severity matters

| Outcome | 1- or 2-star share |
|---|---:|
| On time | 9.5% (9.3–9.7) |
| 1–2 days late | 25.8% |
| 3–7 days late | 64.2% |
| 8–14 days late | 80.3% |
| 15+ days late | 78.4% |

A day or two late is survivable; a week is not. Note the shape — the damage is
not linear, so a policy that shaves a few days off the worst tail is worth more
than one that improves the average.

**This is an association, not a measured effect.** Nothing here was randomised.
Late orders differ from on-time orders in route, seller and category, and any of
those can drive both the delay and the score. The number to take from this table
is the shape of the relationship, not a coefficient to multiply revenue by.

## 5. The prediction model works, and is still not good enough to act on

Predicting lateness from what is knowable **at checkout**, with a chronological
split and the champion chosen on validation:

| Model | ROC-AUC | PR-AUC | ECE |
|---|---:|---:|---:|
| Base rate | 0.500 | 0.035 | 0.044 |
| Promise-length rule | 0.636 | 0.056 | 0.054 |
| **Logistic (champion)** | **0.670** | **0.065** | 0.156 |
| — after isotonic calibration | 0.669 | 0.061 | **0.038** |
| Gradient boosting | 0.656 | 0.053 | 0.065 |
| *With post-delivery fields (leaky)* | *0.994* | *0.908* | *0.012* |

Two things to read here. First, the honest model beats the base rate but not by
much: PR-AUC 0.065 against a 0.035 floor, top-decile lift 1.91, and flagging the
riskiest 20% of orders catches 30% of late ones. Second, the same model handed
post-delivery columns scores 0.994 — that gap is the leak, and it is why
published accuracy figures on this dataset should be read with the feature list
in hand.

Calibration matters more than usual here because the late rate falls from 7.9%
in the training window to 3.5% in the test window. Isotonic regression fitted on
validation cuts expected calibration error from 0.156 to 0.038 without touching
the ranking.

### The policy does not clear break-even

Under the stated defaults — R$60 per late delivery, R$12 per proactive
intervention, 30% of lateness prevented when we act — the best available
threshold **loses** money. The intervention would have to prevent **84%** of the
lateness it touches to break even, which no realistic logistics action does.

Sweeping all three assumptions over 27 combinations, only 4 produce a positive
saving, and the best of those is R$0.031 per order (0.7%). The conclusion is
robust to the assumptions rather than dependent on them: **a per-order
intervention driven by this model is not worth deploying.**

*Every figure in this section is conditional on stated assumptions. None is a
measured business impact, and none should be quoted as one. Establishing real
effectiveness requires an experiment, not this dataset.*

---

## Recommendations

1. **Work the 57-seller list, not the order queue.** The segment is stable,
   statistically clear, and small enough to act on individually. Start with
   handoff-deadline compliance, where those sellers are visibly worse.
2. **Open a carrier review for Rio de Janeiro.** 667 excess late orders against
   comparable routes, on high volume, at short distance. This is the largest
   single addressable block in the data and it is not a distance problem.
3. **Do not deploy the risk model as an intervention trigger.** Publish it as a
   monitoring score if useful, but the economics do not support acting on it.
4. **Before any of this is credited with an effect, run an experiment.** The
   break-even effectiveness for the per-order policy is 0.84; for the seller
   programme the equivalent number should be computed the same way and tested
   against a holdout group of sellers. Everything in sections 1–4 is
   observational and cannot settle causation on its own.

## What would change these conclusions

- **Carrier identity.** The dataset has no carrier field, and section 2 puts
  most of the lost time on the carrier's side. This is the single most valuable
  missing column.
- **A promise-setting policy change.** Olist quotes a median 24-day promise
  against a 9.8-day on-time delivery. The buffer is doing a lot of work, and any
  change to how it is set moves the label itself.
- **Undelivered orders.** The label is defined only on delivered orders.
  Cancellations and never-delivered orders are excluded, which biases the
  population toward better outcomes by an unknown amount.
