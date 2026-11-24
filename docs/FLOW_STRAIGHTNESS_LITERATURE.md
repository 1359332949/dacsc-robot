# Primary-source checks for the flow-geometry experiment

Checked 2026-11-24 from arXiv abstract pages using HTTPS requests; a scoped reading, not an exhaustive novelty search. These notes distinguish the supplied proposal from verified source descriptions.

| Primary source | Verified scope | Consequence here |
|---|---|---|
| Liu et al., Flow Straight and Fast, arXiv:2209.03003 | Learns to follow straight paths as much as possible; recursive rectification produces increasingly straight flows. | A straight training interpolation is not a theorem that each learned conditional trajectory has constant velocity. |
| SALT, arXiv:2610.07946 | Uses overlapping leftover action chunks, nominal-calibrated adaptation gate, disruption onset and subsequent propagation. | H=E leaves no leftover supervisory interval in that mechanism; this does not establish that every possible adaptation of SALT fails. |
| RECLAIM, arXiv:2611.24494 | Frozen-policy attention responses to local visual interventions, robot geometry and patch purification. | The checked abstract does not establish the supplied claim that discrete action-token entropy is required. Avoid that exclusion claim. |
| T3R, arXiv:2611.08413 | Segmentation-guided pruning and saliency-guided attention bias; evaluated on multiple VLA backbones. | Relevant spatial-refinement comparison, but no evidence from the abstract that it cannot help blurred inputs. Our border-prototype bias is a heuristic inspired by this family, not its official implementation. |
| CoRe, arXiv:2608.14822 | Synthesized counterfactual continuations and minimal robot/scene realignment after disruptions. | Different recovery setup; no general impossibility conclusion about static visual shifts follows. |

## Mathematical issue in the proposal

For squared conditional flow matching, the population regression solution is the conditional expectation of the path target velocity given current action state, flow time and observation. Different endpoint pairs can contribute at different states/times. Consequently, straight conditional training paths do not imply a constant learned velocity along a generated ODE trajectory. Conversely, constant incorrect velocity or zero motion has zero velocity variance. Variance reduction can therefore be an optimization shortcut rather than restoration of a valid action distribution.

We test raw variance and normalized variance separately, record speed and angle, check held-out initial noise after one gate step, and retain all negative results. A gradient map measures sensitivity, not a nuisance segmentation mask or causal attribution. A larger action amplitude is not a ground-truth recovery metric. Endpoint diversity can reflect valid multimodality rather than uncertainty.

No reviewed source establishes novelty of this proposal; no claim of first-ever originality, universal flow-policy applicability, or anticipated +10 to +15 recoveries is made. No learned suffix-manifold fallback is launched automatically after a negative diagnostic.
