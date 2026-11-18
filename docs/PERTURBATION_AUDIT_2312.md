# Perturbation audit: Lighting 2292 vs 2312

Audit date: 2026-11-18.

## BDDL

The two BDDL files differ only in the `(define (problem ...))` name:
`...tabletop_light_sync_modified_437` (2292) vs
`...tabletop_light_sync_modified_481` (2312). The task/language/goal are
unchanged.

## Scene XML

The BDDL names resolve to different lighting scene XML files. The audit found:

- 2292: `light1` directional=false, specular=0.7, dir=(0.5,0.5,-1),
  pos=(1,1,4.0);
- 2312: `light_kitchen` directional=true, specular=0.0, dir=(-0.5,0.5,1),
  pos=(-1.75,0,1.6);
- second light also changes name, specular and direction.

Therefore 2312 is **not a pure intensity change**: it changes light type,
direction, specular and position. It should be called a compound
lighting/scene-XML perturbation, not “darker lighting”.

## Action clipping in the traces

Both failure (state 3) and success (state 40) clip the gripper dimension on
about 90–93% of steps; max raw overshoot is ~0.02. Clipping is common in both,
so it is not by itself the discriminator between success and failure.

## Consequence

The 2292/2312 comparison cannot be used to claim “pure lighting robustness”.
It is a perturbation-level configuration holdout. Future claims must name the
exact scene XML/BDDL hashes.
