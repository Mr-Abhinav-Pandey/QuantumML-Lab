# Published findings — study notes

**Everything in this file is literature-derived. It is not a result of this project.**

## McClean et al. (2018)
- Main lesson: under particular random-circuit and cost-function assumptions, gradients can concentrate near zero as the number of qubits grows.
- Study carefully: the circuit ensemble, depth assumptions, observable/cost, and variance definition.
- Do not simplify the conclusion to “deep circuits always have barren plateaus.”

## Cerezo et al. (2021)
- Main lesson: cost-function locality matters. The paper establishes different scaling behaviour for global and local costs under its stated assumptions.
- Important implication: architecture and depth cannot be studied in isolation from the objective/observable.
- For a fair reproduction, the cost function must match the paper's construction.

## Ragone et al. (2024)
- Main lesson: Lie-algebraic structure can provide a general way to characterize gradient variance for deep parameterized circuits.
- Important implication: two circuits with similar depth may have different trainability because their generated algebraic structures differ.

## Cunningham & Zhuang (2025)
- Main lesson: barren plateaus have multiple proposed causes and mitigation directions; ansatz, initial state, observable, loss and noise all matter.
- Use this review to discover primary sources, not as a source for unverified numerical datasets.

## Reproduction status
No paper-level numerical reproduction has been completed merely by including these notes. The current project's original experiments must be run and validated separately before any result is labelled `REPRODUCED_REFERENCE_RESULT`.
