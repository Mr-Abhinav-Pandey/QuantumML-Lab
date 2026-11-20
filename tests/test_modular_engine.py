import numpy as np

from quantumml.circuits.ansatz import VariationalAnsatz
from quantumml.circuits.encoding import angle_encoding
from quantumml.experiments.service import ExperimentService
from quantumml.gradients.finite_difference import finite_difference_gradient
from quantumml.gradients.parameter_shift import parameter_shift_gradient
from quantumml.simulator import StatevectorSimulator
from quantumml.training.initialization import initialize_parameters
from quantumml.utils import stable_seed
from visualization.bloch import _partial_trace_single, state_to_bloch


def test_statevector_gates_preserve_norm():
    sim = StatevectorSimulator(3)
    sim.h(0)
    sim.ry_gate(0.37, 1)
    sim.rz_gate(-0.21, 2)
    sim.apply_cnot(0, 2)

    assert np.isclose(sim.norm(), 1.0)
    assert np.isclose(sim.get_probabilities().sum(), 1.0)


def test_bell_state_reduced_states_are_mixed():
    sim = StatevectorSimulator(2)
    sim.h(0)
    sim.apply_cnot(0, 1)

    for qubit in (0, 1):
        rho = _partial_trace_single(sim.state, qubit)
        assert np.allclose(rho, np.eye(2) / 2)
        assert np.isclose(np.trace(rho @ rho).real, 0.5)
        assert np.allclose(state_to_bloch(sim.state, qubit), (0.0, 0.0, 0.0))


def test_product_state_reduced_state_is_pure():
    sim = StatevectorSimulator(2)
    sim.h(0)
    rho = _partial_trace_single(sim.state, 0)

    assert np.isclose(np.trace(rho @ rho).real, 1.0)
    assert np.allclose(state_to_bloch(sim.state, 0), (1.0, 0.0, 0.0))


def test_ansatz_and_encoding_produce_normalized_state():
    sim = StatevectorSimulator(3)
    angle_encoding(sim, np.array([0.2, 0.7]))
    ansatz = VariationalAnsatz(3, 2, "circular")
    params = initialize_parameters(ansatz.parameter_shape, "small_angle", seed=4)
    ansatz.apply(sim, params)

    assert np.isclose(sim.norm(), 1.0)
    assert ansatz.num_parameters == 12


def test_parameter_shift_matches_finite_difference():
    ansatz = VariationalAnsatz(2, 1, "linear")
    params = initialize_parameters(ansatz.parameter_shape, "small_angle", seed=9)

    def cost(current):
        sim = StatevectorSimulator(2)
        angle_encoding(sim, np.array([0.25, 0.75]))
        ansatz.apply(sim, current)
        return sim.expectation_value_z(0)

    shifted = parameter_shift_gradient(cost, params)
    finite = finite_difference_gradient(cost, params)
    assert np.allclose(shifted, finite, atol=1e-5)


def test_stable_seed_is_process_independent_for_same_parts():
    assert stable_seed("B_entanglement", 0, "linear", 3) == 2123132253
    assert stable_seed("B_entanglement", 0, "linear", 3) != stable_seed(
        "B_entanglement", 0, "circular", 3
    )


def test_experiment_service_rejects_unknown_experiment():
    try:
        ExperimentService.run("unknown")
    except ValueError as exc:
        assert "Unknown experiment" in str(exc)
    else:
        raise AssertionError("Unknown experiments must raise ValueError")
