import os
from typing import List, Tuple

from functools import partial
import numpy as np
from cirq import *

from .utils import get_qubits

from quantum_decomp import matrix_to_cirq_circuit as decompose

import openql.openql as ql

from cirq.contrib.qasm_import import circuit_from_qasm


GATES = {
    1: [X, Y, Z, H, S, T],
    2: [
        google.SycamoreGate(),
        CX,
        XX,
        YY,
        ZZ,
        IdentityGate(num_qubits=2),
    ],
    3: [
        CCX,
        CSWAP,
        ControlledGate(ISWAP ** 0.5),
        CCZ,
        IdentityGate(num_qubits=3),
    ],
    4: [
        ControlledGate(CCX),
        IdentityGate(num_qubits=4),
    ]
}


def test_gate(target_qubits, matrix, ancillae, gate):
    response_circuit = Circuit([gate(*target_qubits)])
    response_unitary = response_circuit.unitary(
        qubit_order=target_qubits + ancillae,
        qubits_that_should_be_present=target_qubits + ancillae
    )
    expected_unitary = kron(matrix, np.eye(2 ** len(ancillae)))
    u = response_unitary @ expected_unitary.conj().T
    trace_distance = trace_distance_from_angle_list(
        np.angle(np.linalg.eigvals(u))
    )
    return trace_distance < 1e-4


def test_gates(target_qubits, matrix, ancillae, gates):
    print('Testing gates')
    print(target_qubits, gates)
    for gate in gates:
        print(f'Gate: {gate}')
        if test_gate(
            target_qubits, matrix, ancillae, gate
        ):
            return [gate(*target_qubits)], ancillae
    return NotImplemented, ancillae


def gen_incrementers_ops(
    target_qubits: List[GridQubit], matrix: np.ndarray
):
    n_qubits = len(target_qubits)
    if n_qubits == 1:
        yield X(target_qubits[-1])
        return

    for i in range(n_qubits - 1, 0, -1):
        if i != n_qubits - 1:
            yield X(target_qubits[i])
        yield CNOT(target_qubits[i], target_qubits[i - 1])
        yield X(target_qubits[i])


def handle_simple_cases(
    target_qubits: List[GridQubit], matrix: np.ndarray
):
    n_qubits = len(target_qubits)

    # Handle identities
    if np.allclose(np.eye(2 ** n_qubits), matrix):
        return [], []

    # Handle CNOT, TOFFOLI, CCZ
    if n_qubits == 2 and test_gate(target_qubits, matrix, ancillae=[], gate=CNOT):
        return [CNOT(*target_qubits)], []
    elif n_qubits == 3 and test_gate(target_qubits, matrix, ancillae=[], gate=CCX):
        return [
            H(target_qubits[-1]),
            CNOT(*target_qubits[1:]),
            inverse(T(target_qubits[-1])),
            CNOT(target_qubits[0], target_qubits[-1]),
            T(target_qubits[-1]),
            CNOT(*target_qubits[1:]),
            inverse(T(target_qubits[-1])),
            CNOT(target_qubits[0], target_qubits[-1]),
            T(target_qubits[1]),
            T(target_qubits[-1]),
            CNOT(*target_qubits[:-1]),
            T(target_qubits[0]),
            inverse(T(target_qubits[1])),
            CNOT(*target_qubits[:-1]),
            H(target_qubits[-1]),
        ], []
    elif n_qubits == 3 and test_gate(target_qubits, matrix, ancillae=[], gate=CCZ):
        return [
            CNOT(*target_qubits[1:]),
            inverse(T(target_qubits[-1])),
            CNOT(target_qubits[0], target_qubits[-1]),
            T(target_qubits[-1]),
            CNOT(*target_qubits[1:]),
            inverse(T(target_qubits[-1])),
            CNOT(target_qubits[0], target_qubits[-1]),
            T(target_qubits[1]),
            T(target_qubits[-1]),
            CNOT(*target_qubits[:-1]),
            T(target_qubits[0]),
            inverse(T(target_qubits[1])),
            CNOT(*target_qubits[:-1]),
        ], []

    # Hnadle incrementers
    target = np.empty((2 ** n_qubits, 2 ** n_qubits))
    target[1:] = np.eye(2 ** n_qubits)[:-1]
    target[:1] = np.eye(2 ** n_qubits)[-1:]
    if np.allclose(target, matrix):
        return gen_incrementers_ops(target_qubits, matrix), []

    # Handle specific 1-4-qubit gates
    ancillae = []
    query_gates = partial(test_gates, target_qubits, matrix, ancillae)

    print(n_qubits)
    gates = GATES[n_qubits]
    response, ancillae = query_gates(gates)
    if response is not NotImplemented:
        return response, ancillae
    
    return NotImplemented, []


def handle_complex_cases(
    target_qubits: List[GridQubit], matrix: np.ndarray
):
    return NotImplemented, []


def handle_all_cases(
    target_qubits: List[GridQubit], matrix: np.ndarray
):
    response, ancillae = handle_simple_cases(target_qubits, matrix)
    if response is not None and ancillae is not None:
        return response, ancillae
    else:
        return handle_complex_cases(target_qubits, matrix)


def optimize_for_score(
    target_qubits, matrix, response, ancillae
):
    n_qubits = len(target_qubits)
    if n_qubits == 1:
        circuit = Circuit(response)
        converted = google.optimized_for_sycamore(circuit)
        ops = [op for moment in converted for op in moment]
        return ops, ancillae
    else:
        return response, ancillae
    


def external_solution(
    target_qubits, matrix
):
    circuit = decompose(matrix)
    ops = [op.transform_qubits(lambda q: target_qubits[q.x]) for moment in circuit for op in moment]
    return ops, []


def openql_solution(
    target_qubits, matrix
):
    response, ancillae = handle_simple_cases(target_qubits, matrix)
    if response is not NotImplemented:
        return response, ancillae

    n_qubits = len(target_qubits)
    ql.set_option('output_dir', os.path.join('/tmp', 'qasm_output'))
    ql.set_option('log_level', 'LOG_ERROR')

    curr_dir = os.path.dirname(os.path.realpath(__file__))
    platform = ql.Platform('starmon', os.path.join(curr_dir, 'simple.json'))
    program = ql.Program('example', platform, n_qubits)
    kernel = ql.Kernel("decomposedKernel")

    compiler = ql.Compiler('compiler')
    compiler.add_pass_alias('Writer', 'scheduledqasmwriter')
    compiler.set_pass_option('scheduledqasmwriter', 'write_qasm_files', 'yes')

    list_matrix = [elem for sublist in matrix.tolist() for elem in sublist]
    # matrix = [0.5+0.5j,0.5-0.5j,0.5-0.5j,0.5+0.5j]

    unitary_matrix = ql.Unitary('decomposed', list_matrix)
    unitary_matrix.decompose()
    kernel.gate(unitary_matrix, range(0, n_qubits))

    program.add_kernel(kernel)
    compiler.compile(program)

    with open('/tmp/qasm_output/example_scheduledqasmwriter_out.qasm', 'r') as f:
        data = f.read()

    data = data.split('.decomposedKernel\n', 1)[1].replace('\t', '')
    data = '\n'.join([line.strip() for line in data.split('\n')])

    result = []
    data = data.replace('cnot', 'cx')
    for line in data.split('\n'):
        if line[:2] in {'ry', 'rz'}:
            index = line.index(', ')
            angle = line[index + 2:]
            line = f'{line[:2]}({angle}){line[2:index]}'
        if line:
            result.append(line + ';')
    data = '\n'.join(result)

    header = """
    OPENQASM 2.0;
    include "qelib1.inc";
    """
    header += f'qreg q[{n_qubits}];\n'
    data = header + data

    circuit = circuit_from_qasm(data)
    print(circuit)
    ops = [op.transform_qubits(lambda q: target_qubits[int(str(q).strip('q_'))]) for moment in circuit for op in moment]
    print(ops)
    return ops, []


def matrix_to_sycamore_operations(
    target_qubits: List[GridQubit], matrix: np.ndarray
) -> Tuple[OP_TREE, List[GridQubit]]:
    """A method to convert a unitary matrix to a list of Sycamore operations.

    This method will return a list of `cirq.Operation`s using the qubits and (optionally) ancilla
    qubits to implement the unitary matrix `matrix` on the target qubits `qubits`.
    The operations are also supported by `cirq.google.gate_sets.SYC_GATESET`.

    Args:
        target_qubits: list of qubits the returned operations will act on. The qubit order defined by the list
            is assumed to be used by the operations to implement `matrix`.
        matrix: a matrix that is guaranteed to be unitary and of size (2**len(qs), 2**len(qs)).
    Returns:
        A tuple of operations and ancilla qubits allocated.
            Operations: In case the matrix is supported, a list of operations `ops` is returned.
                `ops` acts on `qs` qubits and for which `cirq.unitary(ops)` is equal to `matrix` up
                 to certain tolerance. In case the matrix is not supported, it might return NotImplemented to
                 reduce the noise in the judge output.
            Ancilla qubits: In case ancilla qubits are allocated a list of ancilla qubits. Otherwise
                an empty list.
        .
    """
    # print(f'Target qubits:\n{target_qubits}\nMatrix:\n{matrix}')
    solution = (
        # handle_all_cases
        # external_solution
        openql_solution
    )

    response, ancillae = optimize_for_score(
        target_qubits, matrix, *solution(target_qubits, matrix)
    )

    return response, ancillae
