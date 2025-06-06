# Copyright Quantinuum
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Methods to allow conversion between pyQuil and tket data types"""

import math
from collections import defaultdict
from collections.abc import Callable
from logging import warning
from typing import (
    Any,
    Literal,
    TypeVar,
    cast,
    overload,
)

from sympy import Add, Expr, Mul, Number, Pow, Symbol, cos, pi, sin

from pyquil import Program
from pyquil.api import QuantumComputer
from pyquil.external.rpcq import GateInfo, MeasureInfo
from pyquil.quilatom import (
    Add as Add_,
)
from pyquil.quilatom import (
    Div,
    Expression,
    MemoryReference,
    Sub,
    quil_cos,
    quil_sin,
)
from pyquil.quilatom import (
    Function as Function_,
)
from pyquil.quilatom import (
    Mul as Mul_,
)
from pyquil.quilatom import (
    Pow as Pow_,
)
from pyquil.quilatom import (
    Qubit as Qubit_,
)
from pyquil.quilbase import Declare, Gate, Halt, Measurement, Pragma
from pytket.architecture import Architecture
from pytket.circuit import Bit, Circuit, Node, OpType, Qubit

_known_quil_gate = {
    "X": OpType.X,
    "Y": OpType.Y,
    "Z": OpType.Z,
    "H": OpType.H,
    "S": OpType.S,
    "T": OpType.T,
    "RX": OpType.Rx,
    "RY": OpType.Ry,
    "RZ": OpType.Rz,
    "CZ": OpType.CZ,
    "CNOT": OpType.CX,
    "CCNOT": OpType.CCX,
    "CPHASE": OpType.CU1,
    "PHASE": OpType.U1,
    "SWAP": OpType.SWAP,
    "XY": OpType.ISWAP,
    "CH": OpType.CH,
    "CY": OpType.CY,
}


_known_quil_gate_rev = {v: k for k, v in _known_quil_gate.items()}

# Gates with single controlled operation
_single_control_gates = {"CH": "H", "CY": "Y"}


def param_to_pyquil(p: float | Expr) -> float | Expression:
    ppi = p * pi
    if len(ppi.free_symbols) == 0:
        return float(ppi.evalf())

    def to_pyquil(e: Expr) -> float | Expression:  # type: ignore  # noqa: PLR0911
        if isinstance(e, Number):
            return float(e)
        if isinstance(e, Symbol):
            return MemoryReference(str(e))
        if isinstance(e, sin):
            return quil_sin(to_pyquil(e))
        if isinstance(e, cos):
            return quil_cos(to_pyquil(e))
        if isinstance(e, Add):
            args = [to_pyquil(a) for a in e.args]
            acc = args[0]
            for a in args[1:]:
                acc += a
            return acc
        if isinstance(e, Mul):
            args = [to_pyquil(a) for a in e.args]
            acc = args[0]
            for a in args[1:]:
                acc *= a
            return acc
        if isinstance(e, Pow):
            args = Pow_(to_pyquil(e.base), to_pyquil(e.exp))  # type: ignore  # noqa: RET503
        elif e == pi:
            return math.pi
        else:
            raise NotImplementedError(
                "Sympy expression could not be converted to a Quil expression: "
                + str(e)
            )

    return to_pyquil(ppi)


def param_from_pyquil(p: float | Expression) -> Expr:
    def to_sympy(e: Any) -> float | int | Expr | Symbol:  # noqa: PLR0911
        if isinstance(e, (float, int)):  # noqa: UP038
            return e
        if isinstance(e, complex):
            if abs(e.imag) >= 1e-12:  # noqa: PLR2004
                raise NotImplementedError(
                    "Quil expression could not be converted to a parameter: " + str(e)
                )
            return e.real
        if isinstance(e, MemoryReference):
            return Symbol(e.name)
        if isinstance(e, Function_):
            if e.name == "SIN":
                return sin(to_sympy(e.expression))
            if e.name == "COS":
                return cos(to_sympy(e.expression))
            raise NotImplementedError(
                "Quil expression function "
                + e.name
                + " cannot be converted to a sympy expression"
            )
        if isinstance(e, Add_):
            return to_sympy(e.op1) + to_sympy(e.op2)
        if isinstance(e, Sub):
            return to_sympy(e.op1) - to_sympy(e.op2)
        if isinstance(e, Mul_):
            return to_sympy(e.op1) * to_sympy(e.op2)
        if isinstance(e, Div):
            return to_sympy(e.op1) / to_sympy(e.op2)
        if isinstance(e, Pow_):
            return to_sympy(e.op1) ** to_sympy(e.op2)
        raise NotImplementedError(
            "Quil expression could not be converted to a sympy expression: " + str(e)
        )

    return to_sympy(p) / pi


def pyquil_to_tk(prog: Program) -> Circuit:
    """
    Convert a :py:class:`pyquil.Program` to a tket :py:class:`~pytket._tket.circuit.Circuit` .
    Note that not all pyQuil operations are currently supported by pytket.

    :param prog: A circuit to be converted

    :return: The converted circuit
    """
    tkc = Circuit()
    qmap = {}
    for q in prog.get_qubits():
        uid = Qubit("q", q)  # type: ignore
        tkc.add_qubit(uid)
        qmap.update({q: uid})
    cregmap: dict = {}
    for i in prog.instructions:
        if isinstance(i, Gate):
            try:
                optype = _known_quil_gate[i.name]
            except KeyError as error:
                raise NotImplementedError(
                    "Operation not supported by tket: " + str(i)
                ) from error
            qubits = [qmap[cast("Qubit_", q).index] for q in i.qubits]
            params: list[Expr | float] = [param_from_pyquil(p) for p in i.params]  # type: ignore
            tkc.add_gate(optype, params, qubits)
        elif isinstance(i, Measurement):
            qubit = qmap[cast("Qubit_", i.qubit).index]
            reg = cregmap[i.classical_reg.name]  # type: ignore
            bit = reg[i.classical_reg.offset]  # type: ignore
            tkc.Measure(qubit, bit)
        elif isinstance(i, Declare):
            if i.memory_type == "BIT":
                new_reg = tkc.add_c_register(i.name, i.memory_size)
                cregmap.update({i.name: new_reg})
            elif i.memory_type == "REAL":
                continue
            else:
                raise NotImplementedError(
                    "Cannot handle memory of type " + i.memory_type
                )
        elif isinstance(i, Pragma):
            continue
        elif isinstance(i, Halt):
            return tkc
        else:
            raise NotImplementedError("PyQuil instruction is not a gate: " + str(i))
    return tkc


@overload
def tk_to_pyquil(
    tkcirc: Circuit, active_reset: bool = False, return_used_bits: Literal[False] = ...
) -> Program: ...


@overload
def tk_to_pyquil(
    tkcirc: Circuit, active_reset: bool = False, *, return_used_bits: Literal[True]
) -> tuple[Program, list[Bit]]: ...


@overload
def tk_to_pyquil(
    tkcirc: Circuit, active_reset: bool, return_used_bits: Literal[True]
) -> tuple[Program, list[Bit]]: ...


def tk_to_pyquil(  # noqa: PLR0912, PLR0915
    tkcirc: Circuit, active_reset: bool = False, return_used_bits: bool = False
) -> Program | tuple[Program, list[Bit]]:
    """
       Convert a tket :py:class:`~pytket._tket.circuit.Circuit` to a :py:class:`pyquil.Program` .

    :param tkcirc: A circuit to be converted

    :return: The converted circuit
    """
    p = Program()
    qregs = set()
    for qbt in tkcirc.qubits:
        if len(qbt.index) != 1:
            raise NotImplementedError("PyQuil registers must use a single index")
        qregs.add(qbt.reg_name)
    if len(qregs) > 1:
        raise NotImplementedError(
            "Cannot convert circuit with multiple quantum registers to pyQuil"
        )
    creg_sizes: dict = {}
    for b in tkcirc.bits:
        if len(b.index) != 1:
            raise NotImplementedError("PyQuil registers must use a single index")
        if (b.reg_name not in creg_sizes) or (b.index[0] >= creg_sizes[b.reg_name]):
            creg_sizes.update({b.reg_name: b.index[0] + 1})
    cregmap = {}
    for reg_name, size in creg_sizes.items():
        name = reg_name
        if name == "c":
            name = "ro"
        quil_reg = p.declare(name, "BIT", size)
        cregmap.update({reg_name: quil_reg})
    for sym in tkcirc.free_symbols():
        p.declare(str(sym), "REAL")
    if active_reset:
        p.reset()
    measures = []
    measured_qubits: list[Qubit] = []
    used_bits: list[Bit] = []
    for command in tkcirc:
        op = command.op
        optype = op.type
        if optype == OpType.Measure:
            qbt = Qubit_(command.args[0].index[0])  # type: ignore
            if qbt in measured_qubits:
                raise NotImplementedError(
                    "Cannot apply gate on qubit "
                    + qbt.__repr__()
                    + " after measurement"
                )
            bit = cast("Bit", command.args[1])
            b = cregmap[bit.reg_name][bit.index[0]]  # type: ignore
            measures.append(Measurement(qbt, b))  # type: ignore
            measured_qubits.append(qbt)
            used_bits.append(bit)
            continue
        if optype == OpType.Barrier:
            continue  # pyQuil cannot handle barriers
        qubits = [Qubit_(qb.index[0]) for qb in command.args]
        for qbt in qubits:  # type: ignore
            if qbt in measured_qubits:
                raise NotImplementedError(
                    "Cannot apply gate on qubit "
                    + qbt.__repr__()
                    + " after measurement"
                )
        try:
            gatetype = _known_quil_gate_rev[optype]
        except KeyError as error:
            raise NotImplementedError(
                "Cannot convert tket Op to pyQuil gate: " + op.get_name()
            ) from error
        params = [param_to_pyquil(p) for p in op.params]
        if gatetype in _single_control_gates:
            g = Gate(_single_control_gates[gatetype], params, [qubits[1]]).controlled(
                qubits[0]
            )
        else:
            g = Gate(gatetype, params, qubits)

        p += g
    for m in measures:
        p += m
    if return_used_bits:
        return p, used_bits
    return p


def process_characterisation(qc: QuantumComputer) -> dict:  # noqa: PLR0912, PLR0915
    """Convert a :py:class:`pyquil.api.QuantumComputer` to a dictionary containing
    Rigetti device Characteristics

    :param qc: A quantum computer to be converted
    :return: A dictionary containing Rigetti device characteristics
    """
    isa = qc.quantum_processor.to_compiler_isa()
    coupling_map = [(int(e.ids[0]), int(e.ids[1])) for e in isa.edges.values()]

    str_to_gate_1qb = {
        "RX": {
            "PI": OpType.X,
            "PIHALF": OpType.V,
            "-PIHALF": OpType.Vdg,
            "-PI": OpType.X,
            "ANY": OpType.Rx,
        },
        "RZ": {
            "ANY": OpType.Rz,
        },
    }
    str_to_gate_2qb = {"CZ": OpType.CZ, "XY": OpType.ISWAP}

    link_errors: dict[tuple[Node, Node], dict[OpType, float]] = defaultdict(dict)
    node_errors: dict[Node, dict[OpType, float]] = defaultdict(dict)
    readout_errors: dict = {}
    # T1s and T2s are currently left empty
    t1_times_dict: dict = {}
    t2_times_dict: dict = {}

    for q in isa.qubits.values():
        node = Node(q.id)
        for g in q.gates:
            if g.fidelity is None:
                g.fidelity = 1.0
            if isinstance(g, GateInfo) and g.operator in str_to_gate_1qb:
                angle = _get_angle_type(g.parameters[0])
                if angle is not None:
                    try:
                        optype = str_to_gate_1qb[g.operator][angle]
                    except KeyError:
                        warning(  # noqa: LOG015
                            f"Ignoring unrecognised angle {g.parameters[0]} "  # noqa: G004
                            f"for gate {g.operator}. This may mean that some "
                            "hardware-supported gates won't be used."
                        )
                        continue
                    if node in node_errors and optype in node_errors[node]:
                        if abs(1.0 - g.fidelity - node_errors[node][optype]) > 1e-7:  # noqa: PLR2004
                            # fidelities for Rx(PI) and Rx(-PI) are given, hopefully
                            # they are always identical
                            warning(  # noqa: LOG015
                                f"Found two differing fidelities for {optype} on node "  # noqa: G004
                                f"{node}, using error = {node_errors[node][optype]}"
                            )
                    else:
                        node_errors[node].update({optype: 1.0 - g.fidelity})
            elif isinstance(g, MeasureInfo) and g.operator == "MEASURE":
                # for some reason, there are typically two MEASURE entries,
                # one with target="_", and one with target=Node
                # in all pyquil code I have seen, both have the same value
                if node in readout_errors:
                    if abs(1.0 - g.fidelity - readout_errors[node]) > 1e-7:  # noqa: PLR2004
                        warning(  # noqa: LOG015
                            f"Found two differing readout fidelities for node {node},"  # noqa: G004
                            f" using RO error = {readout_errors[node]}"
                        )
                else:
                    readout_errors[node] = 1.0 - g.fidelity
            elif g.operator == "I":
                continue
            else:
                warning(f"Ignoring fidelity for unknown operator {g.operator}")  # noqa: LOG015, G004

    for e in isa.edges.values():
        n1, n2 = Node(e.ids[0]), Node(e.ids[1])
        for g in e.gates:
            if g.fidelity is None:
                g.fidelity = 1.0
            if g.operator in str_to_gate_2qb:
                optype = str_to_gate_2qb[g.operator]
                link_errors[(n1, n2)].update({optype: 1.0 - g.fidelity})
            else:
                warning(f"Ignoring fidelity for unknown operator {g.operator}")  # noqa: LOG015, G004

    arc = Architecture(coupling_map)

    characterisation = dict()  # noqa: C408
    characterisation["NodeErrors"] = node_errors
    characterisation["EdgeErrors"] = link_errors  # type: ignore
    characterisation["Architecture"] = arc  # type: ignore
    characterisation["t1times"] = t1_times_dict
    characterisation["t2times"] = t2_times_dict

    return characterisation


def _get_angle_type(angle: float | str) -> str | None:
    if angle == "theta":
        return "ANY"
    angles = {pi: "PI", pi / 2: "PIHALF", 0: None, -pi / 2: "-PIHALF", -pi: "-PI"}
    if not isinstance(angle, str):
        for val, code in angles.items():
            if abs(angle - val) < 1e-7:  # noqa: PLR2004
                return code
    warning(  # noqa: LOG015
        f"Ignoring unrecognised angle {angle}. This may mean that some "  # noqa: G004
        "hardware-supported gates won't be used."
    )
    return None


def get_avg_characterisation(
    characterisation: dict[str, Any],
) -> dict[str, dict[Node, float]]:
    """
    Convert gate-specific characterisation into readout, one- and two-qubit errors

    Used to convert a typical output from `process_characterisation` into an input
    noise characterisation for NoiseAwarePlacement
    """

    K = TypeVar("K")
    V1 = TypeVar("V1")
    V2 = TypeVar("V2")
    map_values_t = Callable[[Callable[[V1], V2], dict[K, V1]], dict[K, V2]]
    map_values: map_values_t = lambda f, d: {k: f(v) for k, v in d.items()}

    node_errors = cast(
        "dict[Node, dict[OpType, float]]", characterisation["NodeErrors"]
    )
    link_errors = cast(
        "dict[tuple[Node, Node], dict[OpType, float]]", characterisation["EdgeErrors"]
    )

    avg: Callable[[dict[Any, float]], float] = lambda xs: sum(xs.values()) / len(xs)
    avg_node_errors = map_values(avg, node_errors)
    avg_link_errors = map_values(avg, link_errors)

    return {
        "node_errors": avg_node_errors,
        "link_errors": avg_link_errors,
    }
