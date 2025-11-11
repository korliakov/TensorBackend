# TensorBackend

A Python-based quantum circuit simulator that provides tools for simulating quantum circuits, gates, and channels using density matrix formalism.

## Overview

This project implements a comprehensive quantum computing simulation framework with:

- **Custom circuit representation** for building quantum circuits
- **Unsigned tableau formalism** for Clifford circuit simulation
- **Density matrix simulation** with tensor operations
- **Quantum gate and channel implementations**
- **Random circuit generation** and state manipulation

## Features

### Clifford Circuit Simulation
- Unsigned tableau representation for efficient Clifford simulation
- Random Clifford circuit generation
- Pauli string manipulation and commutation checking

### Density Matrix Simulation
- Tensor-based density matrix operations for efficiency
- Unitary and non-unitary evolution
- Partial trace and state reduction


## Requirements

 - Python 3.6+
 - NumPy


## Quick Start

```
from unsigned_tableau import UnsignedTableau
from custom_circuit import CustomCircuit
from gates_channels import H
from dm_simulation import gate_action, generate_random_dm_matrix
from gates_channels import Kraus_depolarizing_channel
from dm_simulation import channel_action

# Generate a random 3-qubit Clifford circuit
clifford_circuit = UnsignedTableau.generate_random_Clifford(3)

# Create a custom circuit
circuit = CustomCircuit()
circuit.add_gate(['H', 0])
circuit.add_gate(['CNOT', [0, 1]])

# Simulate quantum circuit with density matrices using tensor contractions
dm = generate_random_dm_matrix(2)
dm_after_h = gate_action(H(), dm, [0])


# Apply depolarizing noise
channel = Kraus_depolarizing_channel(0.1)
dm_with_noise = channel_action(channel, dm, [0])
```


