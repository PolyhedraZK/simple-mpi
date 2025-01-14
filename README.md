# Simple MPI

A lightweight MPI (Message Passing Interface) implementation for single-machine shared memory communication in Rust. This library provides basic MPI-like functionality using POSIX shared memory for efficient inter-process communication.

## Features

- **Process Management**: Spawn and manage multiple processes with unique ranks
- **Point-to-Point Communication**: Send and receive messages between processes
- **Collective Operations**: 
  - Broadcast: Distribute data from one process to all processes
  - Scatter: Distribute different pieces of data from one process to all processes
  - Gather: Collect data from all processes to one process
- **Shared Memory Arrays**: Create read-only arrays accessible by all processes
- **Safe Memory Management**: Automatic cleanup of shared memory resources
- **Type Safety**: Serialization and deserialization of Rust types
- **Error Handling**: Comprehensive error types for robust error management

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
simple-mpi = "0.1.0"
```

## Usage

### Basic Example

```rust
use simple_mpi::World;

fn main() {
    // Initialize MPI with 4 processes
    let world = World::init(4).unwrap();
    println!("Process {} initialized", world.rank());

    // Send/Receive Example
    if world.rank() == 0 {
        let message = "Hello from rank 0";
        world.send(&message, 1, 0).unwrap();
    } else if world.rank() == 1 {
        let message: String = world.recv(0, 0).unwrap();
        println!("Rank 1 received: {}", message);
    }

    // Broadcast Example
    let data = if world.rank() == 0 {
        vec![1, 2, 3, 4]
    } else {
        Vec::new()
    };
    let result = world.broadcast(&data, 0).unwrap();
    println!("Rank {} received broadcast: {:?}", world.rank(), result);

    // Clean up
    world.destruct();
}
```

### Shared Memory Arrays

```rust
use simple_mpi::World;
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
struct Point {
    x: f32,
    y: f32,
    z: f32,
}

fn main() {
    let world = World::init(4).unwrap();

    if world.rank() == 0 {
        let points = vec![
            Point { x: 1.0, y: 2.0, z: 3.0 },
            Point { x: 4.0, y: 5.0, z: 6.0 },
        ];
        let shared = world.shared_readonly_array(Some(&points), 0).unwrap();
        // Array is now accessible by all processes
    } else {
        let shared = world.shared_readonly_array::<Point>(None, 0).unwrap();
        let view = shared.as_slice();
        // Read access to the shared array
    }

    world.destruct();
}
```

## API Documentation

### World

The main MPI environment structure that manages inter-process communication.

```rust
// Initialize MPI environment
World::init(size: i32) -> Result<World>

// Get current process rank
World::rank(&self) -> i32

// Get total number of processes
World::size(&self) -> i32

// Point-to-Point Communication
World::send<T: Serialize>(&self, data: &T, dest: i32, tag: i32) -> Result<()>
World::recv<T: DeserializeOwned>(&self, source: i32, tag: i32) -> Result<T>

// Collective Operations
World::broadcast<T>(&self, data: &T, root: i32) -> Result<T>
World::scatter<T>(&self, data: Option<&[T]>, root: i32) -> Result<T>
World::gather<T>(&self, data: &T, root: i32) -> Result<Option<Vec<T>>>

// Shared Memory Arrays
World::shared_readonly_array<T: Pod>(&self, data: Option<&[T]>, source: i32) 
    -> Result<SharedReadOnlyArray<T>>

// Cleanup
World::destruct(self)
```

### SharedReadOnlyArray

A shared read-only array that can be accessed by all processes.

```rust
SharedReadOnlyArray::as_slice(&self) -> &[T]
```

## Error Handling

The library provides comprehensive error types through `MPIError`:

- `InvalidRank`: When an invalid process rank is specified
- `CommunicationError`: For communication-related failures
- `InitError`: For initialization failures
- `SerializationError`: For serialization/deserialization errors
- `SharedMemoryError`: For shared memory operation failures
- `ProcessError`: For process management failures

## License

This project is licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later).
