# Crate: **simple_mpi**

A simple MPI (Message Passing Interface) implementation for single-machine shared memory communication. This library provides MPI-like functionality for parallel computing within a single machine using POSIX shared memory. It enables efficient inter-process communication through operations like point-to-point messaging, collective operations, and shared memory arrays.

## Module: **examples.basic**

**Module Attributes:**
- `world.rank() == 0`()
- `world.rank() == 1`()
- `/dev/shm`()

## Module: **examples.shared_array**

### Structs
**Point** (visibility: ``)



**Attributes:**
- `C`()
- `derive`(Debug, Copy, Clone, Pod, Zeroable)

**Fields:**
- **x**: *f32*  

  Visibility: ``  

  
- **y**: *f32*  

  Visibility: ``  

  
- **z**: *f32*  

  Visibility: ``  

  

---
## Module: **src**

A simple MPI (Message Passing Interface) implementation for single-machine shared memory communication. This library provides MPI-like functionality for parallel computing within a single machine using POSIX shared memory. It enables efficient inter-process communication through operations like point-to-point messaging, collective operations, and shared memory arrays.

**Module Attributes:**
- `crate-level`()
- `features`()
- `quick_start`()
- `architecture`()
- `error_handling`()

### Structs
**SharedReadOnlyArray** (visibility: ``)

A shared read-only array that can be accessed by all processes. T must be Pod (plain old data) to ensure it contains no pointers/references.

**Fields:**
- **shmem_idx**: *usize*  

  Visibility: ``  

  
- **_phantom**: *PhantomData<T>*  

  Visibility: ``  

  

**Methods:**
- **as_slice**() -> &[T]

  Visibility: ``

  Get a slice reference to the array. Safe because T is Pod (contains no pointers/references).

  **Examples:**
```rust
#  Get a slice reference to the array.
let shared_array: SharedReadOnlyArray<Point> = ...;
let slice: &[Point] = shared_array.as_slice();
```

---
**World** (visibility: ``)

The main MPI World structure that manages inter-process communication.

**Fields:**
- **rank**: *i32*  

  Visibility: `private`  

  The rank of the process within the world.
- **size**: *i32*  

  Visibility: `private`  

  The total number of processes in the world.
- **shmem_idx**: *usize*  

  Visibility: `private`  

  Index for the shared memory segment associated with this world.

**Methods:**
- **init**(size: i32) -> Result<Self>

  Visibility: `public`

  Initialize the MPI environment and create a new World.

  **Examples:**
```rust
# Initialize the MPI environment and create a new World.
let world = World::init(4).unwrap();
println!("Process {} of {} initialized", world.rank(), world.size());
world.destruct();
```

- **rank**() -> i32

  Visibility: `public`

  Get the rank (unique identifier) of the current process.

  **Examples:**
```rust
# Get the rank (unique identifier) of the current process.
let world = ...;
let rank = world.rank();
```

- **size**() -> i32

  Visibility: `public`

  Get the total number of processes in the World.

  **Examples:**
```rust
# Get the total number of processes in the World.
let world = ...;
let size = world.size();
```

- **barrier**() -> Result<()>

  Visibility: `private`

  Barrier synchronization - blocks until all processes reach this point

  **Examples:**
```rust
# Barrier synchronization
let world = ...;
world.barrier().unwrap();
```

- **send**(data: &T, dest: i32, tag: i32) -> Result<()>

  Visibility: `public`

  Send data to a specific rank (blocking until received).

  **Examples:**
```rust
# Send data to a specific rank (blocking until received).
let world = World::init(2).unwrap();
if world.rank() == 0 {
    let data = vec![1, 2, 3];
    world.send(&data, 1, 0).unwrap();
}
world.destruct();
```

- **recv**(source: i32, tag: i32) -> Result<T>

  Visibility: `public`

  Receive data from a specific rank.

  **Examples:**
```rust
# Receive data from a specific rank.
let world = World::init(2).unwrap();
if world.rank() == 1 {
    let data: Vec<i32> = world.recv(0, 0).unwrap();
    println!("Received: {:?}", data);
}
world.destruct();
```

- **broadcast**(data: &T, root: i32) -> Result<T>

  Visibility: `public`

  Broadcast data from root rank to all other ranks.

  **Examples:**
```rust
# Broadcast data from root rank to all other ranks.
let data = if world.rank() == 0 {
    vec![1, 2, 3]
} else {
    vec![]
};
let result = world.broadcast(&data, 0).unwrap();
```

- **scatter**(data: Option<&[T]>, root: i32) -> Result<T>

  Visibility: `public`

  Scatter data from root rank to all ranks.

  **Examples:**
```rust
# Scatter data from root rank to all ranks.
let data = if world.rank() == 0 {
    Some(&[10, 20, 30, 40][..])
} else {
    None
};
let my_number = world.scatter(data, 0).unwrap();
println!("Process {} got {}", world.rank(), my_number);
```

- **gather**(data: &T, root: i32) -> Result<Option<Vec<T>>>

  Visibility: `public`

  Gather data from all ranks to root rank.

  **Examples:**
```rust
# Gather data from all ranks to root rank.
let result = world.gather(&world.rank(), 0).unwrap();
if world.rank() == 0 {
    println!("Gathered: {:?}", result.unwrap());
}
```

- **destruct**() -> ()

  Visibility: `public`

  Explicitly clean up MPI resources and synchronize process shutdown.

  **Examples:**
```rust
# Explicitly clean up MPI resources and synchronize process shutdown.
let world = World::init(2).unwrap();
//... use the world for communication ...
world.destruct(); // Clean up resources
```

---
### Enums
**MPIError** (visibility: ``)



**Attributes:**
- `derive`(Error, Debug)

**Variants:**
- **InvalidRank**  

  Invalid rank provided.
- **CommunicationError**  

  Communication error occurred.
- **InitError**  

  Error during initialization.
- **SerializationError**  

  Serialization error occurred.
- **SharedMemoryError**  

  Error related to shared memory.
- **ProcessError**  

  Error related to processes.

---