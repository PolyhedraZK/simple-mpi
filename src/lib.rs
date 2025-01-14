//! A simple MPI (Message Passing Interface) implementation for single-machine shared memory communication.
//! 
//! This library provides MPI-like functionality for parallel computing within a single machine using POSIX shared memory.
//! It enables efficient inter-process communication through operations like point-to-point messaging, collective
//! operations, and shared memory arrays.
//!
//! # Features
//!
//! - **Process Management**: Spawn and coordinate multiple processes
//! - **Point-to-Point Communication**: Send and receive messages between processes
//! - **Collective Operations**: Broadcast, scatter, and gather data across processes
//! - **Shared Memory Arrays**: Share read-only arrays between processes efficiently
//! - **Synchronization**: Barrier operations for process synchronization
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use simple_mpi::World;
//!
//! // Initialize MPI with 4 processes
//! let world = World::init(4).unwrap();
//!
//! // Get process rank and size
//! println!("Process {} of {}", world.rank(), world.size());
//!
//! // Example: Root process broadcasts data to all others
//! let data = if world.rank() == 0 { 42 } else { 0 };
//! let result = world.broadcast(&data, 0).unwrap();
//! println!("Process {} received {}", world.rank(), result);
//!
//! // Clean up MPI resources
//! world.destruct();
//! ```
//!
//! # Architecture
//!
//! The library uses POSIX shared memory (`/dev/shm` on Linux) to establish communication channels between processes.
//! Each process pair gets a dedicated memory slot for message passing, and additional shared memory segments are
//! used for shared arrays and synchronization.
//!
//! # Error Handling
//!
//! All operations return a `Result` type with detailed error variants through [`MPIError`].
//! Common errors include invalid ranks, communication failures, and shared memory issues.

use shared_memory::{Shmem, ShmemConf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::process::{Command, Stdio};
use serde::{de::DeserializeOwned, Serialize};
use thiserror::Error;
use log::debug;
use std::marker::PhantomData;
use bytemuck::Pod;

// Store shared memory to prevent it from being dropped
static mut SHARED_MEMORY: Vec<Shmem> = Vec::new();

const MAX_MSG_SIZE: usize = 1024 * 1024; // 1MB max message size
const HEADER_SIZE: usize = std::mem::size_of::<MessageHeader>();
const SHM_NAME: &str = "simple_mpi";

#[derive(Error, Debug)]
pub enum MPIError {
    #[error("Invalid rank: {0}")]
    InvalidRank(i32),
    #[error("Communication error: {0}")]
    CommunicationError(String),
    #[error("Initialization error: {0}")]
    InitError(String),
    #[error("Serialization error: {0}")]
    SerializationError(String),
    #[error("Shared memory error: {0}")]
    SharedMemoryError(String),
    #[error("Process error: {0}")]
    ProcessError(String),
}

pub type Result<T> = std::result::Result<T, MPIError>;

#[repr(C)]
#[derive(Default)]
struct MessageHeader {
    valid: AtomicBool,
    received: AtomicBool,
    tag: i32,
    source: i32,
    size: usize,
}

impl MessageHeader {
    fn new() -> Self {
        Self {
            valid: AtomicBool::new(false),
            received: AtomicBool::new(false),
            tag: 0,
            source: -1,
            size: 0,
        }
    }
}

#[repr(C)]
struct SharedState {
    size: AtomicUsize,
    initialized: AtomicBool,
    process_ids: [AtomicUsize; 32],  // Process IDs for each rank
    init_flags: [AtomicBool; 32],    // Flags for initialization synchronization
    done_flags: [AtomicBool; 32],    // Flags for shutdown synchronization
    root_done: AtomicBool,          // Flag for root process completion
    cleanup_ready: [AtomicBool; 32], // Flags for cleanup synchronization
    cleanup_done: AtomicBool,       // Flag for cleanup completion
}

#[repr(C)]
struct SharedArrayHeader {
    ready: AtomicBool,
    len: usize,
}

/// A shared read-only array that can be accessed by all processes.
/// T must be Pod (plain old data) to ensure it contains no pointers/references.
pub struct SharedReadOnlyArray<T: Pod> {
    shmem_idx: usize,
    _phantom: PhantomData<T>,
}

impl<T: Pod> Drop for SharedReadOnlyArray<T> {
    fn drop(&mut self) {
        // Ensure memory barrier before cleanup
        std::sync::atomic::fence(Ordering::SeqCst);
    }
}

impl<T: Pod> SharedReadOnlyArray<T> {
    /// Get a slice reference to the array.
    /// Safe because T is Pod (contains no pointers/references).
    pub fn as_slice(&self) -> &[T] {
        unsafe {
            let shmem = &SHARED_MEMORY[self.shmem_idx];
            let header = &*(shmem.as_ptr() as *const SharedArrayHeader);
            let data_ptr = shmem.as_ptr()
                .add(std::mem::size_of::<SharedArrayHeader>()) as *const T;
            std::slice::from_raw_parts(data_ptr, header.len)
        }
    }
}

/// The main MPI World structure that manages inter-process communication.
///
/// `World` represents the collection of all processes in the MPI computation. It provides
/// methods for process identification, communication, and synchronization.
///
/// # Examples
///
/// Basic point-to-point communication:
/// ```rust,no_run
/// use simple_mpi::World;
///
/// let world = World::init(2).unwrap();
///
/// if world.rank() == 0 {
///     // Send data from process 0 to process 1
///     world.send(&42, 1, 0).unwrap();
/// } else {
///     // Receive data in process 1
///     let data: i32 = world.recv(0, 0).unwrap();
///     println!("Received: {}", data);
/// }
///
/// world.destruct();
/// ```
///
/// Collective operation (broadcast):
/// ```rust,no_run
/// use simple_mpi::World;
///
/// let world = World::init(4).unwrap();
///
/// // Process 0 broadcasts data to all processes
/// let data = if world.rank() == 0 { vec![1, 2, 3] } else { vec![] };
/// let result = world.broadcast(&data, 0).unwrap();
///
/// assert_eq!(result, vec![1, 2, 3]);
/// world.destruct();
/// ```
pub struct World {
    rank: i32,
    size: i32,
    shmem_idx: usize,
}

impl World {
    /// Initialize the MPI environment and create a new World.
    ///
    /// This function spawns the specified number of processes and establishes shared memory
    /// communication channels between them. It blocks until all processes are initialized
    /// and ready for communication.
    ///
    /// # Arguments
    ///
    /// * `size` - The total number of processes to create (must be positive)
    ///
    /// # Returns
    ///
    /// * `Ok(World)` - A new World instance if initialization succeeds
    /// * `Err(MPIError)` - If initialization fails (e.g., invalid size, shared memory error)
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use simple_mpi::World;
    ///
    /// // Create a world with 4 processes
    /// let world = World::init(4).unwrap();
    /// println!("Process {} of {} initialized", world.rank(), world.size());
    ///
    /// world.destruct();
    /// ```
    pub fn init(size: i32) -> Result<Self> {
        if size <= 0 {
            return Err(MPIError::InitError("Size must be positive".into()));
        }

        // Check if we're a spawned process
        if let Ok(_) = std::env::var("MPI_SPAWNED") {
            return Self::init_spawned(size);
        }

        debug!("Initializing MPI with {} processes", size);

        // Calculate sizes - need a slot for each possible communication pair
        let state_size = std::mem::size_of::<SharedState>();
        let total_slots = size * size; // One slot for each possible src->dst pair
        let total_size = state_size + (total_slots as usize * (HEADER_SIZE + MAX_MSG_SIZE));

        debug!("Creating shared memory of size {}", total_size);

        // Create shared memory
        let shmem = ShmemConf::new()
            .size(total_size)
            .flink(SHM_NAME)
            .create()
            .map_err(|e| MPIError::SharedMemoryError(e.to_string()))?;

        debug!("shmem path {:?}", shmem.get_flink_path());

        unsafe {
            // Initialize shared state
            let state = &mut *(shmem.as_ptr() as *mut SharedState);
            state.size.store(size as usize, Ordering::SeqCst);
            state.initialized.store(true, Ordering::SeqCst);
            state.root_done.store(false, Ordering::SeqCst);
            state.cleanup_done.store(false, Ordering::SeqCst);
            for i in 0..size as usize {
                state.process_ids[i].store(0, Ordering::SeqCst);
                state.init_flags[i].store(false, Ordering::SeqCst);
                state.done_flags[i].store(false, Ordering::SeqCst);
                state.cleanup_ready[i].store(false, Ordering::SeqCst);
            }
            // Store root process ID at rank 0
            state.process_ids[0].store(std::process::id() as usize, Ordering::SeqCst);
            state.init_flags[0].store(true, Ordering::SeqCst);  // Mark rank 0 as initialized

            // Initialize message headers for all communication pairs
            let base_ptr = shmem.as_ptr().add(state_size);
            for i in 0..total_slots {
                let header_ptr = base_ptr.add(i as usize * (HEADER_SIZE + MAX_MSG_SIZE)) as *mut MessageHeader;
                std::ptr::write(header_ptr, MessageHeader::new());
            }

            // Store in global Vec
            SHARED_MEMORY.push(shmem);
            let shmem_idx = SHARED_MEMORY.len() - 1;

            // Now spawn child processes with inherited stdio
            for _ in 1..size {
                Command::new(std::env::current_exe().unwrap())
                    .env("MPI_SPAWNED", "1")
                    .stdout(Stdio::inherit())
                    .stderr(Stdio::inherit())
                    .spawn()
                    .map_err(|e| MPIError::ProcessError(e.to_string()))?;
            }

            debug!("Parent process initialized as rank 0");

            // Wait for all processes to initialize
            let state = &*(SHARED_MEMORY[shmem_idx].as_ptr() as *const SharedState);
            let mut all_init = false;
            while !all_init {
                all_init = true;
                for i in 0..size {
                    if !state.init_flags[i as usize].load(Ordering::SeqCst) {
                        all_init = false;
                        std::hint::spin_loop();
                        break;
                    }
                }
            }

            Ok(World { 
                rank: 0, 
                size,
                shmem_idx,
            })
        }
    }

    /// Initialize as a spawned process
    fn init_spawned(size: i32) -> Result<Self> {
        debug!("Spawned process initializing");
        
        // Open existing shared memory
        let shmem = ShmemConf::new()
            .flink(SHM_NAME)
            .open()
            .map_err(|e| MPIError::SharedMemoryError(e.to_string()))?;

        unsafe {
            SHARED_MEMORY.push(shmem);
            let shmem_idx = SHARED_MEMORY.len() - 1;

            let state = &mut *(SHARED_MEMORY[shmem_idx].as_ptr() as *mut SharedState);

            // Find an available rank and store our process ID
            let pid = std::process::id() as usize;
            let mut rank = -1;
            for i in 1..size {
                // Try to claim this rank by atomically setting process ID from 0 to our PID
                if state.process_ids[i as usize]
                    .compare_exchange(0, pid, Ordering::SeqCst, Ordering::SeqCst)
                    .is_ok() {
                    rank = i;
                    debug!("Process {} initialized as rank {}", pid, rank);
                    break;
                }
            }

            if rank == -1 {
                return Err(MPIError::InitError(format!("No available ranks for process {}", pid).into()));
            }

            // Mark as initialized and wait for all processes
            state.init_flags[rank as usize].store(true, Ordering::SeqCst);

            let mut all_init = false;
            while !all_init {
                all_init = true;
                for i in 0..size {
                    if !state.init_flags[i as usize].load(Ordering::SeqCst) {
                        all_init = false;
                        std::hint::spin_loop();
                        break;
                    }
                }
            }

            debug!("Process initialized as rank {}", rank);
            Ok(World { 
                rank, 
                size, 
                shmem_idx,
            })
        }
    }

    /// Create a shared read-only array that can be accessed by all processes.
    /// T must be Pod (plain old data) to ensure it contains no pointers/references.
    pub fn shared_readonly_array<T: Pod>(&self, data: Option<&[T]>, source: i32) -> Result<SharedReadOnlyArray<T>> {
        if source >= self.size {
            return Err(MPIError::InvalidRank(source));
        }

        // Generate a unique name for this shared array
        let array_id = if self.rank == source {
            std::process::id()
        } else {
            0 // Will be overwritten by broadcast
        };
        
        // Synchronize array_id across all processes before creating/opening the file
        let array_id = self.broadcast(&array_id, source)?;
        let shm_name = format!("simple_mpi_array_{}", array_id);
        debug!("Rank {}, Shared array name: {}", self.rank(), shm_name);

        if self.rank == source {
            // Source process creates and initializes the shared memory
            let data = data.ok_or_else(|| 
                MPIError::InitError("Source must provide data".into())
            )?;

            let header_size = std::mem::size_of::<SharedArrayHeader>();
            let total_size = header_size + std::mem::size_of::<T>() * data.len();

            // Create shared memory
            let shmem = ShmemConf::new()
                .size(total_size)
                .flink(&shm_name)
                .create()
                .map_err(|e| MPIError::SharedMemoryError(e.to_string()))?;

            unsafe {
                // Store in global Vec
                SHARED_MEMORY.push(shmem);
                let shmem_idx = SHARED_MEMORY.len() - 1;

                debug!("shmem path {:?}", SHARED_MEMORY[shmem_idx].get_flink_path());
                // Barrier to ensure all processes have time to open the file
                self.barrier()?;

                // Initialize header
                let header = &mut *(SHARED_MEMORY[shmem_idx].as_ptr() as *mut SharedArrayHeader);
                header.ready.store(false, Ordering::SeqCst);
                header.len = data.len();

                // Copy data
                let data_ptr = SHARED_MEMORY[shmem_idx].as_ptr().add(header_size) as *mut T;
                std::ptr::copy_nonoverlapping(data.as_ptr(), data_ptr, data.len());

                // Signal ready
                header.ready.store(true, Ordering::SeqCst);
                
                Ok(SharedReadOnlyArray {
                    shmem_idx,
                    _phantom: PhantomData,
                })
            }
        } else {
            // Barrier to ensure source doesn't exit too early
            self.barrier()?;
            // Non-source processes open the shared memory
            let shmem = ShmemConf::new()
                .flink(&shm_name)
                .open()
                .map_err(|e| MPIError::SharedMemoryError(e.to_string()))?;

            unsafe {
                // Store in global Vec
                SHARED_MEMORY.push(shmem);
                let shmem_idx = SHARED_MEMORY.len() - 1;

                // Wait for ready flag
                let header = &*(SHARED_MEMORY[shmem_idx].as_ptr() as *const SharedArrayHeader);
                while !header.ready.load(Ordering::SeqCst) {
                    std::hint::spin_loop();
                }

                Ok(SharedReadOnlyArray {
                    shmem_idx,
                    _phantom: PhantomData,
                })
            }
        }
    }

    /// Get the rank (unique identifier) of the current process.
    ///
    /// The rank is a number between 0 and `size()-1` that uniquely identifies each process.
    /// Process rank 0 is typically considered the "root" process for collective operations.
    ///
    /// # Returns
    ///
    /// An integer representing this process's rank
    pub fn rank(&self) -> i32 {
        self.rank
    }

    /// Get the total number of processes in the World.
    ///
    /// This value remains constant throughout the lifetime of the World and matches
    /// the size parameter passed to [`World::init()`].
    ///
    /// # Returns
    ///
    /// The total number of processes
    pub fn size(&self) -> i32 {
        self.size
    }

    fn get_slot_ptr(&self, src: i32, dst: i32) -> *mut u8 {
        let state_size = std::mem::size_of::<SharedState>();
        let slot_size = HEADER_SIZE + MAX_MSG_SIZE;
        let slot_index = src * self.size + dst; // Each src->dst pair gets its own slot
        unsafe {
            SHARED_MEMORY[self.shmem_idx].as_ptr().add(state_size + (slot_index as usize * slot_size))
        }
    }

    /// Barrier synchronization - blocks until all processes reach this point
    fn barrier(&self) -> Result<()> {
        const BARRIER_ARRIVE_TAG: i32 = -1;
        const BARRIER_COMPLETE_TAG: i32 = -2;

        debug!("Rank {} entering barrier", self.rank);

        // First phase: everyone sends to root
        if self.rank == 0 {
            for rank in 1..self.size {
                debug!("Root waiting for rank {}", rank);
                self.recv::<()>(rank, BARRIER_ARRIVE_TAG)?;
            }
        } else {
            debug!("Rank {} signaling root", self.rank);
            self.send(&(), 0, BARRIER_ARRIVE_TAG)?;
        }

        // Second phase: root broadcasts completion
        if self.rank == 0 {
            for rank in 1..self.size {
                self.send(&(), rank, BARRIER_COMPLETE_TAG)?;
            }
        } else {
            self.recv::<()>(0, BARRIER_COMPLETE_TAG)?;
        }

        debug!("Rank {} exiting barrier", self.rank);
        Ok(())
    }

    /// Send data to a specific rank (blocking until received).
    ///
    /// This function serializes the data and sends it to the specified destination process.
    /// It blocks until the receiving process has acknowledged receipt of the message.
    ///
    /// # Arguments
    ///
    /// * `data` - The data to send (must implement Serialize)
    /// * `dest` - The rank of the destination process
    /// * `tag` - A message identifier (useful for matching specific sends/receives)
    ///
    /// # Returns
    ///
    /// * `Ok(())` - If the send succeeds
    /// * `Err(MPIError)` - If the send fails (e.g., invalid rank, serialization error)
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use simple_mpi::World;
    ///
    /// let world = World::init(2).unwrap();
    ///
    /// if world.rank() == 0 {
    ///     let data = vec![1, 2, 3];
    ///     world.send(&data, 1, 0).unwrap();
    /// }
    ///
    /// world.destruct();
    /// ```
    pub fn send<T: Serialize>(&self, data: &T, dest: i32, tag: i32) -> Result<()> {
        if dest >= self.size {
            return Err(MPIError::InvalidRank(dest));
        }

        debug!("Rank {} sending to rank {} with tag {}", self.rank, dest, tag);

        let serialized = bincode::serialize(data)
            .map_err(|e| MPIError::SerializationError(e.to_string()))?;

        if serialized.len() > MAX_MSG_SIZE {
            return Err(MPIError::CommunicationError("Message too large".into()));
        }

        let slot_ptr = self.get_slot_ptr(self.rank, dest);
        let header = unsafe { &mut *(slot_ptr as *mut MessageHeader) };
        
        // Wait for slot to be free
        while header.valid.load(Ordering::SeqCst) {
            std::hint::spin_loop();
        }

        // Write message
        unsafe {
            header.tag = tag;
            header.source = self.rank;
            header.size = serialized.len();
            header.received.store(false, Ordering::SeqCst);
            
            let data_ptr = slot_ptr.add(HEADER_SIZE);
            std::ptr::copy_nonoverlapping(serialized.as_ptr(), data_ptr, serialized.len());
            
            // Mark message as valid and wait for receiver to acknowledge
            header.valid.store(true, Ordering::SeqCst);
            
            debug!("Rank {} waiting for acknowledgment from rank {}", self.rank, dest);
            // Block until receiver acknowledges
            while !header.received.load(Ordering::SeqCst) {
                std::hint::spin_loop();
            }
        }

        debug!("Rank {} completed send to rank {}", self.rank, dest);
        Ok(())
    }

    /// Receive data from a specific rank.
    ///
    /// This function blocks until a message with the specified tag arrives from the source process.
    /// The received data is deserialized into the specified type.
    ///
    /// # Arguments
    ///
    /// * `source` - The rank of the sending process
    /// * `tag` - The message identifier to match
    ///
    /// # Returns
    ///
    /// * `Ok(T)` - The received and deserialized data
    /// * `Err(MPIError)` - If the receive fails (e.g., invalid rank, deserialization error)
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use simple_mpi::World;
    ///
    /// let world = World::init(2).unwrap();
    ///
    /// if world.rank() == 1 {
    ///     let data: Vec<i32> = world.recv(0, 0).unwrap();
    ///     println!("Received: {:?}", data);
    /// }
    ///
    /// world.destruct();
    /// ```
    pub fn recv<T: DeserializeOwned>(&self, source: i32, tag: i32) -> Result<T> {
        if source >= self.size {
            return Err(MPIError::InvalidRank(source));
        }

        debug!("Rank {} receiving from rank {} with tag {}", self.rank, source, tag);

        let slot_ptr = self.get_slot_ptr(source, self.rank);
        let header = unsafe { &mut *(slot_ptr as *mut MessageHeader) };

        // Wait for valid message
        while !header.valid.load(Ordering::SeqCst) 
            || header.source != source 
            || header.tag != tag 
        {
            std::hint::spin_loop();
        }

        // Read message
        let result = unsafe {
            let data_ptr = slot_ptr.add(HEADER_SIZE);
            let data = std::slice::from_raw_parts(data_ptr, header.size);
            
            bincode::deserialize(data)
                .map_err(|e| MPIError::SerializationError(e.to_string()))?
        };

        // Acknowledge receipt and mark slot as free
        header.received.store(true, Ordering::SeqCst);
        header.valid.store(false, Ordering::SeqCst);

        debug!("Rank {} completed receive from rank {}", self.rank, source);
        Ok(result)
    }

    /// Broadcast data from root rank to all other ranks.
    ///
    /// The root process sends its data to all other processes. This is a collective
    /// operation that must be called by all processes with the same root rank.
    ///
    /// # Arguments
    ///
    /// * `data` - The data to broadcast (only used by root process)
    /// * `root` - The rank of the broadcasting process
    ///
    /// # Returns
    ///
    /// * `Ok(T)` - The broadcast data (same for all processes)
    /// * `Err(MPIError)` - If the broadcast fails
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use simple_mpi::World;
    ///
    /// let world = World::init(4).unwrap();
    ///
    /// // Root broadcasts a vector to all processes
    /// let data = if world.rank() == 0 {
    ///     vec![1, 2, 3]
    /// } else {
    ///     vec![]
    /// };
    ///
    /// let result = world.broadcast(&data, 0).unwrap();
    /// assert_eq!(result, vec![1, 2, 3]);
    ///
    /// world.destruct();
    /// ```
    pub fn broadcast<T: Serialize + DeserializeOwned + Clone>(&self, data: &T, root: i32) -> Result<T> {
        if root >= self.size {
            return Err(MPIError::InvalidRank(root));
        }

        debug!("Rank {} entering broadcast", self.rank);

        let result = if self.rank == root {
            // Root sends to all other ranks
            for rank in 0..self.size {
                if rank != root {
                    self.send(data, rank, 0)?;
                }
            }
            Ok(data.clone())
        } else {
            // Non-root ranks receive from root
            self.recv(root, 0)
        };

        debug!("Rank {} completed broadcast", self.rank);
        result
    }

    /// Scatter data from root rank to all ranks.
    ///
    /// Distributes distinct pieces of data from the root process to all processes.
    /// The root process provides a slice with size equal to the number of processes,
    /// and each process receives one piece.
    ///
    /// # Arguments
    ///
    /// * `data` - Slice of data to scatter (only required on root process)
    /// * `root` - The rank of the scattering process
    ///
    /// # Returns
    ///
    /// * `Ok(T)` - This process's piece of the scattered data
    /// * `Err(MPIError)` - If the scatter fails
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use simple_mpi::World;
    ///
    /// let world = World::init(4).unwrap();
    ///
    /// // Root scatters different numbers to each process
    /// let data = if world.rank() == 0 {
    ///     Some(&[10, 20, 30, 40][..])
    /// } else {
    ///     None
    /// };
    ///
    /// let my_number = world.scatter(data, 0).unwrap();
    /// println!("Process {} got {}", world.rank(), my_number);
    ///
    /// world.destruct();
    /// ```
    pub fn scatter<T: Serialize + DeserializeOwned + Clone>(
        &self,
        data: Option<&[T]>,
        root: i32,
    ) -> Result<T> {
        if root >= self.size {
            return Err(MPIError::InvalidRank(root));
        }

        debug!("Rank {} entering scatter", self.rank);

        let result = if self.rank == root {
            let data = data.ok_or_else(|| {
                MPIError::InitError("Root must provide data for scatter".into())
            })?;

            if data.len() != self.size as usize {
                return Err(MPIError::InitError(
                    "Data length must match world size".into(),
                ));
            }

            // Send each piece to corresponding rank
            for (rank, item) in data.iter().enumerate() {
                if rank as i32 != root {
                    debug!("Root sending piece to rank {}", rank);
                    self.send(item, rank as i32, 0)?;
                }
            }

            debug!("Root keeping its piece");
            data[root as usize].clone()
        } else {
            // Non-root ranks receive their piece
            debug!("Rank {} waiting for its piece", self.rank);
            let piece = self.recv(root, 0)?;
            debug!("Rank {} received its piece", self.rank);
            piece
        };

        // Barrier to ensure all processes have completed
        self.barrier()?;

        debug!("Rank {} completed scatter", self.rank);
        Ok(result)
    }

    /// Gather data from all ranks to root rank.
    ///
    /// Collects data from all processes into a vector on the root process.
    /// This is the inverse operation of scatter.
    ///
    /// # Arguments
    ///
    /// * `data` - The local data to contribute to the gather
    /// * `root` - The rank of the gathering process
    ///
    /// # Returns
    ///
    /// * `Ok(Some(Vec<T>))` - Vector of gathered data (only on root process)
    /// * `Ok(None)` - On non-root processes
    /// * `Err(MPIError)` - If the gather fails
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use simple_mpi::World;
    ///
    /// let world = World::init(4).unwrap();
    ///
    /// // Each process contributes its rank
    /// let result = world.gather(&world.rank(), 0).unwrap();
    ///
    /// if world.rank() == 0 {
    ///     // Root process gets vector of all ranks
    ///     println!("Gathered: {:?}", result.unwrap());
    /// }
    ///
    /// world.destruct();
    /// ```
    pub fn gather<T: Serialize + DeserializeOwned + Clone>(
        &self,
        data: &T,
        root: i32,
    ) -> Result<Option<Vec<T>>> {
        if root >= self.size {
            return Err(MPIError::InvalidRank(root));
        }

        debug!("Rank {} entering gather", self.rank);

        let result = if self.rank == root {
            let mut result = Vec::with_capacity(self.size as usize);
            result.push(data.clone());

            // Receive from all other ranks
            for rank in 0..self.size {
                if rank != root {
                    debug!("Root receiving from rank {}", rank);
                    result.push(self.recv(rank, 0)?);
                }
            }

            Some(result)
        } else {
            // Non-root ranks send their data to root
            debug!("Rank {} sending to root", self.rank);
            self.send(data, root, 0)?;
            None
        };

        // Barrier to ensure all processes have completed
        self.barrier()?;

        debug!("Rank {} completed gather", self.rank);
        Ok(result)
    }

    /// Explicitly clean up MPI resources and synchronize process shutdown.
    ///
    /// This method must be called when you're done using the World instance to ensure
    /// proper cleanup of shared memory resources and synchronization of process shutdown.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use simple_mpi::World;
    ///
    /// let world = World::init(2).unwrap();
    /// // ... use the world for communication ...
    /// world.destruct(); // Clean up resources
    /// ```
    pub fn destruct(self) {
        unsafe {
            let state = &mut *(SHARED_MEMORY[self.shmem_idx].as_ptr() as *mut SharedState);
            
            // Signal this process is ready for cleanup
            state.cleanup_ready[self.rank as usize].store(true, Ordering::SeqCst);
            
            // Wait for all processes to be ready for cleanup
            let mut all_ready = false;
            while !all_ready {
                all_ready = true;
                for i in 0..self.size {
                    if !state.cleanup_ready[i as usize].load(Ordering::SeqCst) {
                        all_ready = false;
                        std::hint::spin_loop();
                        break;
                    }
                }
            }
            
            // Wait for cleanup synchronization
            if self.rank == 0 {
                state.cleanup_done.store(true, Ordering::SeqCst);
            } else {
                while !state.cleanup_done.load(Ordering::SeqCst) {
                    std::hint::spin_loop();
                }
            }
            
            if self.rank == 0 {
                debug!("Parent process waiting for children to finish");
                // Wait for all child processes to mark themselves as done
                let mut all_done = false;
                while !all_done {
                    all_done = true;
                    for i in 1..self.size {
                        if !state.done_flags[i as usize].load(Ordering::SeqCst) {
                            all_done = false;
                            std::hint::spin_loop();
                            break;
                        }
                    }
                }
                // reset done flags
                for i in 0..self.size {
                    state.done_flags[i as usize].store(false, Ordering::SeqCst);
                }
                // Clear all shared memory
                debug!("Parent process clearing shared memory");
                // Signal root is done
                state.root_done.store(true, Ordering::SeqCst);

                // receive signal from children
                for i in 1..self.size {
                    while !state.done_flags[i as usize].load(Ordering::SeqCst) {
                        std::hint::spin_loop();
                    }
                }
                // Clear shared memory
                SHARED_MEMORY.clear();
            } else {
                // Child process marks itself as done and waits for root
                state.done_flags[self.rank as usize].store(true, Ordering::SeqCst);
                while !state.root_done.load(Ordering::SeqCst) {
                    std::hint::spin_loop();
                }
                state.done_flags[self.rank as usize].store(true, Ordering::SeqCst);
            }
        }
        debug!("Process {} finished", self.rank());
    }
}
