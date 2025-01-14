//! A simple MPI implementation for single-machine shared memory communication
//! 
//! This library provides basic MPI-like functionality including send, receive,
//! broadcast, scatter, and gather operations using POSIX shared memory.

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

/// The main MPI World structure that manages inter-process communication
pub struct World {
    rank: i32,
    size: i32,
    shmem_idx: usize,
}

impl World {
    /// Initialize the MPI environment and create a new World.
    /// This function blocks until all processes are initialized.
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

    /// Get the rank of the current process
    pub fn rank(&self) -> i32 {
        self.rank
    }

    /// Get the total number of processes
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

    /// Send data to a specific rank (blocking until received)
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

    /// Receive data from a specific rank
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

    /// Broadcast data from root rank to all other ranks
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

    /// Scatter data from root rank to all ranks (blocking until all processes complete)
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

    /// Gather data from all ranks to root rank (blocking until all processes complete)
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
    /// This should be called when you're done using the World instance.
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
