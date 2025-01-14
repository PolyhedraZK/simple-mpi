use simple_mpi::World;

fn main() {
    // Initialize MPI with 4 processes - this will create shared memory and spawn processes
    println!("Starting process with PID: {}", std::process::id());
    let world = World::init(4).unwrap();
    println!("Process {} initialized (PID: {})", world.rank(), std::process::id());

    // Example 1: Send/Receive
    if world.rank() == 0 {
        // Rank 0 sends a message to rank 1
        let message = format!("Hello from rank {}", world.rank());
        world.send(&message, 1, 0).unwrap();
        println!("Rank {} sent message to rank 1", world.rank());
    } else if world.rank() == 1 {
        // Rank 1 receives the message
        let message: String = world.recv(0, 0).unwrap();
        println!("Rank {} received message: {}", world.rank(), message);
    }

    // Example 2: Broadcast
    if world.rank() == 0 {
        // Rank 0 broadcasts data to all ranks
        let data = vec![1, 2, 3, 4];
        let result = world.broadcast(&data, 0).unwrap();
        println!("Rank {} broadcast data: {:?}", world.rank(), result);
    } else {
        // Other ranks receive the broadcast
        let result: Vec<i32> = world.broadcast(&Vec::new(), 0).unwrap();
        println!("Rank {} received broadcast: {:?}", world.rank(), result);
    }

    println!("Rank {} preparing for scatter", world.rank());
    // Example 3: Scatter
    if world.rank() == 0 {
        // Rank 0 scatters data to all ranks
        let data: Vec<i32> = (0..world.size()).map(|i| i * 10).collect();
        println!("Rank {} scattering data: {:?}", world.rank(), data);
        let piece = world.scatter(Some(&data), 0).unwrap();
        println!("Rank {} scattered and got piece: {}", world.rank(), piece);
    } else {
        // Other ranks receive their piece
        println!("Rank {} waiting for scatter", world.rank());
        let piece: i32 = world.scatter(None, 0).unwrap();
        println!("Rank {} received piece: {}", world.rank(), piece);
    }

    println!("Rank {} starting gather phase", world.rank());
    // Example 4: Gather
    let local_data = world.rank() * 5;
    println!("Rank {} sending local data: {}", world.rank(), local_data);
    let gathered = world.gather(&local_data, 0).unwrap();
    if world.rank() == 0 {
        println!("Rank {} gathered data: {:?}", world.rank(), gathered.unwrap());
    } else {
        println!("Rank {} completed gather", world.rank());
    }

    println!("Process {} finished", world.rank());
    
    // Clean up MPI resources
    world.destruct();
}
