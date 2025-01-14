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
    // Initialize MPI with 4 processes
    let world = World::init(4).unwrap();

    // Example: Shared Read-only Array with Pod type
    if world.rank() == 0 {
        // Rank 0 creates and shares an array of points
        let points = vec![
            Point { x: 1.0, y: 2.0, z: 3.0 },
            Point { x: 4.0, y: 5.0, z: 6.0 },
            Point { x: 7.0, y: 8.0, z: 9.0 },
        ];
        println!("Rank {} creating shared array", world.rank());
        
        // Share the array - other processes will get direct read access
        let shared = world.shared_readonly_array(Some(&points), 0).unwrap();
        
        // Even source can read it back through shared memory
        let view = shared.as_slice();
        println!("Rank {} reading points:", world.rank());
        for (i, point) in view.iter().enumerate() {
            println!("  Point {}: ({}, {}, {})", i, point.x, point.y, point.z);
        }
    } else {
        // Other ranks get direct read access to the shared array
        let shared = world.shared_readonly_array::<Point>(None, 0).unwrap();
        let view = shared.as_slice();

        for (i, point) in view.iter().enumerate() {
            assert_eq!(point.x, (i as f32) * 3.0 + 1.0);
        }
    }

    // Clean up MPI resources
    world.destruct();
}
