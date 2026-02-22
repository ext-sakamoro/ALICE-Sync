use alice_sync::{
    Event, EventKind, Fixed, MotionData, Node, NodeId, Vec3Fixed, Vec3Simd, World, WorldSoA,
};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

fn bench_event_apply(c: &mut Criterion) {
    let mut group = c.benchmark_group("event_apply");

    let mut node = Node::new(NodeId(1));
    node.apply_event(&Event::new(EventKind::Spawn {
        entity: 1,
        kind: 0,
        pos: [0, 0, 0],
    }))
    .unwrap();

    let motion = Event::new(EventKind::Motion {
        entity: 1,
        delta: [100, 0, 0],
    });

    group.bench_function("motion_single", |b| {
        b.iter(|| {
            node.apply_event(black_box(&motion)).unwrap();
        })
    });

    group.finish();
}

fn bench_world_hash(c: &mut Criterion) {
    let mut group = c.benchmark_group("world_hash");

    for entity_count in [100, 1000, 10000] {
        let mut world = World::new(42);

        for i in 0..entity_count {
            world
                .apply(&Event::new(EventKind::Spawn {
                    entity: i,
                    kind: 0,
                    pos: [i as i16, 0, 0],
                }))
                .unwrap();
        }

        // O(1) hash (incremental)
        group.bench_with_input(
            BenchmarkId::new("incremental_O1", entity_count),
            &entity_count,
            |b, _| b.iter(|| black_box(world.hash())),
        );

        // O(N) hash (full recalculation)
        group.bench_with_input(
            BenchmarkId::new("full_recalc_ON", entity_count),
            &entity_count,
            |b, _| {
                let mut w = world.clone();
                b.iter(|| black_box(w.recalculate_hash()))
            },
        );
    }

    group.finish();
}

fn bench_simd_vs_scalar(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_vs_scalar");

    // Scalar Vec3 addition
    group.bench_function("vec3_add_scalar", |b| {
        let mut pos = Vec3Fixed::from_f32(1.0, 2.0, 3.0);
        let delta = Vec3Fixed::from_f32(0.1, 0.1, 0.1);
        b.iter(|| {
            pos = black_box(pos) + black_box(delta);
            black_box(pos)
        })
    });

    // SIMD Vec3 addition
    group.bench_function("vec3_add_simd", |b| {
        let mut pos = Vec3Fixed::from_f32(1.0, 2.0, 3.0);
        let delta = Vec3Simd::new(
            Fixed::from_f32(0.1).0,
            Fixed::from_f32(0.1).0,
            Fixed::from_f32(0.1).0,
        );
        b.iter(|| {
            black_box(&delta).add_to_vec3(black_box(&mut pos));
            black_box(pos)
        })
    });

    group.finish();
}

fn bench_entity_lookup(c: &mut Criterion) {
    let mut group = c.benchmark_group("entity_lookup");

    for entity_count in [100, 1000, 10000] {
        let mut world = World::new(42);

        for i in 0..entity_count {
            world
                .apply(&Event::new(EventKind::Spawn {
                    entity: i,
                    kind: 0,
                    pos: [0, 0, 0],
                }))
                .unwrap();
        }

        // Lookup existing entity (O(1) Vec indexing)
        group.bench_with_input(
            BenchmarkId::new("get_entity", entity_count),
            &entity_count,
            |b, &count| {
                let target = count / 2;
                b.iter(|| black_box(world.get_entity(black_box(target))))
            },
        );
    }

    group.finish();
}

fn bench_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("serialization");

    let event = Event::new(EventKind::Motion {
        entity: 1,
        delta: [1000, 2000, 3000],
    });

    group.bench_function("bincode_encode", |b| b.iter(|| black_box(event.to_bytes())));

    group.bench_function("bitcode_encode", |b| {
        b.iter(|| black_box(event.to_compact_bytes()))
    });

    let bincode_bytes = event.to_bytes();
    let bitcode_bytes = event.to_compact_bytes();

    println!(
        "Size: bincode={} bytes, bitcode={} bytes ({}% smaller)",
        bincode_bytes.len(),
        bitcode_bytes.len(),
        100 - (bitcode_bytes.len() * 100 / bincode_bytes.len())
    );

    group.bench_function("bincode_decode", |b| {
        b.iter(|| black_box(Event::from_bytes(&bincode_bytes)))
    });

    group.bench_function("bitcode_decode", |b| {
        b.iter(|| black_box(Event::from_compact_bytes(&bitcode_bytes)))
    });

    group.finish();
}

fn bench_sync_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("sync_throughput");

    let events: Vec<Event> = (0..1000)
        .map(|i| {
            Event::new(EventKind::Motion {
                entity: (i % 100) as u32,
                delta: [1, 0, 0],
            })
        })
        .collect();

    group.bench_function("apply_1000_events", |b| {
        b.iter(|| {
            let mut node = Node::new(NodeId(1));

            for i in 0..100u32 {
                node.apply_event(&Event::new(EventKind::Spawn {
                    entity: i,
                    kind: 0,
                    pos: [0, 0, 0],
                }))
                .unwrap();
            }

            for e in &events {
                node.apply_event(black_box(e)).unwrap();
            }

            black_box(node.world_hash())
        })
    });

    group.finish();
}

fn bench_batch_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_processing");

    // Create motion data for batch processing
    let motions: Vec<MotionData> = (0..1000)
        .map(|i| MotionData {
            entity: (i % 100) as u32,
            delta_x: 1,
            delta_y: 0,
            delta_z: 0,
        })
        .collect();

    // Batch apply (SoA optimized)
    group.bench_function("batch_1000_motions", |b| {
        b.iter(|| {
            let mut world = World::new(42);

            // Spawn 100 entities
            for i in 0..100u32 {
                world
                    .apply(&Event::new(EventKind::Spawn {
                        entity: i,
                        kind: 0,
                        pos: [0, 0, 0],
                    }))
                    .unwrap();
            }

            // Batch apply all motions
            world.apply_motions_batch(black_box(&motions));

            black_box(world.hash())
        })
    });

    // Individual apply (for comparison)
    let events: Vec<Event> = (0..1000)
        .map(|i| {
            Event::new(EventKind::Motion {
                entity: (i % 100) as u32,
                delta: [1, 0, 0],
            })
        })
        .collect();

    group.bench_function("individual_1000_motions", |b| {
        b.iter(|| {
            let mut world = World::new(42);

            for i in 0..100u32 {
                world
                    .apply(&Event::new(EventKind::Spawn {
                        entity: i,
                        kind: 0,
                        pos: [0, 0, 0],
                    }))
                    .unwrap();
            }

            for e in &events {
                world.apply(black_box(e)).unwrap();
            }

            black_box(world.hash())
        })
    });

    group.finish();
}

fn bench_vertical_simd(c: &mut Criterion) {
    let mut group = c.benchmark_group("vertical_simd");

    // WorldSoA with Vertical SIMD (v0.5 Zen Mode)
    group.bench_function("soa_vertical_1024_entities", |b| {
        b.iter(|| {
            let mut world = WorldSoA::new(42);

            // Spawn 1024 entities (aligned for SIMD)
            for i in 0..1024u32 {
                world
                    .apply(&Event::new(EventKind::Spawn {
                        entity: i,
                        kind: 0,
                        pos: [0, 0, 0],
                    }))
                    .unwrap();
            }

            // Apply uniform motion to all 1024 using Vertical SIMD
            // Process 8 entities per SIMD instruction
            world.apply_uniform_motion_vertical(0, 1024, 1, 0, 0);

            black_box(world.hash())
        })
    });

    // Traditional AoS World for comparison
    group.bench_function("aos_batch_1024_entities", |b| {
        let motions: Vec<MotionData> = (0..1024)
            .map(|i| MotionData {
                entity: i as u32,
                delta_x: 1,
                delta_y: 0,
                delta_z: 0,
            })
            .collect();

        b.iter(|| {
            let mut world = World::new(42);

            for i in 0..1024u32 {
                world
                    .apply(&Event::new(EventKind::Spawn {
                        entity: i,
                        kind: 0,
                        pos: [0, 0, 0],
                    }))
                    .unwrap();
            }

            world.apply_motions_batch(black_box(&motions));

            black_box(world.hash())
        })
    });

    // WorldSoA batch (non-SIMD for comparison)
    group.bench_function("soa_batch_1024_entities", |b| {
        let motions: Vec<MotionData> = (0..1024)
            .map(|i| MotionData {
                entity: i as u32,
                delta_x: 1,
                delta_y: 0,
                delta_z: 0,
            })
            .collect();

        b.iter(|| {
            let mut world = WorldSoA::new(42);

            for i in 0..1024u32 {
                world
                    .apply(&Event::new(EventKind::Spawn {
                        entity: i,
                        kind: 0,
                        pos: [0, 0, 0],
                    }))
                    .unwrap();
            }

            world.apply_motions_batch(black_box(&motions));

            black_box(world.hash())
        })
    });

    group.finish();
}

fn bench_demon_mode(c: &mut Criterion) {
    let mut group = c.benchmark_group("demon_mode");

    // ========================================================================
    // Lv.6 "Demon Mode": Sort-Based Batching Benchmarks
    // ========================================================================

    // Random access pattern (worst case)
    group.bench_function("random_access_1000", |b| {
        let motions: Vec<MotionData> = (0..1000)
            .map(|i| MotionData {
                entity: ((i * 7 + 13) % 100) as u32, // Pseudo-random pattern
                delta_x: 1,
                delta_y: 0,
                delta_z: 0,
            })
            .collect();

        b.iter(|| {
            let mut world = WorldSoA::new(42);

            for i in 0..100u32 {
                world
                    .apply(&Event::new(EventKind::Spawn {
                        entity: i,
                        kind: 0,
                        pos: [0, 0, 0],
                    }))
                    .unwrap();
            }

            // Unsorted batch (random access)
            world.apply_motions_batch(black_box(&motions));

            black_box(world.hash())
        })
    });

    // Sorted access pattern (Demon Mode)
    group.bench_function("sorted_batch_1000", |b| {
        let motions: Vec<MotionData> = (0..1000)
            .map(|i| MotionData {
                entity: ((i * 7 + 13) % 100) as u32, // Same pseudo-random pattern
                delta_x: 1,
                delta_y: 0,
                delta_z: 0,
            })
            .collect();

        b.iter(|| {
            let mut world = WorldSoA::new(42);

            for i in 0..100u32 {
                world
                    .apply(&Event::new(EventKind::Spawn {
                        entity: i,
                        kind: 0,
                        pos: [0, 0, 0],
                    }))
                    .unwrap();
            }

            // Demon Mode: sorted batch
            world.apply_motions_sorted(black_box(motions.clone()));

            black_box(world.hash())
        })
    });

    // Sequential ID pattern (best case for SIMD)
    group.bench_function("sequential_ids_1000", |b| {
        let motions: Vec<MotionData> = (0..1000)
            .map(|i| MotionData {
                entity: (i % 100) as u32, // Sequential pattern
                delta_x: 1,
                delta_y: 0,
                delta_z: 0,
            })
            .collect();

        b.iter(|| {
            let mut world = WorldSoA::new(42);

            for i in 0..100u32 {
                world
                    .apply(&Event::new(EventKind::Spawn {
                        entity: i,
                        kind: 0,
                        pos: [0, 0, 0],
                    }))
                    .unwrap();
            }

            // Demon Mode with sequential IDs
            world.apply_motions_sorted(black_box(motions.clone()));

            black_box(world.hash())
        })
    });

    // Contiguous 8-entity SIMD batch (ideal case)
    group.bench_function("contiguous_8_simd_1024", |b| {
        // 1024 motions targeting 128 batches of 8 contiguous entities
        let motions: Vec<MotionData> = (0..1024)
            .map(|i| MotionData {
                entity: i as u32, // Perfectly contiguous
                delta_x: 1,
                delta_y: 0,
                delta_z: 0,
            })
            .collect();

        b.iter(|| {
            let mut world = WorldSoA::new(42);

            for i in 0..1024u32 {
                world
                    .apply(&Event::new(EventKind::Spawn {
                        entity: i,
                        kind: 0,
                        pos: [0, 0, 0],
                    }))
                    .unwrap();
            }

            world.apply_motions_sorted(black_box(motions.clone()));

            black_box(world.hash())
        })
    });

    // Pre-sorted (skip sorting overhead)
    group.bench_function("presorted_1000", |b| {
        let mut motions: Vec<MotionData> = (0..1000)
            .map(|i| MotionData {
                entity: ((i * 7 + 13) % 100) as u32,
                delta_x: 1,
                delta_y: 0,
                delta_z: 0,
            })
            .collect();
        motions.sort_unstable_by_key(|m| m.entity);

        b.iter(|| {
            let mut world = WorldSoA::new(42);

            for i in 0..100u32 {
                world
                    .apply(&Event::new(EventKind::Spawn {
                        entity: i,
                        kind: 0,
                        pos: [0, 0, 0],
                    }))
                    .unwrap();
            }

            world.apply_motions_presorted(black_box(&motions));

            black_box(world.hash())
        })
    });

    group.finish();
}

fn bench_overengineering_check(c: &mut Criterion) {
    let mut group = c.benchmark_group("overengineering_check");

    // ========================================================================
    // Case A: Fresh State (Sequential Spawn, ID ≈ Slot)
    // ========================================================================

    // My impl (full: sort + coalesce + slot-sort)
    group.bench_function("case_a_sorted_full", |b| {
        let motions: Vec<MotionData> = (0..1000)
            .map(|i| MotionData {
                entity: (i % 100) as u32,
                delta_x: 1,
                delta_y: 0,
                delta_z: 0,
            })
            .collect();

        b.iter(|| {
            let mut world = WorldSoA::new(42);
            for i in 0..100u32 {
                world
                    .apply(&Event::new(EventKind::Spawn {
                        entity: i,
                        kind: 0,
                        pos: [0, 0, 0],
                    }))
                    .unwrap();
            }
            world.apply_motions_sorted(black_box(motions.clone()));
            black_box(world.hash())
        })
    });

    // Gemini impl (ID-sort only, no coalesce, no slot-sort)
    group.bench_function("case_a_demon_gemini", |b| {
        let motions: Vec<MotionData> = (0..1000)
            .map(|i| MotionData {
                entity: (i % 100) as u32,
                delta_x: 1,
                delta_y: 0,
                delta_z: 0,
            })
            .collect();

        b.iter(|| {
            let mut world = WorldSoA::new(42);
            for i in 0..100u32 {
                world
                    .apply(&Event::new(EventKind::Spawn {
                        entity: i,
                        kind: 0,
                        pos: [0, 0, 0],
                    }))
                    .unwrap();
            }
            world.apply_motions_demon(black_box(motions.clone()));
            black_box(world.hash())
        })
    });

    // No sort baseline
    group.bench_function("case_a_nosort", |b| {
        let motions: Vec<MotionData> = (0..1000)
            .map(|i| MotionData {
                entity: (i % 100) as u32,
                delta_x: 1,
                delta_y: 0,
                delta_z: 0,
            })
            .collect();

        b.iter(|| {
            let mut world = WorldSoA::new(42);
            for i in 0..100u32 {
                world
                    .apply(&Event::new(EventKind::Spawn {
                        entity: i,
                        kind: 0,
                        pos: [0, 0, 0],
                    }))
                    .unwrap();
            }
            world.apply_motions_nosort(black_box(&motions));
            black_box(world.hash())
        })
    });

    // ========================================================================
    // Case B: Fragmented State (50% Despawn + Re-Spawn)
    // ========================================================================

    group.bench_function("case_b_sorted_full_fragmented", |b| {
        b.iter(|| {
            let mut world = WorldSoA::new(42);
            // Spawn 100
            for i in 0..100u32 {
                world
                    .apply(&Event::new(EventKind::Spawn {
                        entity: i,
                        kind: 0,
                        pos: [0, 0, 0],
                    }))
                    .unwrap();
            }
            // Despawn 50% (even IDs)
            for i in (0..100u32).step_by(2) {
                world
                    .apply(&Event::new(EventKind::Despawn { entity: i }))
                    .unwrap();
            }
            // Re-spawn with new IDs (100-149)
            for i in 100..150u32 {
                world
                    .apply(&Event::new(EventKind::Spawn {
                        entity: i,
                        kind: 0,
                        pos: [0, 0, 0],
                    }))
                    .unwrap();
            }

            // Now motions target 0-99 but only odd exist + 100-149
            let frag_motions: Vec<MotionData> = (0..1000)
                .map(|i| {
                    let entity = if i % 2 == 0 {
                        (i % 50) * 2 + 1 // odd: 1,3,5...99
                    } else {
                        100 + (i % 50) // new: 100-149
                    };
                    MotionData {
                        entity: entity as u32,
                        delta_x: 1,
                        delta_y: 0,
                        delta_z: 0,
                    }
                })
                .collect();

            world.apply_motions_sorted(black_box(frag_motions));
            black_box(world.hash())
        })
    });

    group.bench_function("case_b_demon_gemini_fragmented", |b| {
        b.iter(|| {
            let mut world = WorldSoA::new(42);
            for i in 0..100u32 {
                world
                    .apply(&Event::new(EventKind::Spawn {
                        entity: i,
                        kind: 0,
                        pos: [0, 0, 0],
                    }))
                    .unwrap();
            }
            for i in (0..100u32).step_by(2) {
                world
                    .apply(&Event::new(EventKind::Despawn { entity: i }))
                    .unwrap();
            }
            for i in 100..150u32 {
                world
                    .apply(&Event::new(EventKind::Spawn {
                        entity: i,
                        kind: 0,
                        pos: [0, 0, 0],
                    }))
                    .unwrap();
            }

            let frag_motions: Vec<MotionData> = (0..1000)
                .map(|i| {
                    let entity = if i % 2 == 0 {
                        (i % 50) * 2 + 1
                    } else {
                        100 + (i % 50)
                    };
                    MotionData {
                        entity: entity as u32,
                        delta_x: 1,
                        delta_y: 0,
                        delta_z: 0,
                    }
                })
                .collect();

            world.apply_motions_demon(black_box(frag_motions));
            black_box(world.hash())
        })
    });

    // ========================================================================
    // Case C: Coalesce Benefit Check
    // ========================================================================

    // Same entity 1000 times (coalesce should help)
    group.bench_function("case_c_coalesce_same_entity", |b| {
        let motions: Vec<MotionData> = (0..1000)
            .map(|_| MotionData {
                entity: 50, // Always same entity
                delta_x: 1,
                delta_y: 0,
                delta_z: 0,
            })
            .collect();

        b.iter(|| {
            let mut world = WorldSoA::new(42);
            for i in 0..100u32 {
                world
                    .apply(&Event::new(EventKind::Spawn {
                        entity: i,
                        kind: 0,
                        pos: [0, 0, 0],
                    }))
                    .unwrap();
            }
            world.apply_motions_sorted(black_box(motions.clone()));
            black_box(world.hash())
        })
    });

    group.bench_function("case_c_nocoalesce_same_entity", |b| {
        let motions: Vec<MotionData> = (0..1000)
            .map(|_| MotionData {
                entity: 50,
                delta_x: 1,
                delta_y: 0,
                delta_z: 0,
            })
            .collect();

        b.iter(|| {
            let mut world = WorldSoA::new(42);
            for i in 0..100u32 {
                world
                    .apply(&Event::new(EventKind::Spawn {
                        entity: i,
                        kind: 0,
                        pos: [0, 0, 0],
                    }))
                    .unwrap();
            }
            world.apply_motions_demon(black_box(motions.clone()));
            black_box(world.hash())
        })
    });

    // Unique entities (coalesce overhead with no benefit)
    group.bench_function("case_c_coalesce_unique_entities", |b| {
        let motions: Vec<MotionData> = (0..100)
            .map(|i| MotionData {
                entity: i as u32, // All unique
                delta_x: 1,
                delta_y: 0,
                delta_z: 0,
            })
            .collect();

        b.iter(|| {
            let mut world = WorldSoA::new(42);
            for i in 0..100u32 {
                world
                    .apply(&Event::new(EventKind::Spawn {
                        entity: i,
                        kind: 0,
                        pos: [0, 0, 0],
                    }))
                    .unwrap();
            }
            world.apply_motions_sorted(black_box(motions.clone()));
            black_box(world.hash())
        })
    });

    group.bench_function("case_c_nocoalesce_unique_entities", |b| {
        let motions: Vec<MotionData> = (0..100)
            .map(|i| MotionData {
                entity: i as u32,
                delta_x: 1,
                delta_y: 0,
                delta_z: 0,
            })
            .collect();

        b.iter(|| {
            let mut world = WorldSoA::new(42);
            for i in 0..100u32 {
                world
                    .apply(&Event::new(EventKind::Spawn {
                        entity: i,
                        kind: 0,
                        pos: [0, 0, 0],
                    }))
                    .unwrap();
            }
            world.apply_motions_demon(black_box(motions.clone()));
            black_box(world.hash())
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_event_apply,
    bench_world_hash,
    bench_simd_vs_scalar,
    bench_entity_lookup,
    bench_serialization,
    bench_sync_throughput,
    bench_batch_processing,
    bench_vertical_simd,
    bench_demon_mode,
    bench_overengineering_check
);
criterion_main!(benches);
