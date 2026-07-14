use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Seek, SeekFrom, Write};

use anyhow::{bail, Context, Result};
use num_integer::Integer;

use crate::math::legendre_symbol;

pub(crate) fn prune_singles<F>(file_path: &str, dest_path: &str, logfunc: F) -> Result<()>
where
    F: Fn(String),
{
    prune(file_path, dest_path, false, logfunc)
}

pub(crate) fn prune_cliques<F>(file_path: &str, dest_path: &str, logfunc: F) -> Result<()>
where
    F: Fn(String),
{
    prune(file_path, dest_path, true, logfunc)
}

fn prune<F>(file_path: &str, dest_path: &str, cliques: bool, logfunc: F) -> Result<()>
where
    F: Fn(String),
{
    let file = File::open(file_path).context("failed to open file")?;
    let mut reader = BufReader::new(file);

    let mut fmap: HashMap<i64, [i64; 2]> = HashMap::new();
    let mut gmap: HashMap<i64, [i64; 2]> = HashMap::new();
    let mut sizemap: HashMap<i64, u32> = HashMap::new();

    let mut n_total = 0;
    let mut position = 0;
    loop {
        let mut line = String::new();
        let bytes_read = reader.read_line(&mut line).context("Failed to read line")?;
        let current_offset: i64 = position;
        position += bytes_read as i64;

        if bytes_read == 0 {
            break; // End of file
        }

        let line = line.trim();

        // Skip comments
        if line.starts_with('#') {
            continue;
        }

        // Split the line into parts
        let parts: Vec<&str> = line.split(':').collect();
        if parts.len() < 3 {
            continue;
        }
        n_total += 1;

        let [x, y] = parts[0].split(',').collect::<Vec<_>>()[..] else {
            bail!("invalid x,y coordinates at offset {current_offset}")
        };
        let x = x.parse::<i64>().context("invalid x")?;
        let y = y.parse::<i64>().context("invalid y")?;

        let facf = parts[1];
        let facg = parts[2];

        let mut relsize = 0;
        for f in facf.split(",") {
            relsize += 1;
            let l = i64::from_str_radix(f.trim(), 16).context("invalid hex number")?;
            if let Some(v) = fmap.get_mut(&l) {
                if v[0] >= 0 {
                    if v[1] >= 0 {
                        // Found 3 times
                        *v = [-1, -1];
                    } else {
                        v[1] = current_offset;
                    }
                }
            } else {
                fmap.insert(l, [current_offset, -1]);
            }
        }

        for f in facg.split(",") {
            relsize += 1;
            let l = i64::from_str_radix(f.trim(), 16).context("invalid hex number")?;
            let d = y.extended_gcd(&l); // {gcd, d.x=yinv, y=..}
            let mut r = if d.gcd == l {
                l
            } else {
                x.checked_mul(d.x).expect("overflow") % l
            };
            if r < 0 {
                r += l;
            }
            let idx = (l << 32) + r;
            if let Some(v) = gmap.get_mut(&idx) {
                if v[0] >= 0 {
                    if v[1] >= 0 {
                        // Found 3 times
                        *v = [-1, -1];
                    } else {
                        v[1] = current_offset;
                    }
                }
            } else {
                gmap.insert(idx, [current_offset, -1]);
            }
        }
        if cliques {
            sizemap.insert(current_offset, relsize);
        }
    }

    //logfunc(format!("relations {n_total} primes {}", fmap.len() + gmap.len()));

    // Make a undirected graph with primes appearing 1x or 2x
    let mut gr = petgraph::graph::UnGraph::<i64, ()>::new_undirected();
    let mut gridx = HashMap::new();
    let nsink = gr.add_node(-1);
    for off in fmap.values().chain(gmap.values()) {
        if off[0] > 0 && off[1] < 0 {
            // singletons
            let n0 = if let Some(&n) = gridx.get(&off[0]) {
                n
            } else {
                let n = gr.add_node(off[0]);
                gridx.insert(off[0], n);
                n
            };
            gr.add_edge(n0, nsink, ());
        }
        if off[0] > 0 && off[1] > 0 {
            let n0 = if let Some(&n) = gridx.get(&off[0]) {
                n
            } else {
                let n = gr.add_node(off[0]);
                gridx.insert(off[0], n);
                n
            };
            let n1 = if let Some(&n) = gridx.get(&off[1]) {
                n
            } else {
                let n = gr.add_node(off[1]);
                gridx.insert(off[1], n);
                n
            };
            gr.add_edge(n0, n1, ());
        }
    }

    let excess: i64 = n_total as i64 - (fmap.len() + gmap.len()) as i64;
    let max_removed = (excess - 200) / 2;
    //logfunc(format!("excess = {excess}"));

    let comps = petgraph::algo::tarjan_scc(&gr);
    let mut orphans = vec![];
    let mut comps2 = vec![];
    for c in comps {
        if c.contains(&nsink) {
            // At least 1 connected component contains nsink (singletons)
            //println!("singletons {}", c.len() - 1);
            orphans = c
                .iter()
                .filter_map(|&n| if n != nsink { gr.node_weight(n) } else { None })
                .collect();
        } else {
            comps2.push(c);
        }
    }

    let n_removed = orphans.len();
    if cliques && max_removed > 0 {
        // Also remove cliques
        comps2.sort_by_cached_key(|c| {
            c.iter()
                .map(|&n| {
                    sizemap
                        .get(gr.node_weight(n).unwrap())
                        .copied()
                        .unwrap_or(0)
                })
                .sum::<u32>()
        });
        let mut n_removed2 = 0;
        for c in &comps2[comps2.len().saturating_sub(max_removed as usize)..] {
            n_removed2 += c.len();
            orphans.extend(c.iter().filter_map(|&n| gr.node_weight(n)))
        }
        logfunc(format!(
            "Pruning {} relations ({} singletons and {} in cliques)",
            orphans.len(),
            n_removed,
            n_removed2
        ));
    } else {
        logfunc(format!(
            "Removing {n_removed} relations with singleton primes"
        ));
    }
    orphans.sort();
    let mut orphan_iter = orphans.into_iter();
    let mut orphan_next: i64 = orphan_iter.next().copied().unwrap_or(-1);

    // Stream file again to filter relations
    reader.seek(SeekFrom::Start(0))?;
    let w = File::create(dest_path).context("failed to open file")?;
    let mut bufw = BufWriter::new(w);
    //let mut n_out = 0;
    let mut position = 0;
    loop {
        let mut line = String::new();
        let bytes_read = reader.read_line(&mut line).context("Failed to read line")?;
        let current_offset = position;
        position += bytes_read;

        if bytes_read == 0 {
            break; // End of file
        }

        // Skip comments
        if line.starts_with('#') {
            continue;
        }
        if current_offset as i64 == orphan_next {
            orphan_next = orphan_iter.next().copied().unwrap_or(-1);
            continue;
        }
        bufw.write_all(line.as_bytes())?;
        //n_out += 1;
    }
    //println!("relations {n_total} => {n_out}");

    Ok(())
}

///
/// Returns a list of encoded relations (list of sparse rows with column indices)
/// and rational relations (x, y, facs)
///
/// The relations are modulo 2 and no duplicate indices are returned in the first list.
///
/// All vectors are returned as LE32 buffers.
///
/// FIXME: currently assumes that primes are not ramified.
pub(crate) fn parse_with_characters(
    file_path: &str,
    characters: Vec<(i64, i64)>,
) -> Result<(Vec<Vec<u8>>, Vec<(i64, i64, Vec<u8>)>)> {
    let file = File::open(file_path).context("failed to open file")?;
    let mut reader = BufReader::new(file);

    let mut rel_idx = 0;
    let mut basis_idx = HashMap::<(i32, i32), u64>::new();
    let mut last_basis_idx = characters.len() as u64; // Reserve 0 and 1..=N_CHARS

    let mut all_rels = vec![];
    let mut zrels = vec![];

    loop {
        let mut line = String::new();
        let read = reader.read_line(&mut line).context("Failed to read line")?;
        if read == 0 {
            break; // End of file
        }

        let line = line.trim();

        // Skip comments
        if line.starts_with('#') {
            continue;
        }

        // Split the line into parts
        let parts: Vec<&str> = line.split(':').collect();
        if parts.len() < 3 {
            continue;
        }

        rel_idx += 1;
        // Invariant: rel_idx > 0

        let mut rel = Vec::<i32>::with_capacity(32);
        rel.push(-rel_idx);
        rel.push(0);

        let [x, y] = parts[0].split(',').collect::<Vec<_>>()[..] else {
            bail!("invalid x,y coordinates at {}", parts[0])
        };
        let x = x.parse::<i64>().context("invalid x")?;
        let y = y.parse::<i64>().context("invalid y")?;

        let facf = parts[1];
        let facg = parts[2];

        let mut zrel = vec![];

        // Compute characters
        for (cidx, &(l, r)) in characters.iter().enumerate() {
            let v = x as i128 - r as i128 * y as i128;
            if legendre_symbol((v % l as i128) as i64, l) < 0 {
                rel.push(cidx as i32 + 1);
            }
        }

        for f in facf.split(",") {
            let l = i32::from_str_radix(f.trim(), 16).context("invalid hex number")?;
            let idx = if let Some(idx) = basis_idx.get(&(l, -1)) {
                *idx
            } else {
                last_basis_idx += 1;
                basis_idx.insert((l, -1), last_basis_idx);
                last_basis_idx
            };
            zrel.push(l);
            rel.push(idx as i32);
        }

        for f in facg.split(",") {
            let l = i64::from_str_radix(f.trim(), 16).context("invalid hex number")?;
            let d = y.extended_gcd(&l); // {gcd, d.x=yinv, y=..}
            let mut r = if d.gcd == l {
                l
            } else {
                x.checked_mul(d.x).expect("overflow") % l
            };
            if r < 0 {
                r += l;
            }
            let key = (l as i32, r as i32);
            let idx = if let Some(idx) = basis_idx.get(&key) {
                *idx
            } else {
                last_basis_idx += 1;
                basis_idx.insert(key, last_basis_idx);
                last_basis_idx
            };
            rel.push(idx as i32);
        }

        // Relations are modulo 2, code expects no duplicates
        rel.sort();
        let mut i = 0;
        let mut j = 0;
        while i < rel.len() {
            if i + 1 < rel.len() && rel[i] == rel[i + 1] {
                i += 2
            } else {
                rel[j] = rel[i];
                i += 1;
                j += 1;
            }
        }
        rel.truncate(j);
        all_rels.push(le32_vector(rel));
        zrels.push((x, y, le32_vector(zrel)));
    }
    assert_eq!(all_rels.len(), zrels.len());
    Ok((all_rels, zrels))
}

fn le32_vector(v: Vec<i32>) -> Vec<u8> {
    let len = 4 * v.len();
    let mut b = Vec::with_capacity(len);
    for x in v {
        let xb = x.to_le_bytes();
        b.extend_from_slice(&xb);
    }
    assert_eq!(b.len(), len);
    b
}
