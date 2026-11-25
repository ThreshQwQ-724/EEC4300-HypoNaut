"""Count legal single-edge components directly from HypoSpace's raw answer file."""
import argparse
from pathlib import Path

from common import DATA, HERE, digest
import ijson

from run_local_causal import write_json
from run_qwen_rr_budget import read
from run_partial_graph_pipeline import program_anchor
from stream_causal import HashReader


RAW = DATA/'causal/datasets/node06/n6_all_observations.json'


def component_sizes(masks, toggles, anchor):
    remaining = masks.copy()
    sizes = []
    assert anchor in remaining
    root = anchor
    while remaining:
        remaining.remove(root)
        queue = [root]
        for graph in queue:
            for bit in toggles:
                neighbor = graph ^ bit
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    queue.append(neighbor)
        sizes.append(len(queue))
        if remaining:
            root = next(iter(remaining))
    return sizes


def audit(raw):
    from validate_constructive_hypospace import read_object

    rows = []
    with raw.open('rb') as stream:
        reader = HashReader(stream)
        events = iter(ijson.parse(reader))
        for prefix, event, _ in events:
            if prefix == 'metadata' and event == 'start_map':
                metadata = read_object(events)
                nodes = metadata['nodes']
                edges = [(a, b) for a in nodes for b in nodes if a != b]
                bits = {edge: 1 << i for i, edge in enumerate(edges)}
            if not (prefix.startswith('datasets_by_n_observations.')
                    and prefix.count('.') == 2 and event == 'start_map'):
                continue
            builder = ijson.common.ObjectBuilder()
            builder.event('start_map', None)
            masks, used = set(), 0
            for field, kind, value in events:
                if field == prefix+'.ground_truth_graphs' and kind == 'start_array':
                    graph_item = field+'.item'
                    edge_item = graph_item+'.edges.item.item'
                    mask, source = 0, None
                    for path, token, data in events:
                        if path == edge_item:
                            if source is None:
                                source = data
                            else:
                                mask |= bits[source, data]
                                source = None
                        elif path == graph_item and token == 'end_map':
                            assert source is None and mask not in masks
                            masks.add(mask)
                            used |= mask
                            mask = 0
                        elif path == field and token == 'end_array':
                            break
                else:
                    builder.event(kind, value)
                if field == prefix and kind == 'end_map':
                    break
            row = builder.value
            anchor = program_anchor(nodes, metadata['max_edges'], row['observations'])
            assert anchor is not None and len(masks) == row['n_compatible_graphs']
            sizes = component_sizes(masks, [bit for bit in bits.values() if used & bit],
                                    sum(bits[edge] for edge in anchor))
            rows.append({'id': row['observation_set_id'], 'answers': len(masks), 'sizes': sizes})
            if len(rows) % 25 == 0 or len(sizes) > 1:
                print(f"{len(rows)}/{metadata['total_observation_sets']} {row['observation_set_id']} "
                      f"components={len(sizes)}", flush=True)
        assert reader.bytes_read == raw.stat().st_size
        assert len(rows) == metadata['total_observation_sets'] == len({r['id'] for r in rows})
        return rows, reader.hasher.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    rows, raw_sha = audit(RAW)
    prior = read(HERE/'results/causal_audit.json')
    assert raw_sha == prior['source_sha256']
    assert {r['id']:r['answers'] for r in rows} == {r['id']:r['n_ground_truths'] for r in prior['instances']}
    write_json(args.output, {'raw_sha': raw_sha, 'code_sha': digest(__file__), 'cases': rows})
    print('Disconnected cases:', [r for r in rows if len(r['sizes']) > 1], flush=True)


if __name__ == '__main__':
    main()
