"""Compare independent second Qwen choices over the same frozen rewiring menu."""
import argparse
from pathlib import Path

import requests

from common import HERE, digest
from run_uot_debate_menu import BASE_URL, MODEL, SECOND_AGENT_SEED_OFFSET, decode, read, request, write_json


SOURCE = HERE/'runs/uot_debate_menu_path_rank_20261113'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    output = parser.parse_args().output
    output.mkdir(parents=True, exist_ok=True)
    source_path = SOURCE/'results.json'
    source = read(source_path)
    records = []
    with requests.Session() as session:
        session.trust_env = False
        for row in source['rows']:
            case, slot, menu = row['case'], row['slot'], row['menu']
            folder = output/case
            folder.mkdir(exist_ok=True)
            prior = read(SOURCE/case/f'{slot:02}_A.json')
            payload = prior['request']
            seed = payload['options']['seed']+SECOND_AGENT_SEED_OFFSET
            prompt = payload['messages'][0]['content']
            a = row['choices']['qwen']['graph']
            program = row['choices']['program']['graph']
            legal = row['legal_options']
            scores = {'A': int(a in legal), 'program': int(program in legal)}
            choices = {}
            for role, suffix in (
                ('repeat', ''),
                ('B', '\nEvaluate the route from each positive intervention source to its required targets. '
                 'Favor a coordinated rewire that preserves those routes and introduces no forbidden reachability.'),
            ):
                call = request(session, folder/f'{slot:02}_{role}.json', prompt+suffix, seed, payload['format'])
                assert call['request']['model'] == MODEL
                pick = decode(call).get('choice')
                chosen = menu[pick-1] if type(pick) is int and 1 <= pick <= len(menu) else None
                probes = [a] + ([chosen] if chosen is not None and chosen != a else [])
                choices[role] = {'graph': chosen, 'probes': len(probes), 'found': sum(g in legal for g in probes),
                                 'new': int(chosen != a and chosen in legal), 'seconds': call['seconds']}
            records.append({'case': case, 'slot': slot, 'A': a, 'A_valid': bool(scores['A']), 'second': choices})
    summary = {role: {key: sum(row['second'][role][key] for row in records)
                      for key in ('probes', 'found', 'new', 'seconds')} for role in ('repeat', 'B')}
    write_json(output/'results.json', {'source_sha': digest(source_path), 'runner_sha': digest(__file__),
                                      'states': records, 'summary': summary})
    print(summary, flush=True)


if __name__ == '__main__':
    main()
