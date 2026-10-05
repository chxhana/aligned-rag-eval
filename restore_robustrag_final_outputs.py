"""Recover full final aggregation outputs from saved traces, without model inference."""
import argparse
import copy
from pathlib import Path
from exp11_ablation import fingerprint,save_json
from aligned_rag.data import read


def restore(source):
    if not source.get('complete') or source['records_sha256']!=fingerprint(source['records']):
        raise ValueError('Incomplete or modified source records')
    if source['config']['method']!='robustrag_keyword':raise ValueError('Expected RobustRAG keyword output')
    if 'postprocessing_repair' in source['config']:raise ValueError('Source already repaired')
    result=copy.deepcopy(source);count=0;isolated_changed=0
    for key,row in result['records'].items():
        n=len(row['texts']);trace=row['trace']
        if len(trace) not in (n,n+1):raise ValueError(f'Unexpected trace structure {key}')
        for call in trace[:n]:
            isolated_changed+=call['raw_output'].strip()!=call['output']
        if len(trace)==n+1:
            final=trace[-1]
            if row['output']!=final['output']:raise ValueError('Final trace and saved answer disagree')
            raw=final['raw_output'].strip()
            count+=raw!=row['output']
            row['pre_repair_output']=row['output']
            row['output']=raw
            final['pre_repair_output']=final['output']
            final['output']=raw
            final['original_first_line_only']=final['first_line_only']
            final['first_line_only']=False
    result['config']['postprocessing_repair']={
        'method':'restore_full_final_aggregation_from_saved_raw_output',
        'source_signature':source['signature'],'source_records_sha256':source['records_sha256'],
        'changed_final_outputs':count,'isolated_outputs_previously_trimmed':isolated_changed,
        'scope':'Final response only. Original isolated responses and aggregation prompts are unchanged; this is not a rerun with different isolated answers.',
        'labels':'Still requires semantic endorsement/answer evaluation; restoring text is not adjudication.'}
    result['signature']=fingerprint(result['config'])
    result['records_sha256']=fingerprint(result['records'])
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True);p.add_argument('--out',required=True)
    args=p.parse_args()
    if Path(args.out).exists():raise ValueError('Output exists; preserve original and previous repairs')
    result=restore(read(args.input));save_json(args.out,result)
    print(result['config']['postprocessing_repair'])
    print('[out]',args.out)

if __name__=='__main__':main()
