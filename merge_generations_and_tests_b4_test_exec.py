'''
This script takes as input the knowledge base of the code solutions and
the knowledge base of the test generations, and merges them to create a new knowledge base 
where each entry is a <code_solution, test_statement> pair. 
'''
import os
import re
import json
import pandas as pd
import sys
import gzip
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--auto-save', action='store_true', help='Automatically save merged data without asking for confirmation')
args = parser.parse_args()

if __name__ == "__main__":
    tqdm.pandas()
    
    # IMPORT CODE KNOWLEDGE BASE
    knowlbase_path = "./Evalplus/knowlbase/knowlbase_evalplus.json"
    # import code knowledge base
    with open(knowlbase_path, 'r') as f:
        kb = [inst for inst in json.load(f)]
    kb = pd.DataFrame(kb)
    kb = kb[['solution_idx', 'task_idx', 'generated_by', 'prompt', 'method']]
    kb['onwhichtomerge'] = kb['solution_idx'].apply(lambda x: x.split("--SampleID")[0])
    
    # FILTER KNOWLEDGE BASE TO TEST SET SOLUTIONS
    with open("../data/testset/test.jsonl", "r") as f:
        test_set_uptp20 = [json.loads(line) for line in f]
    test_set_uptp20 = pd.DataFrame(test_set_uptp20)
    solution_ids_to_keep = test_set_uptp20.solution_idx.unique().tolist()
    kb = kb[kb.solution_idx.isin(solution_ids_to_keep)]
    del test_set_uptp20
    del solution_ids_to_keep

    # SPLIT BY GENERATED_BY
    for generated_by in kb['generated_by'].unique():

        # IMPORT TESTS KNOWLEDGE BASE
        test_path = f"../data/knowlbase-tests/{generated_by}_knowlbase_tests_evalplus.jsonl"
        if not os.path.exists(test_path):
            print(f"Test knowledge base file '{test_path}' not found. Skipping '{generated_by}'.")
            continue
        kbt_genby = pd.read_json(test_path, lines=True)
        kbt_genby['onwhichtomerge'] = kbt_genby['test_idx'].apply(lambda x: x.split("--TestID")[0])
        kbt_genby = kbt_genby.drop(columns=['task_idx', 'generated_by'])

        kb_genby = kb.loc[kb['generated_by'] == generated_by]
    
        merged = kb_genby.merge(kbt_genby, on="onwhichtomerge", how="inner")

        ###############
        num_task_ids = json.load(open('../../../constants/ids_train_val_test.json', 'r'))['Evalplus']['test']
        g = merged.groupby('solution_idx', as_index=False).agg({'test_idx': 'count', 'task_idx': 'first'})
        print(f"################## {generated_by} ##################")
        print(kb_genby.loc[kb_genby['task_idx'].isin(num_task_ids)]['task_idx'].nunique())
        print(f"Max number of unique asserts per task ID:\t\t{g['test_idx'].max()}")
        print(f"Min number of unique asserts per task ID:\t\t{g['test_idx'].min()}")
        print(f"Average number of unique asserts per task ID:\t\t{g['test_idx'].mean():.2f}")
        print(f"Median number of unique asserts per task ID:\t\t{g['test_idx'].median()}")
        print(f"Number of task IDs with no asserts (out of {len(num_task_ids)}):\t\t{len(num_task_ids) - g['task_idx'].nunique()}")
        print()
        ###############

        # unique id of the pair <code_solution, test_statement>
        merged['test_execution_idx'] = merged.apply(lambda row : \
        '{}--TestID::{}--SampleID::{}'.format(
                row['solution_idx'].split('--SampleID::')[0],
                row['test_idx'].split('--TestID::')[-1],
                row['solution_idx'].split('--SampleID::')[-1]
            ), axis=1
        )

        merged = merged[[
            "task_idx",
            "generated_by",
            "prompt",
            "test_idx",
            "test_statement",
            "method",
            "test_execution_idx"
        ]].rename(columns={
            "method" : "solution",
            "task_idx" : "task_id"
            })

        
        # authorize saving form input
        save_dir = "../test-4-execution"
        authorized = args.auto_save
        if not authorized:
            authorized = input(f"Do you want to save the tests extracted for {generated_by} to '{save_dir}'? (y/n): ")
            authorized = authorized.lower() in ['y', 'yes']
    
        if authorized:
            # chunk merged and save to json
            chunk_size = 10000
            num_chunks = (len(merged) + chunk_size - 1) // chunk_size
            print(f"Saving merged data to {save_dir} in {num_chunks} chunks...")
            for i in tqdm(range(num_chunks)):
                chunk = merged.iloc[i*chunk_size:(i+1)*chunk_size]
                chunk.to_json(os.path.join(save_dir, f"{generated_by}_test4execution_{i:02d}.jsonl"), lines=True, orient="records")
        else:
            print("Skipped.")
