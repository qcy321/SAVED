# coding: utf-8

def consensus_mechanism(ds_result_path, qw_result_path, input_path, output_path):
    with open(ds_result_path, 'r', encoding='utf-8') as f_ds, \
            open(qw_result_path, 'r', encoding='utf-8') as f_wq, \
            open(input_path, 'r', encoding='utf-8') as f_input:

        ds_results = [line.strip() for line in f_ds]
        qw_results = [line.strip() for line in f_wq]
        input_records = [line.strip() for line in f_input]

    if len(ds_results) != len(qw_results) or len(ds_results) != len(input_records):
        print(
            f"Error: Mismatched lengths - ds_results: {len(ds_results)}, wq_results: {len(qw_results)}, input_records: {len(input_records)}")
        return

    common_responses = []
    for idx, (ds_res, wq_res, record) in enumerate(zip(ds_results, qw_results, input_records), 1):
        if ds_res == "[响应]" and wq_res == "[响应]":
            common_responses.append(record)
            print(f"Record {idx}: 均为响应 - {record}")

    with open(output_path, 'w', encoding='utf-8') as f_out:
        f_out.write('\n'.join(common_responses) + '\n')

    print(f"{len(common_responses)} 条记录均为响应.")
    print(f"已保存:  {output_path}")


if __name__ == '__main__':
    ds_result_path = '../data/pair/tencent/ds_result.txt'
    qw_result_path = '../data/pair/tencent/qw_result.txt'
    input_path = '../data/pair/tencent/candidate_pair.txt'
    output_path = '../data/pair/tencent/consensus_result.txt'

    consensus_mechanism(ds_result_path, qw_result_path, input_path, output_path)
