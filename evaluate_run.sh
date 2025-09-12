# python evaluate_if.py --framework ifbench \
# --output_file eval_res/ifbench_evaluation_results.json
# Instruction-level accuracy: 28.40%

# python evaluate_if.py --framework ifeval \
# --output_file eval_res/ifeval_evaluation_results.json
# Instruction-level accuracy: 84.44%

# python evaluate_logicif.py --output_file eval_res/logicif_evaluation_results.json
# # Task-level accuracy: 88.24%

for model in qwen38b; do
    for benchmark in ifbench ifeval logicifevalmini; do 
        if [ "$benchmark" = "ifbench" ] || [ "$benchmark" = "ifeval" ]; then
            python evaluate_if.py --framework ${benchmark} \
                    --input_file eval_res/${benchmark}-${model}.jsonl \
                    --output_file eval_res/${benchmark}-${model}-evaluation.json
        elif [ "$benchmark" = "logicifevalmini" ]; then
            python evaluate_logicif.py \
                    --input_file eval_res/${benchmark}-${model}.jsonl \
                    --output_file eval_res/${benchmark}-${model}-evaluation.json
        fi
    done
done
