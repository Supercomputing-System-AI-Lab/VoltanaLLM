import argparse
import orjson
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import ORJSONResponse, Response

app = FastAPI()

@app.get("/health")
async def health_check():
    return Response(status_code=200, content="OK")

@app.get("/get_model_info")
async def get_model_info():
    ret = '{"model_path":"meta-llama/Llama-3.1-8B-Instruct", "tokenizer_path":"meta-llama/Llama-3.1-8B-Instruct", "is_generation":true}'
    return ORJSONResponse(content=orjson.loads(ret), status_code=200, media_type="application/json")

@app.post("/set_freq_manager_state")
async def set_freq_manager_state(request: Request):
    return Response(status_code=200)

@app.post("/set_freq")
async def set_freq(request: Request):
    return Response(status_code=200)

@app.post("/unset_freq")
async def unset_freq():
    return Response(status_code=200)

@app.get("/get_server_info")
async def get_server_info():
    global num
    json_str = '{"model_path":"meta-llama/Llama-3.1-8B-Instruct","tokenizer_path":"meta-llama/Llama-3.1-8B-Instruct","tokenizer_mode":"auto","skip_tokenizer_init":false,"load_format":"auto","trust_remote_code":true,"dtype":"auto","kv_cache_dtype":"auto","quantization":null,"quantization_param_path":null,"context_length":null,"device":"cuda","served_model_name":"meta-llama/Llama-3.1-8B-Instruct","chat_template":null,"completion_template":null,"is_embedding":false,"enable_multimodal":null,"revision":null,"impl":"auto","host":"127.0.0.1","port":8200,"mem_fraction_static":0.9,"max_running_requests":512,"max_total_tokens":null,"chunked_prefill_size":8192,"max_prefill_tokens":16384,"schedule_policy":"fcfs","schedule_conservativeness":1.0,"cpu_offload_gb":0,"page_size":1,"tp_size":1,"pp_size":1,"max_micro_batch_size":null,"stream_interval":1,"stream_output":false,"random_seed":0,"constrained_json_whitespace_pattern":null,"watchdog_timeout":86400.0,"dist_timeout":null,"download_dir":null,"base_gpu_id":1,"gpu_id_step":1,"sleep_on_idle":false,"log_level":"info","log_level_http":null,"log_requests":false,"log_requests_level":0,"show_time_cost":false,"enable_metrics":false,"bucket_time_to_first_token":null,"bucket_e2e_request_latency":null,"bucket_inter_token_latency":null,"collect_tokens_histogram":false,"decode_log_interval":40,"enable_request_time_stats_logging":false,"kv_events_config":null,"api_key":null,"file_storage_path":"sglang_storage","enable_cache_report":false,"reasoning_parser":null,"tool_call_parser":null,"dp_size":1,"load_balance_method":"round_robin","dist_init_addr":null,"nnodes":1,"node_rank":0,"json_model_override_args":"{}","preferred_sampling_params":null,"lora_paths":null,"max_loras_per_batch":8,"lora_backend":"triton","attention_backend":"flashinfer","sampling_backend":"flashinfer","grammar_backend":"xgrammar","mm_attention_backend":null,"speculative_algorithm":null,"speculative_draft_model_path":null,"speculative_num_steps":null,"speculative_eagle_topk":null,"speculative_num_draft_tokens":null,"speculative_accept_threshold_single":1.0,"speculative_accept_threshold_acc":1.0,"speculative_token_map":null,"ep_size":1,"enable_ep_moe":false,"enable_deepep_moe":false,"deepep_mode":"auto","ep_num_redundant_experts":0,"ep_dispatch_algorithm":"static","init_expert_location":"trivial","enable_eplb":false,"eplb_algorithm":"auto","eplb_rebalance_num_iterations":1000,"eplb_rebalance_layers_per_chunk":null,"expert_distribution_recorder_mode":null,"expert_distribution_recorder_buffer_size":1000,"enable_expert_distribution_metrics":false,"deepep_config":null,"moe_dense_tp_size":null,"enable_double_sparsity":false,"ds_channel_config_path":null,"ds_heavy_channel_num":32,"ds_heavy_token_num":256,"ds_heavy_channel_type":"qk","ds_sparse_decode_threshold":4096,"disable_radix_cache":true,"cuda_graph_max_bs":null,"cuda_graph_bs":[1,2,4,8,16,24,32,40,48,56,64,72,80,88,96,104,112,120,128,192,256,320,384,448,512,576,640,704,768,832,896,960,1024],"disable_cuda_graph":false,"disable_cuda_graph_padding":false,"enable_profile_cuda_graph":false,"enable_nccl_nvls":false,"enable_tokenizer_batch_encode":false,"disable_outlines_disk_cache":false,"disable_custom_all_reduce":false,"enable_mscclpp":false,"disable_overlap_schedule":false,"disable_overlap_cg_plan":false,"enable_mixed_chunk":false,"enable_dp_attention":false,"enable_dp_lm_head":false,"enable_two_batch_overlap":false,"enable_torch_compile":false,"torch_compile_max_bs":32,"torchao_config":"","enable_nan_detection":false,"enable_p2p_check":false,"triton_attention_reduce_in_fp32":false,"triton_attention_num_kv_splits":8,"num_continuous_decode_steps":1,"delete_ckpt_after_loading":false,"enable_memory_saver":false,"allow_auto_truncate":false,"enable_custom_logit_processor":false,"enable_hierarchical_cache":false,"hicache_ratio":2.0,"hicache_size":0,"hicache_write_policy":"write_through_selective","flashinfer_mla_disable_ragged":false,"disable_shared_experts_fusion":false,"disable_chunked_prefix_cache":false,"disable_fast_image_processor":false,"enable_return_hidden_states":false,"warmups":null,"debug_tensor_dump_output_folder":null,"debug_tensor_dump_input_file":null,"debug_tensor_dump_inject":false,"debug_tensor_dump_prefill_only":false,"disaggregation_mode":"decode","disaggregation_transfer_backend":"fake","disaggregation_bootstrap_port":8998,"disaggregation_decode_tp":null,"disaggregation_decode_dp":null,"disaggregation_prefill_pp":1,"disaggregation_ib_device":"enP62544s1","num_reserved_decode_tokens":8,"pdlb_url":null,"num_forward_repeat":null,"collect_iteration_energy":false,"enable_energy_monitor":true,"energy_monitor_interval":0.1,"enable_freq_manager":true,"freq_manager_dummy_run":true,"freq_manager_interval":0.5,"freq_manager_lookup_table_path":"/u/jyu10/hpca/sglang-hpca/jiahuan2/lookup_table_707_13b.csv","freq_manager_f_high":1410,"freq_manager_f_low":1005,"freq_manager_slo_p50":50.0,"freq_manager_slo_p99":120.0,"freq_manager_gamma_p50":0.5,"freq_manager_gamma_p99":0.5,"freq_manager_rps_threshold":1000.0,"freq_manager_rps_window":5.0,"status":"ready","max_total_num_tokens":163642,"max_req_input_len":131066,"internal_states":[{"attention_backend":"flashinfer","mm_attention_backend":null,"debug_tensor_dump_inject":false,"debug_tensor_dump_output_folder":null,"chunked_prefill_size":8192,"device":"cuda","disable_chunked_prefix_cache":true,"disable_radix_cache":true,"enable_dp_attention":false,"enable_two_batch_overlap":false,"enable_dp_lm_head":false,"enable_deepep_moe":false,"deepep_mode":"auto","enable_ep_moe":false,"moe_dense_tp_size":null,"ep_dispatch_algorithm":"static","deepep_config":null,"ep_num_redundant_experts":0,"enable_nan_detection":false,"flashinfer_mla_disable_ragged":false,"max_micro_batch_size":512,"disable_shared_experts_fusion":false,"sampling_backend":"flashinfer","speculative_accept_threshold_acc":1.0,"speculative_accept_threshold_single":1.0,"torchao_config":"","triton_attention_reduce_in_fp32":false,"num_reserved_decode_tokens":8,"use_mla_backend":false,"last_gen_throughput":0.0,"load":0,"bs":' + str(num) + ',"num_running_tokens":0,"len_waiting_queue":0,"last_itl": 0.334,"len_decode_prealloc_queue":0,"len_decode_transfer_queue":0,"len_decode_retracted_queue":0,"gpu_freq":1410,"gpu_power":57651,"gpu_energy":132907453,"freq":1410, "time": 114514, "slo": 50}],"version":"0.4.7.post1"}'
    ret_json = orjson.loads(json_str)
    return ORJSONResponse(content=ret_json, status_code=200, media_type="application/json")

num = 0

@app.post("/generate")
async def generate():
    # await asyncio.sleep(3)  # Simulate some processing delay
    data = {
        'text': 'nop ',
        'meta_info': {
            'id': '3ee9b4aeed0549e8a3d9bce3078cdb2f',
            'finish_reason': None,
            'prompt_tokens': 14,
            'completion_tokens': 26,
            'cached_tokens': 0,
            'extra_batch_info': {
                'iteration_counter': 122672755,
                'timestamp_begin': 2124328.435999897,
                'timestamp_after_schedule': 2124328.436480898,
                'timestamp_before_forward': 2124328.475036341,
                'timestamp_after_forward': 2124328.497071416,
                'timestamp_before_worker_iteration': 2124328.474883278,
                'timestamp_after_worker_iteration': 2124328.497135535,
                'timestamp_before_output': 2124328.492836641,
                'num_forward_repeat': 1,
                'energy_forward': None,
                'time_forward_cuda_event': None,
                'batch_size': 15,
                'num_total_computed_tokens_list': [80, 90, 328, 82, 86, 212, 486, 194, 2120, 278, 2103, 54, 37, 64, 14],
                'forward_mode': 2
            },
            'e2e_latency': 999,
            'cur_token_time': 2124328.497640862
        },
        'cur_token_time_list': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'non_stream_metrics': {
            "ttft": 233,
            "itl_list": [233, 233],
        }
    }
    return ORJSONResponse(content=data, status_code=200, media_type="application/json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dummy Server for Testing")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address to bind the server")
    parser.add_argument("--port", type=int, default=8000, help="Port number to bind the server")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
