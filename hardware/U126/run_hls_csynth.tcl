set script_dir [file normalize [file dirname [info script]]]

cd $script_dir
open_project -reset fpt_u126_l512_runtime_length_no_q_zero_fill
set_top int8_gqa_prefill_decode_ip
add_files unified_ip.cpp -cflags "-std=c++14 -I."
add_files prefill_core.cpp -cflags "-std=c++14 -I."
add_files decode_core.cpp -cflags "-std=c++14 -I."
open_solution -reset solution1 -flow_target vivado
config_compile -pipeline_loops 0
config_export -format xo
set_part {xck26-sfvc784-2LV-c}
create_clock -period 5.0 -name default
csynth_design
export_design -format xo
exit
