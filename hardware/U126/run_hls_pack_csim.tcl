set script_dir [file normalize [file dirname [info script]]]
if {![info exists ::env(FPT_DATA_DIR)]} {
  error "FPT_DATA_DIR must name the frozen L512 vector directory"
}
set data_dir [file normalize $::env(FPT_DATA_DIR)]
set ::env(PATH) "/usr/bin:/bin:$::env(PATH)"
set ::env(LD) "/usr/bin/ld"

cd $script_dir
open_project -reset fpt_u126_l512_runtime_length_no_q_zero_fill_pack_csim
set_top int8_gqa_prefill_decode_ip
add_files unified_ip.cpp -cflags "-std=c++14 -I."
add_files prefill_core.cpp -cflags "-std=c++14 -I."
add_files decode_core.cpp -cflags "-std=c++14 -I."
add_files -tb tb_prefill_core.cpp -cflags "-std=c++14 -I."
open_solution -reset solution1 -flow_target vivado
config_compile -pipeline_loops 0
set_part {xck26-sfvc784-2LV-c}
create_clock -period 5.0 -name default
csim_design -argv "$data_dir pack-only"
exit
