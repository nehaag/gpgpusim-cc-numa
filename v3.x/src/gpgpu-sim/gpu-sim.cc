// Copyright (c) 2009-2011, Tor M. Aamodt, Wilson W.L. Fung, George L. Yuan,
// Ali Bakhoda, Andrew Turner, Ivan Sham
// The University of British Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice, this
// list of conditions and the following disclaimer in the documentation and/or
// other materials provided with the distribution.
// Neither the name of The University of British Columbia nor the names of its
// contributors may be used to endorse or promote products derived from this
// software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


#include "gpu-sim.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include <sstream>
#include <string>
#include <map>
#include <bitset>
#include "zlib.h"


#include "shader.h"
#include "dram.h"
#include "mem_fetch.h"

#include <time.h>
#include "gpu-cache.h"
#include "gpu-misc.h"
#include "delayqueue.h"
#include "shader.h"
#include "icnt_wrapper.h"
#include "dram.h"
#include "addrdec.h"
#include "stat-tool.h"
#include "l2cache.h"

#include "../cuda-sim/ptx-stats.h"
#include "../statwrapper.h"
#include "../abstract_hardware_model.h"
#include "../debug.h"
#include "../gpgpusim_entrypoint.h"
#include "../cuda-sim/cuda-sim.h"
#include "../trace.h"
#include "mem_latency_stat.h"
#include "power_stat.h"
#include "visualizer.h"
#include "stats.h"

#ifdef GPGPUSIM_POWER_MODEL
#include "power_interface.h"
#else
class  gpgpu_sim_wrapper {};
#endif

#include <stdio.h>
#include <string.h>
#include <iostream>
#include <sstream>
#include <string>

#define MAX(a,b) (((a)>(b))?(a):(b))


bool g_interactive_debugger_enabled=false;

unsigned long long  gpu_sim_cycle = 0;
unsigned long long  gpu_tot_sim_cycle = 0;
unsigned long long last_updated_at = 0;
unsigned long long pageBlockingStall = 0;
unsigned int bw_equal = 0;
//bool enableMigration = true;
bool enableMigration;
bool pauseMigration = false;
/* Implementing a state machine for migration
 * 3 state FSM: evicting -> probing mshr -> migrating
 * 1. evicting: evict all dirty lines by trying to find mshrs
 * 2. probing: probe only mshrs of l1 and l2 until no reqs left... also
 * mark all read reqs to this page as bypass so that they dont get inserted
 * 3. migrating: migrate
 * migrationQueue data structure is a map, with memory address as the key and
 * state as the value
 */
typedef unsigned long long new_addr_type;
std::map<unsigned, std::list<unsigned long long> >sendForMigrationPid;
std::map<unsigned long long, uint64_t> migrationQueue;
std::map<unsigned long long, unsigned> migrationWaitCycle;
std::map<unsigned long long, std::array<unsigned long long, 10> > migrationFinished;
std::map<unsigned long long, std::array<unsigned long long, 3> > accessDistribution;
std::map<unsigned long long, unsigned> reCheckForMigration;
std::map<unsigned long long, std::map<unsigned, unsigned> > globalPageCount;
bool readyForNextMigration[4] = {true, true, true, true};

/* request_uid->address map*/
std::map<unsigned, std::pair<new_addr_type, unsigned> >  l1_wr_miss_no_wa_map;
std::map<unsigned, new_addr_type>  l1_wb_map;
std::map<unsigned, new_addr_type>  l2_wb_map;
unsigned int migration_threshold;
// TODO: merge it with the one in the config file
unsigned int migrationThreshold = 1;
int range_expansion;
unsigned int max_migrations;
unsigned int migration_cost;
bool magical_migration;
bool flush_on_migration_enable;
bool block_on_migration;
bool limit_migration_rate;
bool drain_all_mshrs;

// performance counter for stalls due to congestion.
unsigned int gpu_stall_dramfull = 0; 
unsigned int gpu_stall_icnt2sh = 0;

/* Clock Domains */

#define  CORE  0x01
#define  L2    0x02
#define  DRAM  0x04
#define  ICNT  0x08  
#define  DRAM_t2  0x10


#define MEM_LATENCY_STAT_IMPL




#include "mem_latency_stat.h"

void migration_reg_options(class OptionParser * opp) {

    option_parser_register(opp, "-enable_migration", OPT_BOOL,
            &enableMigration, "whether to enable migration or not",
            "false");
    option_parser_register(opp, "-migration_threshold", OPT_UINT32,
            &migration_threshold, "minimum number of touches to a 4kB dram page",
            "128");
    option_parser_register(opp, "-range_expansion", OPT_INT32,
            &range_expansion, "number of neighbouring pages to be migrated",
            "4");
    option_parser_register(opp, "-max_migrations", OPT_INT32,
            &max_migrations, "maximum number of pages to be migrated with the specified range",
            "0");
    option_parser_register(opp, "-migration_cost", OPT_UINT32,
            &migration_cost, "cost of migration to include TLB flushing etc",
            "1000");
    option_parser_register(opp, "-magical_migration", OPT_BOOL,
            &magical_migration, "no overhead of putting migration requests into memory controllers, jsut change the mapping and migrate instantaneously",
            "false");
    option_parser_register(opp, "-flush_on_migration_enable", OPT_BOOL,
            &flush_on_migration_enable, "flush the caches on migration, and fake the migration by changing mem_fetch mapping whenever possible",
            "true");
    option_parser_register(opp, "-block_on_migration", OPT_BOOL,
            &block_on_migration, "block any request to the migrating page",
            "true");
    option_parser_register(opp, "-limit_migration_rate", OPT_BOOL,
            &limit_migration_rate, "enable optimal bw-ratio cap on migration",
            "true");
    option_parser_register(opp, "-drain_all_mshrs", OPT_BOOL,
            &drain_all_mshrs, "drain all the mshrs on tlb shootdown",
            "false");
}


void power_config::reg_options(class OptionParser * opp)
{


	  option_parser_register(opp, "-gpuwattch_xml_file", OPT_CSTR,
			  	  	  	  	 &g_power_config_name,"GPUWattch XML file",
	                   "gpuwattch.xml");

	   option_parser_register(opp, "-power_simulation_enabled", OPT_BOOL,
	                          &g_power_simulation_enabled, "Turn on power simulator (1=On, 0=Off)",
	                          "0");

	   option_parser_register(opp, "-power_per_cycle_dump", OPT_BOOL,
	                          &g_power_per_cycle_dump, "Dump detailed power output each cycle",
	                          "0");

	   // Output Data Formats
	   option_parser_register(opp, "-power_trace_enabled", OPT_BOOL,
	                          &g_power_trace_enabled, "produce a file for the power trace (1=On, 0=Off)",
	                          "0");

	   option_parser_register(opp, "-power_trace_zlevel", OPT_INT32,
	                          &g_power_trace_zlevel, "Compression level of the power trace output log (0=no comp, 9=highest)",
	                          "6");

	   option_parser_register(opp, "-steady_power_levels_enabled", OPT_BOOL,
	                          &g_steady_power_levels_enabled, "produce a file for the steady power levels (1=On, 0=Off)",
	                          "0");

	   option_parser_register(opp, "-steady_state_definition", OPT_CSTR,
			   	  &gpu_steady_state_definition, "allowed deviation:number of samples",
	                 	  "8:4");

}

void memory_config::reg_options(class OptionParser * opp, unsigned num)
{
    char num_str[10];
    sprintf(num_str, "%d", num);
    type = num;

    option_parser_register_mem(opp, "-gpgpu_dram_scheduler", OPT_INT32, &scheduler_type, 
                                "0 = fifo, 1 = FR-FCFS (defaul)", "1", num_str);
    option_parser_register_mem(opp, "-gpgpu_dram_partition_queues", OPT_CSTR, &gpgpu_L2_queue_config, 
                           "i2$:$2d:d2$:$2i",
                           "8:8:8:8", num_str);

    option_parser_register_mem(opp, "-l2_ideal", OPT_BOOL, &l2_ideal, 
                           "Use a ideal L2 cache that always hit",
                           "0", num_str);
    option_parser_register_mem(opp, "-gpgpu_cache:dl2", OPT_CSTR, &m_L2_config.m_config_string, 
                   "unified banked L2 data cache config "
                   " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_alloc>,<mshr>:<N>:<merge>,<mq>}",
                   "64:128:8,L:B:m:N,A:16:4,4", num_str);
    option_parser_register_mem(opp, "-gpgpu_cache:dl2_texture_only", OPT_BOOL, &m_L2_texure_only, 
                           "L2 cache used for texture only",
                           "1", num_str);
    option_parser_register_mem(opp, "-gpgpu_n_mem", OPT_UINT32, &m_n_mem, 
                 "number of memory modules (e.g. memory controllers) in gpu",
                 "8", num_str);
    option_parser_register_mem(opp, "-gpgpu_n_sub_partition_per_mchannel", OPT_UINT32, &m_n_sub_partition_per_memory_channel, 
                 "number of memory subpartition in each memory module",
                 "1", num_str);
    option_parser_register_mem(opp, "-gpgpu_n_mem_per_ctrlr", OPT_UINT32, &gpu_n_mem_per_ctrlr, 
                 "number of memory chips per memory controller",
                 "1", num_str);
    option_parser_register_mem(opp, "-gpgpu_memlatency_stat", OPT_INT32, &gpgpu_memlatency_stat, 
                "track and display latency statistics 0x2 enables MC, 0x4 enables queue logs",
                "0", num_str);
    option_parser_register_mem(opp, "-gpgpu_frfcfs_dram_sched_queue_size", OPT_INT32, &gpgpu_frfcfs_dram_sched_queue_size, 
                "0 = unlimited (default); # entries per chip",
                "0", num_str);
    option_parser_register_mem(opp, "-gpgpu_dram_return_queue_size", OPT_INT32, &gpgpu_dram_return_queue_size, 
                "0 = unlimited (default); # entries per chip",
                "0", num_str);
    option_parser_register_mem(opp, "-gpgpu_dram_buswidth", OPT_UINT32, &busW, 
                 "default = 4 bytes (8 bytes per cycle at DDR)",
                 "4", num_str);
    option_parser_register_mem(opp, "-gpgpu_dram_burst_length", OPT_UINT32, &BL, 
                 "Burst length of each DRAM request (default = 4 data bus cycle)",
                 "4", num_str);
    option_parser_register_mem(opp, "-dram_data_command_freq_ratio", OPT_UINT32, &data_command_freq_ratio, 
                 "Frequency ratio between DRAM data bus and command bus (default = 2 times, i.e. DDR)",
                 "2", num_str);
    option_parser_register_mem(opp, "-gpgpu_dram_timing_opt", OPT_CSTR, &gpgpu_dram_timing_opt, 
                "DRAM timing parameters = {nbk:tCCD:tRRD:tRCD:tRAS:tRP:tRC:CL:WL:tCDLR:tWR:nbkgrp:tCCDL:tRTPL}",
                "4:2:8:12:21:13:34:9:4:5:13:1:0:0", num_str);
    option_parser_register_mem(opp, "-rop_latency", OPT_UINT32, &rop_latency,
                     "ROP queue latency (default 85)",
                     "85", num_str);
    option_parser_register_mem(opp, "-dram_latency", OPT_UINT32, &dram_latency,
                     "DRAM latency (default 30)",
                     "30", num_str);
    option_parser_register_mem(opp, "-addr_limit", OPT_UINT32, &addr_limit,
                     "Address above this to be maaped to different type of memory",
                     "0", num_str);

    m_address_mapping.addrdec_setoption(opp, num_str);
}

void shader_core_config::reg_options(class OptionParser * opp)
{
    option_parser_register(opp, "-gpgpu_simd_model", OPT_INT32, &model, 
                   "1 = post-dominator", "1");
    option_parser_register(opp, "-gpgpu_shader_core_pipeline", OPT_CSTR, &gpgpu_shader_core_pipeline_opt, 
                   "shader core pipeline config, i.e., {<nthread>:<warpsize>}",
                   "1024:32");
    option_parser_register(opp, "-gpgpu_tex_cache:l1", OPT_CSTR, &m_L1T_config.m_config_string, 
                   "per-shader L1 texture cache  (READ-ONLY) config "
                   " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_alloc>,<mshr>:<N>:<merge>,<mq>:<rf>}",
                   "8:128:5,L:R:m:N,F:128:4,128:2");
    option_parser_register(opp, "-gpgpu_const_cache:l1", OPT_CSTR, &m_L1C_config.m_config_string, 
                   "per-shader L1 constant memory cache  (READ-ONLY) config "
                   " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_alloc>,<mshr>:<N>:<merge>,<mq>} ",
                   "64:64:2,L:R:f:N,A:2:32,4" );
    option_parser_register(opp, "-gpgpu_cache:il1", OPT_CSTR, &m_L1I_config.m_config_string, 
                   "shader L1 instruction cache config "
                   " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_alloc>,<mshr>:<N>:<merge>,<mq>} ",
                   "4:256:4,L:R:f:N,A:2:32,4" );
    option_parser_register(opp, "-gpgpu_cache:dl1", OPT_CSTR, &m_L1D_config.m_config_string,
                   "per-shader L1 data cache config "
                   " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_alloc>,<mshr>:<N>:<merge>,<mq> | none}",
                   "none" );
    option_parser_register(opp, "-gpgpu_cache:dl1PrefL1", OPT_CSTR, &m_L1D_config.m_config_stringPrefL1,
                   "per-shader L1 data cache config "
                   " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_alloc>,<mshr>:<N>:<merge>,<mq> | none}",
                   "none" );
    option_parser_register(opp, "-gpgpu_cache:dl1PreShared", OPT_CSTR, &m_L1D_config.m_config_stringPrefShared,
                   "per-shader L1 data cache config "
                   " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_alloc>,<mshr>:<N>:<merge>,<mq> | none}",
                   "none" );
    option_parser_register(opp, "-gmem_skip_L1D", OPT_BOOL, &gmem_skip_L1D, 
                   "global memory access skip L1D cache (implements -Xptxas -dlcm=cg, default=no skip)",
                   "0");

    option_parser_register(opp, "-gpgpu_perfect_mem", OPT_BOOL, &gpgpu_perfect_mem, 
                 "enable perfect memory mode (no cache miss)",
                 "0");
    option_parser_register(opp, "-n_regfile_gating_group", OPT_UINT32, &n_regfile_gating_group,
                 "group of lanes that should be read/written together)",
                 "4");
    option_parser_register(opp, "-gpgpu_clock_gated_reg_file", OPT_BOOL, &gpgpu_clock_gated_reg_file,
                 "enable clock gated reg file for power calculations",
                 "0");
    option_parser_register(opp, "-gpgpu_clock_gated_lanes", OPT_BOOL, &gpgpu_clock_gated_lanes,
                 "enable clock gated lanes for power calculations",
                 "0");
    option_parser_register(opp, "-gpgpu_shader_registers", OPT_UINT32, &gpgpu_shader_registers, 
                 "Number of registers per shader core. Limits number of concurrent CTAs. (default 8192)",
                 "8192");
    option_parser_register(opp, "-gpgpu_shader_cta", OPT_UINT32, &max_cta_per_core, 
                 "Maximum number of concurrent CTAs in shader (default 8)",
                 "8");
    option_parser_register(opp, "-gpgpu_n_clusters", OPT_UINT32, &n_simt_clusters, 
                 "number of processing clusters",
                 "10");
    option_parser_register(opp, "-gpgpu_n_cores_per_cluster", OPT_UINT32, &n_simt_cores_per_cluster, 
                 "number of simd cores per cluster",
                 "3");
    option_parser_register(opp, "-gpgpu_n_cluster_ejection_buffer_size", OPT_UINT32, &n_simt_ejection_buffer_size, 
                 "number of packets in ejection buffer",
                 "8");
    option_parser_register(opp, "-gpgpu_n_ldst_response_buffer_size", OPT_UINT32, &ldst_unit_response_queue_size, 
                 "number of response packets in ld/st unit ejection buffer",
                 "2");
    option_parser_register(opp, "-gpgpu_shmem_size", OPT_UINT32, &gpgpu_shmem_size,
                 "Size of shared memory per shader core (default 16kB)",
                 "16384");
    option_parser_register(opp, "-gpgpu_shmem_size", OPT_UINT32, &gpgpu_shmem_sizeDefault,
                 "Size of shared memory per shader core (default 16kB)",
                 "16384");
    option_parser_register(opp, "-gpgpu_shmem_size_PrefL1", OPT_UINT32, &gpgpu_shmem_sizePrefL1,
                 "Size of shared memory per shader core (default 16kB)",
                 "16384");
    option_parser_register(opp, "-gpgpu_shmem_size_PrefShared", OPT_UINT32, &gpgpu_shmem_sizePrefShared,
                 "Size of shared memory per shader core (default 16kB)",
                 "16384");
    option_parser_register(opp, "-gpgpu_shmem_num_banks", OPT_UINT32, &num_shmem_bank, 
                 "Number of banks in the shared memory in each shader core (default 16)",
                 "16");
    option_parser_register(opp, "-gpgpu_shmem_limited_broadcast", OPT_BOOL, &shmem_limited_broadcast, 
                 "Limit shared memory to do one broadcast per cycle (default on)",
                 "1");
    option_parser_register(opp, "-gpgpu_shmem_warp_parts", OPT_INT32, &mem_warp_parts,  
                 "Number of portions a warp is divided into for shared memory bank conflict check ",
                 "2");
    option_parser_register(opp, "-gpgpu_warpdistro_shader", OPT_INT32, &gpgpu_warpdistro_shader, 
                "Specify which shader core to collect the warp size distribution from", 
                "-1");
    option_parser_register(opp, "-gpgpu_warp_issue_shader", OPT_INT32, &gpgpu_warp_issue_shader, 
                "Specify which shader core to collect the warp issue distribution from", 
                "0");
    option_parser_register(opp, "-gpgpu_local_mem_map", OPT_BOOL, &gpgpu_local_mem_map, 
                "Mapping from local memory space address to simulated GPU physical address space (default = enabled)", 
                "1");
    option_parser_register(opp, "-gpgpu_num_reg_banks", OPT_INT32, &gpgpu_num_reg_banks, 
                "Number of register banks (default = 8)", 
                "8");
    option_parser_register(opp, "-gpgpu_reg_bank_use_warp_id", OPT_BOOL, &gpgpu_reg_bank_use_warp_id,
             "Use warp ID in mapping registers to banks (default = off)",
             "0");
    option_parser_register(opp, "-gpgpu_operand_collector_num_units_sp", OPT_INT32, &gpgpu_operand_collector_num_units_sp,
                "number of collector units (default = 4)", 
                "4");
    option_parser_register(opp, "-gpgpu_operand_collector_num_units_sfu", OPT_INT32, &gpgpu_operand_collector_num_units_sfu,
                "number of collector units (default = 4)", 
                "4");
    option_parser_register(opp, "-gpgpu_operand_collector_num_units_mem", OPT_INT32, &gpgpu_operand_collector_num_units_mem,
                "number of collector units (default = 2)", 
                "2");
    option_parser_register(opp, "-gpgpu_operand_collector_num_units_gen", OPT_INT32, &gpgpu_operand_collector_num_units_gen,
                "number of collector units (default = 0)", 
                "0");
    option_parser_register(opp, "-gpgpu_operand_collector_num_in_ports_sp", OPT_INT32, &gpgpu_operand_collector_num_in_ports_sp,
                           "number of collector unit in ports (default = 1)", 
                           "1");
    option_parser_register(opp, "-gpgpu_operand_collector_num_in_ports_sfu", OPT_INT32, &gpgpu_operand_collector_num_in_ports_sfu,
                           "number of collector unit in ports (default = 1)", 
                           "1");
    option_parser_register(opp, "-gpgpu_operand_collector_num_in_ports_mem", OPT_INT32, &gpgpu_operand_collector_num_in_ports_mem,
                           "number of collector unit in ports (default = 1)", 
                           "1");
    option_parser_register(opp, "-gpgpu_operand_collector_num_in_ports_gen", OPT_INT32, &gpgpu_operand_collector_num_in_ports_gen,
                           "number of collector unit in ports (default = 0)", 
                           "0");
    option_parser_register(opp, "-gpgpu_operand_collector_num_out_ports_sp", OPT_INT32, &gpgpu_operand_collector_num_out_ports_sp,
                           "number of collector unit in ports (default = 1)", 
                           "1");
    option_parser_register(opp, "-gpgpu_operand_collector_num_out_ports_sfu", OPT_INT32, &gpgpu_operand_collector_num_out_ports_sfu,
                           "number of collector unit in ports (default = 1)", 
                           "1");
    option_parser_register(opp, "-gpgpu_operand_collector_num_out_ports_mem", OPT_INT32, &gpgpu_operand_collector_num_out_ports_mem,
                           "number of collector unit in ports (default = 1)", 
                           "1");
    option_parser_register(opp, "-gpgpu_operand_collector_num_out_ports_gen", OPT_INT32, &gpgpu_operand_collector_num_out_ports_gen,
                           "number of collector unit in ports (default = 0)", 
                           "0");
    option_parser_register(opp, "-gpgpu_coalesce_arch", OPT_INT32, &gpgpu_coalesce_arch, 
                            "Coalescing arch (default = 13, anything else is off for now)", 
                            "13");
    option_parser_register(opp, "-gpgpu_num_sched_per_core", OPT_INT32, &gpgpu_num_sched_per_core, 
                            "Number of warp schedulers per core", 
                            "1");
    option_parser_register(opp, "-gpgpu_max_insn_issue_per_warp", OPT_INT32, &gpgpu_max_insn_issue_per_warp,
                            "Max number of instructions that can be issued per warp in one cycle by scheduler",
                            "2");
    option_parser_register(opp, "-gpgpu_simt_core_sim_order", OPT_INT32, &simt_core_sim_order,
                            "Select the simulation order of cores in a cluster (0=Fix, 1=Round-Robin)",
                            "1");
    option_parser_register(opp, "-gpgpu_pipeline_widths", OPT_CSTR, &pipeline_widths_string,
                            "Pipeline widths "
                            "ID_OC_SP,ID_OC_SFU,ID_OC_MEM,OC_EX_SP,OC_EX_SFU,OC_EX_MEM,EX_WB",
                            "1,1,1,1,1,1,1" );
    option_parser_register(opp, "-gpgpu_num_sp_units", OPT_INT32, &gpgpu_num_sp_units,
                            "Number of SP units (default=1)",
                            "1");
    option_parser_register(opp, "-gpgpu_num_sfu_units", OPT_INT32, &gpgpu_num_sfu_units,
                            "Number of SF units (default=1)",
                            "1");
    option_parser_register(opp, "-gpgpu_num_mem_units", OPT_INT32, &gpgpu_num_mem_units,
                            "Number if ldst units (default=1) WARNING: not hooked up to anything",
                             "1");
    option_parser_register(opp, "-gpgpu_scheduler", OPT_CSTR, &gpgpu_scheduler_string,
                                "Scheduler configuration: < lrr | gto | two_level_active > "
                                "If two_level_active:<num_active_warps>:<inner_prioritization>:<outer_prioritization>"
                                "For complete list of prioritization values see shader.h enum scheduler_prioritization_type"
                                "Default: gto",
                                 "gto");
}

void gpgpu_sim_config::reg_options(option_parser_t opp)
{
    gpgpu_functional_sim_config::reg_options(opp);
    m_shader_config.reg_options(opp);
    m_memory_config.reg_options(opp);
    power_config::reg_options(opp);

    migration_reg_options(opp);

   option_parser_register(opp, "-gpgpu_max_cycle", OPT_INT32, &gpu_max_cycle_opt, 
               "terminates gpu simulation early (0 = no limit)",
               "0");
   option_parser_register(opp, "-gpgpu_max_insn", OPT_INT32, &gpu_max_insn_opt, 
               "terminates gpu simulation early (0 = no limit)",
               "0");
   option_parser_register(opp, "-gpgpu_max_cta", OPT_INT32, &gpu_max_cta_opt, 
               "terminates gpu simulation early (0 = no limit)",
               "0");
   option_parser_register(opp, "-gpgpu_runtime_stat", OPT_CSTR, &gpgpu_runtime_stat, 
                  "display runtime statistics such as dram utilization {<freq>:<flag>}",
                  "10000:0");
   option_parser_register(opp, "-liveness_message_freq", OPT_INT64, &liveness_message_freq, 
               "Minimum number of seconds between simulation liveness messages (0 = always print)",
               "1");
   option_parser_register(opp, "-gpgpu_flush_l1_cache", OPT_BOOL, &gpgpu_flush_l1_cache,
                "Flush L1 cache at the end of each kernel call",
                "0");
   option_parser_register(opp, "-gpgpu_flush_l2_cache", OPT_BOOL, &gpgpu_flush_l2_cache,
                   "Flush L2 cache at the end of each kernel call",
                   "0");

   option_parser_register(opp, "-gpgpu_deadlock_detect", OPT_BOOL, &gpu_deadlock_detect, 
                "Stop the simulation at deadlock (1=on (default), 0=off)", 
                "1");
   option_parser_register(opp, "-gpgpu_ptx_instruction_classification", OPT_INT32, 
               &gpgpu_ptx_instruction_classification, 
               "if enabled will classify ptx instruction types per kernel (Max 255 kernels now)", 
               "0");
   option_parser_register(opp, "-gpgpu_ptx_sim_mode", OPT_INT32, &g_ptx_sim_mode, 
               "Select between Performance (default) or Functional simulation (1)", 
               "0");
   option_parser_register(opp, "-gpgpu_clock_domains", OPT_CSTR, &gpgpu_clock_domains, 
                  "Clock Domain Frequencies in MhZ {<Core Clock>:<ICNT Clock>:<L2 Clock>:<DRAM Clock>}",
                  "500.0:2000.0:2000.0:2000.0");
   option_parser_register(opp, "-gpgpu_max_concurrent_kernel", OPT_INT32, &max_concurrent_kernel,
                          "maximum kernels that can run concurrently on GPU", "8" );
   option_parser_register(opp, "-gpgpu_cflog_interval", OPT_INT32, &gpgpu_cflog_interval, 
               "Interval between each snapshot in control flow logger", 
               "0");
   option_parser_register(opp, "-visualizer_enabled", OPT_BOOL,
                          &g_visualizer_enabled, "Turn on visualizer output (1=On, 0=Off)",
                          "1");
   option_parser_register(opp, "-visualizer_outputfile", OPT_CSTR, 
                          &g_visualizer_filename, "Specifies the output log file for visualizer",
                          NULL);
   option_parser_register(opp, "-visualizer_zlevel", OPT_INT32,
                          &g_visualizer_zlevel, "Compression level of the visualizer output log (0=no comp, 9=highest)",
                          "6");
    option_parser_register(opp, "-trace_enabled", OPT_BOOL, 
                          &Trace::enabled, "Turn on traces",
                          "0");
    option_parser_register(opp, "-trace_components", OPT_CSTR, 
                          &Trace::config_str, "comma seperated list of traces to enable. "
                          "Complete list found in trace_streams.tup. "
                          "Default none",
                          "none");
    option_parser_register(opp, "-trace_sampling_core", OPT_INT32, 
                          &Trace::sampling_core, "The core which is printed using CORE_DPRINTF. Default 0",
                          "0");
    option_parser_register(opp, "-trace_sampling_memory_partition", OPT_INT32, 
                          &Trace::sampling_memory_partition, "The memory partition which is printed using MEMPART_DPRINTF. Default -1 (i.e. all)",
                          "-1");
   ptx_file_line_stats_options(opp);
}

/////////////////////////////////////////////////////////////////////////////

void increment_x_then_y_then_z( dim3 &i, const dim3 &bound)
{
   i.x++;
   if ( i.x >= bound.x ) {
      i.x = 0;
      i.y++;
      if ( i.y >= bound.y ) {
         i.y = 0;
         if( i.z < bound.z ) 
            i.z++;
      }
   }
}

void gpgpu_sim::launch( kernel_info_t *kinfo )
{
   unsigned cta_size = kinfo->threads_per_cta();
   if ( cta_size > m_shader_config->n_thread_per_shader ) {
      printf("Execution error: Shader kernel CTA (block) size is too large for microarch config.\n");
      printf("                 CTA size (x*y*z) = %u, max supported = %u\n", cta_size, 
             m_shader_config->n_thread_per_shader );
      printf("                 => either change -gpgpu_shader argument in gpgpusim.config file or\n");
      printf("                 modify the CUDA source to decrease the kernel block size.\n");
      abort();
   }
   unsigned n=0;
   for(n=0; n < m_running_kernels.size(); n++ ) {
       if( (NULL==m_running_kernels[n]) || m_running_kernels[n]->done() ) {
           m_running_kernels[n] = kinfo;
           break;
       }
   }
   assert(n < m_running_kernels.size());
}

bool gpgpu_sim::can_start_kernel()
{
   for(unsigned n=0; n < m_running_kernels.size(); n++ ) {
       if( (NULL==m_running_kernels[n]) || m_running_kernels[n]->done() ) 
           return true;
   }
   return false;
}

bool gpgpu_sim::get_more_cta_left() const
{ 
   if (m_config.gpu_max_cta_opt != 0) {
      if( m_total_cta_launched >= m_config.gpu_max_cta_opt )
          return false;
   }
   for(unsigned n=0; n < m_running_kernels.size(); n++ ) {
       if( m_running_kernels[n] && !m_running_kernels[n]->no_more_ctas_to_run() ) 
           return true;
   }
   return false;
}

kernel_info_t *gpgpu_sim::select_kernel()
{
    for(unsigned n=0; n < m_running_kernels.size(); n++ ) {
        unsigned idx = (n+m_last_issued_kernel+1)%m_config.max_concurrent_kernel;
        if( m_running_kernels[idx] && !m_running_kernels[idx]->no_more_ctas_to_run() ) {
            m_last_issued_kernel=idx;
            // record this kernel for stat print if it is the first time this kernel is selected for execution  
            unsigned launch_uid = m_running_kernels[idx]->get_uid(); 
            if (std::find(m_executed_kernel_uids.begin(), m_executed_kernel_uids.end(), launch_uid) == m_executed_kernel_uids.end()) {
               m_executed_kernel_uids.push_back(launch_uid); 
               m_executed_kernel_names.push_back(m_running_kernels[idx]->name()); 
            }

            return m_running_kernels[idx];
        }
    }
    return NULL;
}

unsigned gpgpu_sim::finished_kernel()
{
    if( m_finished_kernel.empty() ) 
        return 0;
    unsigned result = m_finished_kernel.front();
    m_finished_kernel.pop_front();
    if (result) epoch_number++;
    return result;
}

void gpgpu_sim::set_kernel_done( kernel_info_t *kernel ) 
{ 
    unsigned uid = kernel->get_uid();
    m_finished_kernel.push_back(uid);
    std::vector<kernel_info_t*>::iterator k;
    for( k=m_running_kernels.begin(); k!=m_running_kernels.end(); k++ ) {
        if( *k == kernel ) {
            *k = NULL;
            break;
        }
    }
    assert( k != m_running_kernels.end() ); 
}

void set_ptx_warp_size(const struct core_config * warp_size);

gpgpu_sim::gpgpu_sim( const gpgpu_sim_config &config ) 
    : gpgpu_t(config), m_config(config)
{ 
    m_shader_config = &m_config.m_shader_config;
    m_memory_config = &m_config.m_memory_config;
    set_ptx_warp_size(m_shader_config);
    ptx_file_line_stats_create_exposed_latency_tracker(m_config.num_shader());

#ifdef GPGPUSIM_POWER_MODEL
        m_gpgpusim_wrapper = new gpgpu_sim_wrapper(config.g_power_simulation_enabled,config.g_power_config_name);
#endif

    m_shader_stats = new shader_core_stats(m_shader_config);
    //Handle different memory types
    m_memory_stats = new memory_stats_t*[m_memory_config->m_n_mem_types];
//    m_power_stats = new power_stat_t*[m_memory_config->m_n_mem];
    for (unsigned i=0;i<m_memory_config->m_n_mem_types;i++) {
//        unsigned index = (m_memory_config->m_n_mem_types == 1) ? ((m_memory_config->m_n_mem_t1) ? i : 1) : i;
        const memory_config* memory_config_type = &(m_memory_config->memory_config_array[i]);
        m_memory_stats[i] = new memory_stats_t(m_config.num_shader(),m_shader_config,memory_config_type);
//      m_power_stats[i] = new power_stat_t(m_shader_config,average_pipeline_duty_cycle,active_sms,m_shader_stats,memory_config_type,m_memory_stats[i]);

    }
    //TODO
    m_power_stats = new power_stat_t(m_shader_config,average_pipeline_duty_cycle,active_sms,m_shader_stats,&(m_memory_config->memory_config_array[0]),m_memory_stats[0]);

    average_pipeline_duty_cycle = (float *)malloc(sizeof(float));
    active_sms=(float *)malloc(sizeof(float));
//    m_power_stats = new power_stat_t(m_shader_config,average_pipeline_duty_cycle,active_sms,m_shader_stats,m_memory_config,m_memory_stats);

    gpu_sim_insn = 0;
    gpu_tot_sim_insn = 0;
    gpu_tot_issued_cta = 0;
    gpu_deadlock = false;


    m_cluster = new simt_core_cluster*[m_shader_config->n_simt_clusters];
    for (unsigned i=0;i<m_shader_config->n_simt_clusters;i++)  {
//        m_cluster[i] = new simt_core_cluster(this,i,m_shader_config,m_memory_config,m_shader_stats,m_memory_stats[0]);
        m_cluster[i] = new simt_core_cluster(this,i,m_shader_config,&m_memory_config->memory_config_array[0],m_shader_stats,m_memory_stats);
    }

    //TODO: for now, assume all memories have same partition parameters except
    //dram timing
    m_memory_partition_unit = new memory_partition_unit*[m_memory_config->m_n_mem];
    unsigned t1 = m_memory_config->memory_config_array[0].m_n_mem_sub_partition;
    unsigned t2 = m_memory_config->memory_config_array[1].m_n_mem_sub_partition;
    unsigned t = t1+t2;
    m_memory_sub_partition = new memory_sub_partition*[t];
//    m_memory_sub_partition = new memory_sub_partition*[&m_memory_config->memory_config_array[0].m_n_mem_sub_partition];
    for (unsigned i=0;i<m_memory_config->m_n_mem;i++) {
        //sdduming only 2 types of memory for now
        unsigned type = 0;
        if (i >= m_memory_config->memory_config_array[0].m_n_mem) {
            type = 1;
        }
        const memory_config* memory_config_type = &(m_memory_config->memory_config_array[type]);
        m_memory_partition_unit[i] = new memory_partition_unit(i, memory_config_type, m_memory_stats[type], &epoch_number);
//        m_memory_partition_unit[i] = new memory_partition_unit(i, const_cast<memory_config*>(&memory_config_type), m_memory_stats, &epoch_number);
        unsigned t3 = m_memory_config->memory_config_array[0].m_n_sub_partition_per_memory_channel;
//        for (unsigned p = 0; p < &m_memory_config->memory_config_array[0].m_n_sub_partition_per_memory_channel; p++) {
        for (unsigned p = 0; p < t3; p++) {
            unsigned submpid = i * t3 + p; 
//            unsigned submpid = i * m_memory_config->m_n_sub_partition_per_memory_channel + p; 
            m_memory_sub_partition[submpid] = m_memory_partition_unit[i]->get_sub_partition(p); 
        }
    }

    /*
     * Migration unit
     */
    migration_unit = new migrate(&(m_memory_config->memory_config_array[0]), &(m_memory_config->memory_config_array[1]), m_memory_partition_unit);

    icnt_wrapper_init();
    icnt_create(m_shader_config->n_simt_clusters, t);
//    icnt_create(m_shader_config->n_simt_clusters,m_memory_config->m_n_mem_sub_partition);

    time_vector_create(NUM_MEM_REQ_STAT);
    fprintf(stdout, "GPGPU-Sim uArch: performance model initialization complete.\n");

    m_running_kernels.resize( config.max_concurrent_kernel, NULL );
    m_last_issued_kernel = 0;
    m_last_cluster_issue = 0;
    *average_pipeline_duty_cycle=0;
    *active_sms=0;

    last_liveness_message_time = 0;

    epoch_number = 0;
}

int gpgpu_sim::shared_mem_size() const
{
   return m_shader_config->gpgpu_shmem_size;
}

int gpgpu_sim::num_registers_per_core() const
{
   return m_shader_config->gpgpu_shader_registers;
}

int gpgpu_sim::wrp_size() const
{
   return m_shader_config->warp_size;
}

int gpgpu_sim::shader_clock() const
{
   return m_config.core_freq/1000;
}

void gpgpu_sim::set_prop( cudaDeviceProp *prop )
{
   m_cuda_properties = prop;
}

const struct cudaDeviceProp *gpgpu_sim::get_prop() const
{
   return m_cuda_properties;
}

enum divergence_support_t gpgpu_sim::simd_model() const
{
   return m_shader_config->model;
}

void gpgpu_sim_config::init_clock_domains(void ) 
{
   sscanf(gpgpu_clock_domains,"%lf:%lf:%lf:%lf:%lf", 
          &core_freq, &icnt_freq, &l2_freq, &dram_freq, &dram_freq_t2);
   core_freq = core_freq MhZ;
   icnt_freq = icnt_freq MhZ;
   l2_freq = l2_freq MhZ;
   dram_freq = dram_freq MhZ;        
   dram_freq_t2 = dram_freq_t2 MhZ;        
   core_period = 1/core_freq;
   icnt_period = 1/icnt_freq;
   dram_period = 1/dram_freq;
   dram_period_t2 = 1/dram_freq_t2;
   l2_period = 1/l2_freq;
//   printf("GPGPU-Sim uArch: clock freqs: %lf:%lf:%lf:%lf\n",core_freq,icnt_freq,l2_freq,dram_freq);
//   printf("GPGPU-Sim uArch: clock periods: %.20lf:%.20lf:%.20lf:%.20lf\n",core_period,icnt_period,l2_period,dram_period);
   printf("GPGPU-Sim uArch: clock freqs: %lf:%lf:%lf:%lf:%lf\n",core_freq,icnt_freq,l2_freq,dram_freq,dram_freq_t2);
   printf("GPGPU-Sim uArch: clock periods: %.20lf:%.20lf:%.20lf:%.20lf:%.20lf\n",core_period,icnt_period,l2_period,dram_period,dram_period_t2);
}

void gpgpu_sim::reinit_clock_domains(void)
{
   core_time = 0;
   dram_time = 0;
   dram_time_t2 = 0;
   icnt_time = 0;
   l2_time = 0;
}

bool gpgpu_sim::active()
{
    if (m_config.gpu_max_cycle_opt && (gpu_tot_sim_cycle + gpu_sim_cycle) >= m_config.gpu_max_cycle_opt) 
       return false;
    if (m_config.gpu_max_insn_opt && (gpu_tot_sim_insn + gpu_sim_insn) >= m_config.gpu_max_insn_opt) 
       return false;
    if (m_config.gpu_max_cta_opt && (gpu_tot_issued_cta >= m_config.gpu_max_cta_opt) )
       return false;
    if (m_config.gpu_deadlock_detect && gpu_deadlock) 
       return false;
    for (unsigned i=0;i<m_shader_config->n_simt_clusters;i++) 
       if( m_cluster[i]->get_not_completed()>0 ) 
           return true;;
    for (unsigned i=0;i<m_memory_config->m_n_mem;i++) 
       if( m_memory_partition_unit[i]->busy()>0 )
           return true;;
    if( icnt_busy() )
        return true;
    if( get_more_cta_left() )
        return true;
    return false;
}

void gpgpu_sim::init()
{
    // run a CUDA grid on the GPU microarchitecture simulator
    gpu_sim_cycle = 0;
    gpu_sim_insn = 0;
    last_gpu_sim_insn = 0;
    m_total_cta_launched=0;

    reinit_clock_domains();
    set_param_gpgpu_num_shaders(m_config.num_shader());
    for (unsigned i=0;i<m_shader_config->n_simt_clusters;i++) 
       m_cluster[i]->reinit();
    m_shader_stats->new_grid();
    // initialize the control-flow, memory access, memory latency logger
    if (m_config.g_visualizer_enabled) {
        create_thread_CFlogger( m_config.num_shader(), m_shader_config->n_thread_per_shader, 0, m_config.gpgpu_cflog_interval );
    }
    shader_CTA_count_create( m_config.num_shader(), m_config.gpgpu_cflog_interval);
    if (m_config.gpgpu_cflog_interval != 0) {
       insn_warp_occ_create( m_config.num_shader(), m_shader_config->warp_size );
       shader_warp_occ_create( m_config.num_shader(), m_shader_config->warp_size, m_config.gpgpu_cflog_interval);
       shader_mem_acc_create( m_config.num_shader(), m_memory_config->m_n_mem, 4, m_config.gpgpu_cflog_interval);
       shader_mem_lat_create( m_config.num_shader(), m_config.gpgpu_cflog_interval);
       shader_cache_access_create( m_config.num_shader(), 3, m_config.gpgpu_cflog_interval);
       set_spill_interval (m_config.gpgpu_cflog_interval * 40);
    }

    if (g_network_mode)
       icnt_init();

    // McPAT initialization function. Called on first launch of GPU
#ifdef GPGPUSIM_POWER_MODEL
    if(m_config.g_power_simulation_enabled){
        init_mcpat(m_config, m_gpgpusim_wrapper, m_config.gpu_stat_sample_freq,  gpu_tot_sim_insn, gpu_sim_insn);
    }
#endif
}

void gpgpu_sim::update_stats() {
    for (unsigned i=0; i<m_memory_config->m_n_mem_types; i++)
        m_memory_stats[i]->memlatstat_lat_pw();
    gpu_tot_sim_cycle += gpu_sim_cycle;
    gpu_tot_sim_insn += gpu_sim_insn;
}

void gpgpu_sim::print_stats()
{
    ptx_file_line_stats_write_file();
    gpu_print_stat();

    if (g_network_mode) {
        printf("----------------------------Interconnect-DETAILS--------------------------------\n" );
        icnt_display_stats();
        icnt_display_overall_stats();
        printf("----------------------------END-of-Interconnect-DETAILS-------------------------\n" );
    }
}

void gpgpu_sim::deadlock_check()
{
   if (m_config.gpu_deadlock_detect && gpu_deadlock) {
      fflush(stdout);
      printf("\n\nGPGPU-Sim uArch: ERROR ** deadlock detected: last writeback core %u @ gpu_sim_cycle %u (+ gpu_tot_sim_cycle %u) (%u cycles ago)\n", 
             gpu_sim_insn_last_update_sid,
             (unsigned) gpu_sim_insn_last_update, (unsigned) (gpu_tot_sim_cycle-gpu_sim_cycle),
             (unsigned) (gpu_sim_cycle - gpu_sim_insn_last_update )); 
      unsigned num_cores=0;
      for (unsigned i=0;i<m_shader_config->n_simt_clusters;i++) {
         unsigned not_completed = m_cluster[i]->get_not_completed();
         if( not_completed ) {
             if ( !num_cores )  {
                 printf("GPGPU-Sim uArch: DEADLOCK  shader cores no longer committing instructions [core(# threads)]:\n" );
                 printf("GPGPU-Sim uArch: DEADLOCK  ");
                 m_cluster[i]->print_not_completed(stdout);
             } else if (num_cores < 8 ) {
                 m_cluster[i]->print_not_completed(stdout);
             } else if (num_cores >= 8 ) {
                 printf(" + others ... ");
             }
             num_cores+=m_shader_config->n_simt_cores_per_cluster;
         }
      }
      printf("\n");
      for (unsigned i=0;i<m_memory_config->m_n_mem;i++) {
         bool busy = m_memory_partition_unit[i]->busy();
         if( busy ) 
             printf("GPGPU-Sim uArch DEADLOCK:  memory partition %u busy\n", i );
      }
      if( icnt_busy() ) {
         printf("GPGPU-Sim uArch DEADLOCK:  iterconnect contains traffic\n");
         icnt_display_state( stdout );
      }
      printf("\nRe-run the simulator in gdb and use debug routines in .gdbinit to debug this\n");
      fflush(stdout);
      //abort();
   }
}

/// printing the names and uids of a set of executed kernels (usually there is only one)
std::string gpgpu_sim::executed_kernel_info_string() 
{
   std::stringstream statout; 

   statout << "kernel_name = "; 
   for (unsigned int k = 0; k < m_executed_kernel_names.size(); k++) {
      statout << m_executed_kernel_names[k] << " "; 
   }
   statout << std::endl; 
   statout << "kernel_launch_uid = ";
   for (unsigned int k = 0; k < m_executed_kernel_uids.size(); k++) {
      statout << m_executed_kernel_uids[k] << " "; 
   }
   statout << std::endl; 

   return statout.str(); 
}
void gpgpu_sim::set_cache_config(std::string kernel_name,  FuncCache cacheConfig )
{
	m_special_cache_config[kernel_name]=cacheConfig ;
}

FuncCache gpgpu_sim::get_cache_config(std::string kernel_name)
{
	for (	std::map<std::string, FuncCache>::iterator iter = m_special_cache_config.begin(); iter != m_special_cache_config.end(); iter++){
		    std::string kernel= iter->first;
			if (kernel_name.compare(kernel) == 0){
				return iter->second;
			}
	}
	return (FuncCache)0;
}

bool gpgpu_sim::has_special_cache_config(std::string kernel_name)
{
	for (	std::map<std::string, FuncCache>::iterator iter = m_special_cache_config.begin(); iter != m_special_cache_config.end(); iter++){
	    	std::string kernel= iter->first;
			if (kernel_name.compare(kernel) == 0){
				return true;
			}
	}
	return false;
}


void gpgpu_sim::set_cache_config(std::string kernel_name)
{
	if(has_special_cache_config(kernel_name)){
		change_cache_config(get_cache_config(kernel_name));
	}else{
		change_cache_config(FuncCachePreferNone);
	}
}


void gpgpu_sim::change_cache_config(FuncCache cache_config)
{
	if(cache_config != m_shader_config->m_L1D_config.get_cache_status()){
		printf("FLUSH L1 Cache at configuration change between kernels\n");
		for (unsigned i=0;i<m_shader_config->n_simt_clusters;i++) {
			m_cluster[i]->cache_flush();
	    }
	}

	switch(cache_config){
	case FuncCachePreferNone:
		m_shader_config->m_L1D_config.init(m_shader_config->m_L1D_config.m_config_string, FuncCachePreferNone);
		m_shader_config->gpgpu_shmem_size=m_shader_config->gpgpu_shmem_sizeDefault;
		break;
	case FuncCachePreferL1:
		if((m_shader_config->m_L1D_config.m_config_stringPrefL1 == NULL) || (m_shader_config->gpgpu_shmem_sizePrefL1 == (unsigned)-1))
		{
			printf("WARNING: missing Preferred L1 configuration\n");
			m_shader_config->m_L1D_config.init(m_shader_config->m_L1D_config.m_config_string, FuncCachePreferNone);
			m_shader_config->gpgpu_shmem_size=m_shader_config->gpgpu_shmem_sizeDefault;

		}else{
			m_shader_config->m_L1D_config.init(m_shader_config->m_L1D_config.m_config_stringPrefL1, FuncCachePreferL1);
			m_shader_config->gpgpu_shmem_size=m_shader_config->gpgpu_shmem_sizePrefL1;
		}
		break;
	case FuncCachePreferShared:
		if((m_shader_config->m_L1D_config.m_config_stringPrefShared == NULL) || (m_shader_config->gpgpu_shmem_sizePrefShared == (unsigned)-1))
		{
			printf("WARNING: missing Preferred L1 configuration\n");
			m_shader_config->m_L1D_config.init(m_shader_config->m_L1D_config.m_config_string, FuncCachePreferNone);
			m_shader_config->gpgpu_shmem_size=m_shader_config->gpgpu_shmem_sizeDefault;
		}else{
			m_shader_config->m_L1D_config.init(m_shader_config->m_L1D_config.m_config_stringPrefShared, FuncCachePreferShared);
			m_shader_config->gpgpu_shmem_size=m_shader_config->gpgpu_shmem_sizePrefShared;
		}
		break;
	default:
		break;
	}
}


void gpgpu_sim::clear_executed_kernel_info()
{
   m_executed_kernel_names.clear();
   m_executed_kernel_uids.clear();
}
void gpgpu_sim::gpu_print_stat() 
{  
   FILE *statfout = stdout; 

   std::string kernel_info_str = executed_kernel_info_string(); 
   fprintf(statfout, "%s", kernel_info_str.c_str()); 

   printf("gpu_sim_cycle = %lld\n", gpu_sim_cycle);
   printf("gpu_sim_insn = %lld\n", gpu_sim_insn);
   printf("gpu_ipc = %12.4f\n", (float)gpu_sim_insn / gpu_sim_cycle);
   printf("gpu_tot_sim_cycle = %lld\n", gpu_tot_sim_cycle+gpu_sim_cycle);
   printf("gpu_tot_sim_insn = %lld\n", gpu_tot_sim_insn+gpu_sim_insn);
   printf("gpu_tot_ipc = %12.4f\n", (float)(gpu_tot_sim_insn+gpu_sim_insn) / (gpu_tot_sim_cycle+gpu_sim_cycle));
   printf("gpu_tot_issued_cta = %lld\n", gpu_tot_issued_cta);



   // performance counter for stalls due to congestion.
   printf("gpu_stall_dramfull = %d\n", gpu_stall_dramfull);
   printf("gpu_stall_icnt2sh    = %d\n", gpu_stall_icnt2sh );

   time_t curr_time;
   time(&curr_time);
   unsigned long long elapsed_time = MAX( curr_time - g_simulation_starttime, 1 );
   printf( "gpu_total_sim_rate=%u\n", (unsigned)( ( gpu_tot_sim_insn + gpu_sim_insn ) / elapsed_time ) );

   //shader_print_l1_miss_stat( stdout );
   shader_print_cache_stats(stdout);

   cache_stats core_cache_stats;
   core_cache_stats.clear();
   for(unsigned i=0; i<m_config.num_cluster(); i++){
       m_cluster[i]->get_cache_stats(core_cache_stats);
   }
   printf("\nTotal_core_cache_stats:\n");
   core_cache_stats.print_stats(stdout, "Total_core_cache_stats_breakdown");
   shader_print_scheduler_stat( stdout, false );

   m_shader_stats->print(stdout);
#ifdef GPGPUSIM_POWER_MODEL
   if(m_config.g_power_simulation_enabled){
	   m_gpgpusim_wrapper->print_power_kernel_stats(gpu_sim_cycle, gpu_tot_sim_cycle, gpu_tot_sim_insn + gpu_sim_insn, kernel_info_str, true );
	   mcpat_reset_perf_count(m_gpgpusim_wrapper);
   }
#endif

   // performance counter that are not local to one shader
   unsigned t[2];
   t[0] = m_memory_config->memory_config_array[0].nbk;
   t[1] = m_memory_config->memory_config_array[1].nbk;
    for (unsigned i=0; i<m_memory_config->m_n_mem_types; i++)
        m_memory_stats[i]->memlatstat_print(m_memory_config->memory_config_array[i].m_n_mem,t[i]);
//   m_memory_stats->memlatstat_print(m_memory_config->m_n_mem,m_memory_config->nbk);
   for (unsigned i=0;i<m_memory_config->m_n_mem;i++)
      m_memory_partition_unit[i]->print(stdout);

   // L2 cache stats
   // Partition type 1:
    for (unsigned j=0; j<m_memory_config->m_n_mem_types; j++) {
        if(!m_memory_config->memory_config_array[j].m_L2_config.disabled()) {
            cache_stats l2_stats;
            struct cache_sub_stats l2_css;
            struct cache_sub_stats total_l2_css;
            l2_stats.clear();
            l2_css.clear();
            total_l2_css.clear();

            printf("\n========= L2 cache stats =========\n");
            for (unsigned i=0;i<m_memory_config->memory_config_array[j].m_n_mem_sub_partition;i++){
                m_memory_sub_partition[i]->accumulate_L2cache_stats(l2_stats);
                m_memory_sub_partition[i]->get_L2cache_sub_stats(l2_css);

                fprintf( stdout, "L2_cache_bank[%d]: Access = %u, Miss = %u, Miss_rate = %.3lf, Pending_hits = %u, Reservation_fails = %u\n",
                         i, l2_css.accesses, l2_css.misses, (double)l2_css.misses / (double)l2_css.accesses, l2_css.pending_hits, l2_css.res_fails);

                total_l2_css += l2_css;
            }
            if (!m_memory_config->memory_config_array[j].m_L2_config.disabled() && m_memory_config->memory_config_array[j].m_L2_config.get_num_lines()) {
               //L2c_print_cache_stat();
               printf("L2_total_cache_accesses = %u\n", total_l2_css.accesses);
               printf("L2_total_cache_misses = %u\n", total_l2_css.misses);
               if(total_l2_css.accesses > 0)
                   printf("L2_total_cache_miss_rate = %.4lf\n", (double)total_l2_css.misses/(double)total_l2_css.accesses);
               printf("L2_total_cache_pending_hits = %u\n", total_l2_css.pending_hits);
               printf("L2_total_cache_reservation_fails = %u\n", total_l2_css.res_fails);
               printf("L2_total_cache_breakdown:\n");
               l2_stats.print_stats(stdout, "L2_cache_stats_breakdown");
               total_l2_css.print_port_stats(stdout, "L2_cache");
            }
        }
    }

    // Migration stats
    printf("Migration stats start\n");
    printf("Addr->time\n");
    printMigrationFinishedQueue();
    printf("Migration stats end\n");

    printf("Number of stalls because of page locking: %llu\n", pageBlockingStall);
    printf("Total page count: %llu\n", globalPageCount.size());

    printMigrationQueue();

    printAccessDistribution();

    printf("\nNumber of touches per epoch\n");

    std::map<unsigned long long, std::map<unsigned, unsigned> >::iterator it_pageCount = globalPageCount.begin();
    for (; it_pageCount != globalPageCount.end(); it_pageCount++) {
        printf("%llu ", it_pageCount->first);
        std::map<unsigned, unsigned>::iterator it_second = it_pageCount->second.begin();
        for (; it_second != it_pageCount->second.end(); it_second++)
            printf("%u:%u ", it_second->first, it_second->second);
        printf("\n");
    }



   if (m_config.gpgpu_cflog_interval != 0) {
      spill_log_to_file (stdout, 1, gpu_sim_cycle);
      insn_warp_occ_print(stdout);
   }
   if ( gpgpu_ptx_instruction_classification ) {
      StatDisp( g_inst_classification_stat[g_ptx_kernel_count]);
      StatDisp( g_inst_op_classification_stat[g_ptx_kernel_count]);
   }

#ifdef GPGPUSIM_POWER_MODEL
   if(m_config.g_power_simulation_enabled){
       m_gpgpusim_wrapper->detect_print_steady_state(1,gpu_tot_sim_insn+gpu_sim_insn);
   }
#endif


   // Interconnect power stat print
   long total_simt_to_mem=0;
   long total_mem_to_simt=0;
   long temp_stm=0;
   long temp_mts = 0;
   for(unsigned i=0; i<m_config.num_cluster(); i++){
	   m_cluster[i]->get_icnt_stats(temp_stm, temp_mts);
	   total_simt_to_mem += temp_stm;
	   total_mem_to_simt += temp_mts;
   }
   printf("\nicnt_total_pkts_mem_to_simt=%ld\n", total_mem_to_simt);
   printf("icnt_total_pkts_simt_to_mem=%ld\n", total_simt_to_mem);

   time_vector_print();
   fflush(stdout);

   clear_executed_kernel_info(); 
}


// performance counter that are not local to one shader
unsigned gpgpu_sim::threads_per_core() const 
{ 
   return m_shader_config->n_thread_per_shader; 
}

void shader_core_ctx::mem_instruction_stats(const warp_inst_t &inst)
{
    unsigned active_count = inst.active_count(); 
    //this breaks some encapsulation: the is_[space] functions, if you change those, change this.
    switch (inst.space.get_type()) {
    case undefined_space:
    case reg_space:
        break;
    case shared_space:
        m_stats->gpgpu_n_shmem_insn += active_count; 
        break;
    case const_space:
        m_stats->gpgpu_n_const_insn += active_count;
        break;
    case param_space_kernel:
    case param_space_local:
        m_stats->gpgpu_n_param_insn += active_count;
        break;
    case tex_space:
        m_stats->gpgpu_n_tex_insn += active_count;
        break;
    case global_space:
    case local_space:
        if( inst.is_store() )
            m_stats->gpgpu_n_store_insn += active_count;
        else 
            m_stats->gpgpu_n_load_insn += active_count;
        break;
    default:
        abort();
    }
}


////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * Launches a cooperative thread array (CTA). 
 *  
 * @param kernel 
 *    object that tells us which kernel to ask for a CTA from 
 */

void shader_core_ctx::issue_block2core( kernel_info_t &kernel ) 
{
    set_max_cta(kernel);

    // find a free CTA context 
    unsigned free_cta_hw_id=(unsigned)-1;
    for (unsigned i=0;i<kernel_max_cta_per_shader;i++ ) {
      if( m_cta_status[i]==0 ) {
         free_cta_hw_id=i;
         break;
      }
    }
    assert( free_cta_hw_id!=(unsigned)-1 );

    // determine hardware threads and warps that will be used for this CTA
    int cta_size = kernel.threads_per_cta();

    // hw warp id = hw thread id mod warp size, so we need to find a range 
    // of hardware thread ids corresponding to an integral number of hardware
    // thread ids
    int padded_cta_size = cta_size; 
    if (cta_size%m_config->warp_size)
      padded_cta_size = ((cta_size/m_config->warp_size)+1)*(m_config->warp_size);
    unsigned start_thread = free_cta_hw_id * padded_cta_size;
    unsigned end_thread  = start_thread +  cta_size;

    // reset the microarchitecture state of the selected hardware thread and warp contexts
    reinit(start_thread, end_thread,false);
     
    // initalize scalar threads and determine which hardware warps they are allocated to
    // bind functional simulation state of threads to hardware resources (simulation) 
    warp_set_t warps;
    unsigned nthreads_in_block= 0;
    for (unsigned i = start_thread; i<end_thread; i++) {
        m_threadState[i].m_cta_id = free_cta_hw_id;
        unsigned warp_id = i/m_config->warp_size;
        nthreads_in_block += ptx_sim_init_thread(kernel,&m_thread[i],m_sid,i,cta_size-(i-start_thread),m_config->n_thread_per_shader,this,free_cta_hw_id,warp_id,m_cluster->get_gpu());
        m_threadState[i].m_active = true; 
        warps.set( warp_id );
    }
    assert( nthreads_in_block > 0 && nthreads_in_block <= m_config->n_thread_per_shader); // should be at least one, but less than max
    m_cta_status[free_cta_hw_id]=nthreads_in_block;

    // now that we know which warps are used in this CTA, we can allocate
    // resources for use in CTA-wide barrier operations
    m_barriers.allocate_barrier(free_cta_hw_id,warps);

    // initialize the SIMT stacks and fetch hardware
    init_warps( free_cta_hw_id, start_thread, end_thread);
    m_n_active_cta++;

    shader_CTA_count_log(m_sid, 1);
    printf("GPGPU-Sim uArch: core:%3d, cta:%2u initialized @(%lld,%lld)\n", m_sid, free_cta_hw_id, gpu_sim_cycle, gpu_tot_sim_cycle );
}

///////////////////////////////////////////////////////////////////////////////////////////

void dram_t::dram_log( int task ) 
{
   if (task == SAMPLELOG) {
      StatAddSample(mrqq_Dist, que_length());   
   } else if (task == DUMPLOG) {
      printf ("Queue Length DRAM[%d] ",id);StatDisp(mrqq_Dist);
   }
}

//Find next clock domain and increment its time
int gpgpu_sim::next_clock_domain(void) 
{
   double smallest = min4(core_time,icnt_time,dram_time,dram_time_t2);
   int mask = 0x00;
   if ( l2_time <= smallest ) {
      smallest = l2_time;
      mask |= L2 ;
      l2_time += m_config.l2_period;
   }
   if ( icnt_time <= smallest ) {
      mask |= ICNT;
      icnt_time += m_config.icnt_period;
   }
   if ( dram_time <= smallest ) {
      mask |= DRAM;
      dram_time += m_config.dram_period;
   }
   if ( dram_time_t2 <= smallest ) {
      mask |= DRAM_t2;
      dram_time_t2 += m_config.dram_period_t2;
   }
   if ( core_time <= smallest ) {
      mask |= CORE;
      core_time += m_config.core_period;
   }
   return mask;
}

void gpgpu_sim::issue_block2core()
{
    unsigned last_issued = m_last_cluster_issue; 
    for (unsigned i=0;i<m_shader_config->n_simt_clusters;i++) {
        unsigned idx = (i + last_issued + 1) % m_shader_config->n_simt_clusters;
        unsigned num = m_cluster[idx]->issue_block2core();
        if( num ) {
            m_last_cluster_issue=idx;
            m_total_cta_launched += num;
        }
    }
}

unsigned long long g_single_step=0; // set this in gdb to single step the pipeline

void increaseMigrationThreshold() {
    migrationThreshold += 16;
}

void decreaseMigrationThreshold() {
    if (migrationThreshold >= 16)
        migrationThreshold -= 16;
}

/* FDO for adjusting migration
 */
void gpgpu_sim::calculateMigrationThreshold() {
    if ((calculateBWRatio() > 65) && (calculateBWRatio() < 75)) {
        pauseMigration = true;
    } else if (calculateBWRatio() >= 75) {
        pauseMigration = true;
//        increaseMigrationThreshold();
    } else {
//        decreaseMigrationThreshold();
        pauseMigration = false;
    }
}

void gpgpu_sim::cycle()
{
    if (enableMigration 
            && !pauseMigration
            && !sendForMigrationPid.empty()) {
        for (auto &it_pid : sendForMigrationPid) {
            if (it_pid.second.empty()) 
                continue;
            unsigned long long page_addr_to_migrate = it_pid.second.front();
            if (migrationFinished[page_addr_to_migrate][1] == 0) 
            {
                // Timestamp at which front page is blocked until
                // migration is completed
                migrationFinished[page_addr_to_migrate][1] = gpu_sim_cycle + gpu_tot_sim_cycle;
            }
        }
    }

    if ((gpu_sim_cycle + gpu_tot_sim_cycle) / 100000ULL > last_updated_at) {
        last_updated_at++;
        printf("gpu_tot_ipc = %12.4f\n", (float)(gpu_tot_sim_insn+gpu_sim_insn) / (gpu_tot_sim_cycle+gpu_sim_cycle));

        if (limit_migration_rate)
            calculateMigrationThreshold();
        printf("BW-ratio: %u\n", calculateBWRatio());
    
        for (unsigned i=0;i<m_memory_config->m_n_mem;i++){
            m_memory_partition_unit[i]->get_dram()->incrementVectors(); 
        }
        
//        std::map<unsigned long long, std::vector<unsigned> >::iterator it_pageCount = globalPageCount.begin();
//        for (; it_pageCount != globalPageCount.end(); it_pageCount++) {
//            it_pageCount->second.push_back(0);
//        }
    }

   int clock_mask = next_clock_domain();

   if (clock_mask & CORE ) {
       // shader core loading (pop from ICNT into core) follows CORE clock
      for (unsigned i=0;i<m_shader_config->n_simt_clusters;i++) 
         m_cluster[i]->icnt_cycle(); 
   }
    unsigned tot_mem_sub_partitions=0; 
    for  (unsigned i=0; i<2; i++)
        tot_mem_sub_partitions += m_memory_config->memory_config_array[i].m_n_mem_sub_partition;
    if (clock_mask & ICNT) {
        // pop from memory controller to interconnect
//        for (unsigned i=0;i<m_memory_config->memory_config_array[0].m_n_mem_sub_partition;i++) {
        for (unsigned i=0;i<tot_mem_sub_partitions;i++) {
            mem_fetch* mf = m_memory_sub_partition[i]->top();
            if (mf) {
                unsigned response_size = mf->get_is_write()?mf->get_ctrl_size():mf->size();
                if ( ::icnt_has_buffer( m_shader_config->mem2device(i), response_size ) ) {
                    if (!mf->get_is_write()) 
                       mf->set_return_timestamp(gpu_sim_cycle+gpu_tot_sim_cycle);
                    mf->set_status(IN_ICNT_TO_SHADER,gpu_sim_cycle+gpu_tot_sim_cycle);
                    ::icnt_push( m_shader_config->mem2device(i), mf->get_tpc(), mf, response_size );
                    m_memory_sub_partition[i]->pop();
                } else {
                    gpu_stall_icnt2sh++;
                }
            } else {
               m_memory_sub_partition[i]->pop();
            }
        }
    }

    /*
     * Page monitoring unit
     */
    if (enableMigration && (clock_mask & L2)) {
        bool checkMigration = false;
        if (checkMigration) {
            //Migrate page once to check if the migration mechanism is working
            //Use a mapping table and determine a page in each memory and migrate it
            std::map<unsigned long long, unsigned>::iterator it = m_map_online.begin();
            unsigned found_pages_to_migrate = 0;
            bool found_page_HBM = false;
            bool found_page_SDDR = false;
            unsigned long long int addrHBM;
            unsigned long long int addrSDDR;

            for (; it != m_map_online.end() && (found_pages_to_migrate < 2); it++) {
                if (it->second == 1 && !found_page_HBM) {    //page in HBM
                    addrHBM = it->first;
                    found_pages_to_migrate++;
                    found_page_HBM = true;
                } else if (it->second == 0 && !found_page_SDDR) {
                    addrSDDR = it->first;
                    found_pages_to_migrate++;
                    found_page_SDDR = true;
                }
            }
            if (found_pages_to_migrate == 2) {
                migration_unit->migratePage(addrSDDR, addrHBM);
                checkMigration = false;
            }
        } else {
            // TODO: For now write the code here, later on call it as a
            // fuction: triggerMigrate()

            /* Policy1: Migrate after a threshold is met
             * Assume everything is located in DDR/CO memory. Migrate a page
             * from CO to BO memory when a page(4kB) is touched 128 in the
             * runtime. Assume for now a perfect table which can track the counter for
             * all the pages in the application. In this policy we are only
             * migrating pages from CO -> BO memory, not swapping.
             */
            /* Currently the table of counters is maintained in l2cache.cc,
             * it should be able to send a message to this piece of code to
             * perform migration. The key is to get access to pointer of
             * migration_unit. TODO: get the pointer reference in l2 class
             */
            /* addrHBM = NULL, since it is unidirectional migration
             * addrSDDR will be sent by the l2cache.cc unit
             * or maybe we can put it in a queue of things to migrate and
             * then migrate from here
             */
            /* Migration queue: migrationQueue is structure which will contain addresses to
             * migrate from CO memory to BO memory
             */

            for (auto &it_mig : sendForMigrationPid) {
                if (it_mig.second.empty())
                    continue;
                unsigned long long page_addr = it_mig.second.front();

                if (migrationQueue[page_addr] == 0) {
                    // If the page reaches "migrating" state then migrate it
                    if (readyForNextMigration[it_mig.first]) {
                        if (migrationWaitCycle[page_addr] >= migration_cost) {

                            // clear migration wait cycle
                            migrationWaitCycle[page_addr] = 0;

                            // migrate the page, send requests to DRAMs
                            migration_unit->migratePage(page_addr);

                            /* For magical migration
                            */
                            if (magical_migration) {
                                migrationQueue.erase(page_addr);
                                migrationWaitCycle.erase(page_addr);
                            
                                // Timestamp at which front page's migration is complete
                                migrationFinished[page_addr][2] = gpu_sim_cycle +
                                    gpu_tot_sim_cycle;
                                migrationFinished[page_addr][3] = gpu_sim_cycle
                                    + gpu_tot_sim_cycle;
//                                sendForMigrationPid[it_mig.first].remove(page_addr);
//                                readyForNextMigration[it_mig.first] = false;
                                unsigned partition = whichDDRPartition(page_addr,&(m_memory_config->memory_config_array[0]));
                                assert(partition < 4);
                                readyForNextMigration[partition] = true;
                                sendForMigrationPid[partition].remove(page_addr);
                            } else {
                           
                            
                            // Migrate one page in a cycle and try for others in the
                            // next cycle
                            readyForNextMigration[it_mig.first] = false;
                            
                            // set the migrationQueue state such that it cannot
                            // re-enter to be re-migrated
                            migrationQueue[page_addr] = (1<<43);
                            
                            // Timestamp at which front page's is ready to be
                            // migrated and read and write requests are now sent
                            // to the respective memory controllers
                            migrationFinished[page_addr][2] = gpu_sim_cycle +
                                gpu_tot_sim_cycle;
                            }

                        } else migrationWaitCycle[page_addr]++;
                    }
                }
            }
        }
    }

   if (clock_mask & DRAM) {
      for (unsigned i=0;i<m_memory_config->memory_config_array[0].m_n_mem;i++){
//      for (unsigned i=0;i<m_memory_config->m_n_mem;i++){
         m_memory_partition_unit[i]->dram_cycle(); // Issue the dram command (scheduler + delay model)
         // Update performance counters for DRAM
         m_memory_partition_unit[i]->set_dram_power_stats(m_power_stats->pwr_mem_stat->n_cmd[CURRENT_STAT_IDX][i], m_power_stats->pwr_mem_stat->n_activity[CURRENT_STAT_IDX][i],
         m_power_stats->pwr_mem_stat->n_nop[CURRENT_STAT_IDX][i], m_power_stats->pwr_mem_stat->n_act[CURRENT_STAT_IDX][i], m_power_stats->pwr_mem_stat->n_pre[CURRENT_STAT_IDX][i],
         m_power_stats->pwr_mem_stat->n_rd[CURRENT_STAT_IDX][i], m_power_stats->pwr_mem_stat->n_wr[CURRENT_STAT_IDX][i], m_power_stats->pwr_mem_stat->n_req[CURRENT_STAT_IDX][i]);
      }
   }

   if (clock_mask & DRAM_t2) {
       unsigned i = 0;
      for (unsigned j=0;j<m_memory_config->memory_config_array[1].m_n_mem;j++){
          i = j + m_memory_config->memory_config_array[0].m_n_mem;
         m_memory_partition_unit[i]->dram_cycle(); // Issue the dram command (scheduler + delay model)
//          Update performance counters for DRAM
         m_memory_partition_unit[i]->set_dram_power_stats(
                 m_power_stats->pwr_mem_stat->n_cmd[CURRENT_STAT_IDX][i],
                 m_power_stats->pwr_mem_stat->n_activity[CURRENT_STAT_IDX][i],
                 m_power_stats->pwr_mem_stat->n_nop[CURRENT_STAT_IDX][i],
                 m_power_stats->pwr_mem_stat->n_act[CURRENT_STAT_IDX][i],
                 m_power_stats->pwr_mem_stat->n_pre[CURRENT_STAT_IDX][i],
                 m_power_stats->pwr_mem_stat->n_rd[CURRENT_STAT_IDX][i],
                 m_power_stats->pwr_mem_stat->n_wr[CURRENT_STAT_IDX][i],
                 m_power_stats->pwr_mem_stat->n_req[CURRENT_STAT_IDX][i]);
      }
   }

   // L2 operations follow L2 clock domain
   if (clock_mask & L2) {
       m_power_stats->pwr_mem_stat->l2_cache_stats[CURRENT_STAT_IDX].clear();
//      for (unsigned i=0;i<m_memory_config->memory_config_array[0].m_n_mem_sub_partition;i++) {
      for (unsigned i=0;i<tot_mem_sub_partitions;i++) {
          //move memory request from interconnect into memory partition (if not backed up)
          //Note:This needs to be called in DRAM clock domain if there is no L2 cache in the system
          if ( m_memory_sub_partition[i]->full() ) {
             gpu_stall_dramfull++;
          } else {
              mem_fetch* mf = (mem_fetch*) icnt_pop( m_shader_config->mem2device(i) );
              m_memory_sub_partition[i]->push( mf, gpu_sim_cycle + gpu_tot_sim_cycle );
              if(mf != NULL) {
                if (i != mf->get_sub_partition_id())
                    printf("i: %d,mf_sub_part: %d\n", i, mf->get_sub_partition_id());
                assert(i == mf->get_sub_partition_id());
                if (mf->get_addr() == 2152209920)
                      printf("break me here \n");
              }
          }
          m_memory_sub_partition[i]->cache_cycle(gpu_sim_cycle+gpu_tot_sim_cycle);
          m_memory_sub_partition[i]->accumulate_L2cache_stats(m_power_stats->pwr_mem_stat->l2_cache_stats[CURRENT_STAT_IDX]);
       }
   }

   if (clock_mask & ICNT) {
      icnt_transfer();
   }

   if (clock_mask & CORE) {
      // L1 cache + shader core pipeline stages
      
      /* For migration purpose only
       * Assume there can be only 15 simt_clusters, but if there are less than
       * that then we need to reset bit of extra L1 caches of extra sim_clusters 
       */
      if (enableMigration) {
          std::map<unsigned long long, uint64_t>::iterator it = migrationQueue.begin();
          for (; it != migrationQueue.end(); ++it) {
              for (unsigned i=m_shader_config->n_simt_clusters; i<15 ;i++) {
                  if (it->second != 0 && it->second != (1<<43))
                  {
                      resetBit(it->second, i);
                  }
              }
          }
      }
      m_power_stats->pwr_mem_stat->core_cache_stats[CURRENT_STAT_IDX].clear();
      for (unsigned i=0;i<m_shader_config->n_simt_clusters;i++) {
         if (m_cluster[i]->get_not_completed() || get_more_cta_left() ) {
               m_cluster[i]->core_cycle();
               *active_sms+=m_cluster[i]->get_n_active_sms();
         } else {
            // if shader is empty then clear the migrating bit of L1 pending in
            // the migrationQueue data structure
             m_cluster[i]->flushOnMigration();
//             std::map<unsigned long long, uint64_t>::iterator it = migrationQueue.begin();
//             for (; it != migrationQueue.end(); ++it) {
//                 if (it->second != 0 && it->second != (1<<43))
//                 {
//                         resetBit(it->second, i);
//                 }
//             }
         }
         // Update core icnt/cache stats for GPUWattch
         m_cluster[i]->get_icnt_stats(m_power_stats->pwr_mem_stat->n_simt_to_mem[CURRENT_STAT_IDX][i], m_power_stats->pwr_mem_stat->n_mem_to_simt[CURRENT_STAT_IDX][i]);
         m_cluster[i]->get_cache_stats(m_power_stats->pwr_mem_stat->core_cache_stats[CURRENT_STAT_IDX]);
      }
      float temp=0;
      for (unsigned i=0;i<m_shader_config->num_shader();i++){
        temp+=m_shader_stats->m_pipeline_duty_cycle[i];
      }
      temp=temp/m_shader_config->num_shader();
      *average_pipeline_duty_cycle=((*average_pipeline_duty_cycle)+temp);
        //cout<<"Average pipeline duty cycle: "<<*average_pipeline_duty_cycle<<endl;


      if( g_single_step && ((gpu_sim_cycle+gpu_tot_sim_cycle) >= g_single_step) ) {
          asm("int $03");
      }
      gpu_sim_cycle++;
      if( g_interactive_debugger_enabled ) 
         gpgpu_debug();

      // McPAT main cycle (interface with McPAT)
#ifdef GPGPUSIM_POWER_MODEL
      if(m_config.g_power_simulation_enabled){
          mcpat_cycle(m_config, getShaderCoreConfig(), m_gpgpusim_wrapper, m_power_stats, m_config.gpu_stat_sample_freq, gpu_tot_sim_cycle, gpu_sim_cycle, gpu_tot_sim_insn, gpu_sim_insn);
      }
#endif

      issue_block2core();
      
      // Depending on configuration, flush the caches once all of threads are completed.
      int all_threads_complete = 1;
      if (m_config.gpgpu_flush_l1_cache) {
         for (unsigned i=0;i<m_shader_config->n_simt_clusters;i++) {
            if (m_cluster[i]->get_not_completed() == 0)
                m_cluster[i]->cache_flush();
            else
               all_threads_complete = 0 ;
         }
      }

      if(m_config.gpgpu_flush_l2_cache){
          if(!m_config.gpgpu_flush_l1_cache){
              for (unsigned i=0;i<m_shader_config->n_simt_clusters;i++) {
                  if (m_cluster[i]->get_not_completed() != 0){
                      all_threads_complete = 0 ;
                      break;
                  }
              }
          }

          //TODO: assume both emories either enable or disabled
         if (all_threads_complete && !m_memory_config->memory_config_array[0].m_L2_config.disabled() ) {
            printf("Flushed L2 caches...\n");
            if (m_memory_config->memory_config_array[0].m_L2_config.get_num_lines()) {
               int dlc = 0;
               for (unsigned i=0;i<m_memory_config->m_n_mem;i++) {
                  dlc = m_memory_sub_partition[i]->flushL2();
                  assert (dlc == 0); // need to model actual writes to DRAM here
                  printf("Dirty lines flushed from L2 %d is %d\n", i, dlc  );
               }
            }
         }
      }

      if (!(gpu_sim_cycle % m_config.gpu_stat_sample_freq)) {
         time_t days, hrs, minutes, sec;
         time_t curr_time;
         time(&curr_time);
         unsigned long long  elapsed_time = MAX(curr_time - g_simulation_starttime, 1);
         if ( (elapsed_time - last_liveness_message_time) >= m_config.liveness_message_freq ) {
            days    = elapsed_time/(3600*24);
            hrs     = elapsed_time/3600 - 24*days;
            minutes = elapsed_time/60 - 60*(hrs + 24*days);
            sec = elapsed_time - 60*(minutes + 60*(hrs + 24*days));
            printf("GPGPU-Sim uArch: cycles simulated: %lld  inst.: %lld (ipc=%4.1f) sim_rate=%u (inst/sec) elapsed = %u:%u:%02u:%02u / %s", 
                   gpu_tot_sim_cycle + gpu_sim_cycle, gpu_tot_sim_insn + gpu_sim_insn, 
                   (double)gpu_sim_insn/(double)gpu_sim_cycle,
                   (unsigned)((gpu_tot_sim_insn+gpu_sim_insn) / elapsed_time),
                   (unsigned)days,(unsigned)hrs,(unsigned)minutes,(unsigned)sec,
                   ctime(&curr_time));
            fflush(stdout);
            last_liveness_message_time = elapsed_time; 
            if (gpu_tot_sim_insn + gpu_sim_insn == 6477530)
                printf("break me here");
         }
         visualizer_printstat();
        for (unsigned i=0; i<m_memory_config->m_n_mem_types; i++)
            m_memory_stats[i]->memlatstat_lat_pw();
         if (m_config.gpgpu_runtime_stat && (m_config.gpu_runtime_stat_flag != 0) ) {
            if (m_config.gpu_runtime_stat_flag & GPU_RSTAT_BW_STAT) {
               for (unsigned i=0;i<m_memory_config->m_n_mem;i++) 
                  m_memory_partition_unit[i]->print_stat(stdout);
                for (unsigned i=0; i<m_memory_config->m_n_mem_types; i++) {
                    printf("maxmrqlatency = %d \n", m_memory_stats[i]->max_mrq_latency);
                    printf("maxmflatency = %d \n", m_memory_stats[i]->max_mf_latency);
                }
            }
            if (m_config.gpu_runtime_stat_flag & GPU_RSTAT_SHD_INFO) 
               shader_print_runtime_stat( stdout );
            if (m_config.gpu_runtime_stat_flag & GPU_RSTAT_L1MISS) {
               shader_print_l1_miss_stat( stdout );
            }
            if (m_config.gpu_runtime_stat_flag & GPU_RSTAT_SCHED) 
               shader_print_scheduler_stat( stdout, false );
         }
      }

      if (!(gpu_sim_cycle % 20000)) {
         // deadlock detection 
         if (m_config.gpu_deadlock_detect && gpu_sim_insn == last_gpu_sim_insn) {
            gpu_deadlock = true;
         } else {
            last_gpu_sim_insn = gpu_sim_insn;
         }
      }
      try_snap_shot(gpu_sim_cycle);
      spill_log_to_file (stdout, 0, gpu_sim_cycle);
   }
}


void shader_core_ctx::dump_warp_state( FILE *fout ) const
{
   fprintf(fout, "\n");
   fprintf(fout, "per warp functional simulation status:\n");
   for (unsigned w=0; w < m_config->max_warps_per_shader; w++ ) 
       m_warp[w].print(fout);
}

void gpgpu_sim::dump_pipeline( int mask, int s, int m ) const
{
/*
   You may want to use this function while running GPGPU-Sim in gdb.
   One way to do that is add the following to your .gdbinit file:
 
      define dp
         call g_the_gpu.dump_pipeline_impl((0x40|0x4|0x1),$arg0,0)
      end
 
   Then, typing "dp 3" will show the contents of the pipeline for shader core 3.
*/

   printf("Dumping pipeline state...\n");
   if(!mask) mask = 0xFFFFFFFF;
   for (unsigned i=0;i<m_shader_config->n_simt_clusters;i++) {
      if(s != -1) {
         i = s;
      }
      if(mask&1) m_cluster[m_shader_config->sid_to_cluster(i)]->display_pipeline(i,stdout,1,mask & 0x2E);
      if(s != -1) {
         break;
      }
   }
   if(mask&0x10000) {
      for (unsigned i=0;i<m_memory_config->m_n_mem;i++) {
         if(m != -1) {
            i=m;
         }
         printf("DRAM / memory controller %u:\n", i);
         if(mask&0x100000) m_memory_partition_unit[i]->print_stat(stdout);
         if(mask&0x1000000)   m_memory_partition_unit[i]->visualize();
         if(mask&0x10000000)   m_memory_partition_unit[i]->print(stdout);
         if(m != -1) {
            break;
         }
      }
   }
   fflush(stdout);
}

const struct shader_core_config * gpgpu_sim::getShaderCoreConfig()
{
   return m_shader_config;
}

const struct memory_config_types * gpgpu_sim::getMemoryConfig()
{
   return m_memory_config;
}

simt_core_cluster * gpgpu_sim::getSIMTCluster()
{
   return *m_cluster;
}

unsigned gpgpu_sim::canMigrate(unsigned long long addr, unsigned migrationState)
{
    unsigned state = migrationState;
    // Check all the L1 caches
    // Check all the L2 caches
    // Check in all the sub-partitions that no pages have request to the page to
   // be migrated.
    unsigned tot_mem_sub_partitions = 0;
    for (unsigned i=0; i<2; i++)
        tot_mem_sub_partitions += m_memory_config->memory_config_array[i].m_n_mem_sub_partition;
    for (unsigned i=0;i<tot_mem_sub_partitions;i++) {
        // TODO check the logic
//       state = migrationState && m_memory_sub_partition[i]->checkIfPresent(addr);
        ;
    }
    return state;
}

bool checkBit(uint64_t x, uint64_t pos) {
    return x & (1UL<<pos);
}

bool checkAllBitsBelow(uint64_t x, uint64_t pos) {
    return ((x & ((1UL << pos)-1)) == ((1UL << pos)-1));
}

bool checkAllBitsBelowReset(uint64_t x, uint64_t pos) {
    return ((x | ~((1UL << pos)-1)) == ~((1UL << pos)-1));
}

void setBit(uint64_t &x, uint64_t pos) {
    x |= (1UL << pos);
}

void resetBit(uint64_t &x, uint64_t pos) {
    x &= ~(1UL << pos);
}

void printMigrationFinishedQueue() {
    for (auto it_migration : migrationFinished) {
        printf("%llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu\n", it_migration.first, it_migration.second[0], it_migration.second[1], it_migration.second[2], it_migration.second[3], it_migration.second[4], it_migration.second[5], it_migration.second[6], it_migration.second[7], it_migration.second[8], it_migration.second[9]);
    }
}

void printMigrationQueue() {
    std::map<unsigned long long, uint64_t>::iterator it = migrationQueue.begin();
    for (; it != migrationQueue.end(); ++it) {
        printf("addr: %llu, state: 0x%lx\n", it->first, it->second);
    }
    printf("Migration done for addresses: \n");
    printMigrationFinishedQueue();
}

void printAccessDistribution() {
    printf("Access distribution before, when, after\n");
    for (auto it : accessDistribution) {
        printf("%llu %llu %llu %llu\n", it.first, it.second[0], it.second[1], it.second[2]);
    }
}

void printSendForMigration() {
    for (auto &it_pid : sendForMigrationPid) {
        if (it_pid.second.empty()) 
            continue;
        for (auto &it : it_pid.second) {
            printf("%u %llu 0x%lx\n", it_pid.first, it, migrationQueue[it]);
        }
    }
}

void printMap() {
    std::map<unsigned long long, unsigned>::iterator it = m_map_online.begin();
    for (; it != m_map_online.end(); ++it) {
        printf("addr: %llu, partition: %u\n", it->first, it->second);
    }
}

void printPartFromMap(unsigned long long addr) {
    if (m_map_online.count(addr)) {
        printf("addr: %llu, part: %u\n", addr, m_map_online[addr]);
    } else {
        printf("addr not found\n");
    }
}

void print_l1_wb() {
    std::map<unsigned, new_addr_type>::iterator it = l1_wb_map.begin();
    for (; it != l1_wb_map.end(); ++it) {
        printf("uid: %u, addr: %llu\n", it->first, it->second);
    }
}

void print_l2_wb() {
    std::map<unsigned, new_addr_type>::iterator it = l2_wb_map.begin();
    for (; it != l2_wb_map.end(); ++it) {
        printf("uid: %u, addr: %llu\n", it->first, it->second);
    }
}

void print_l1_wr_miss_no_wa() {
    std::map<unsigned, std::pair<new_addr_type, unsigned> >::iterator it =  l1_wr_miss_no_wa_map.begin();
    for (; it != l1_wr_miss_no_wa_map.end(); ++it) {
        printf("uid: %u, addr: %llu, core_id: %d\n", it->first, it->second.first, it->second.second);
    }
}

unsigned gpgpu_sim::calculateBWRatio() {
    unsigned ddr = 0;
    unsigned gddr = 0;
    for (unsigned i=0;i<m_memory_config->m_n_mem;i++) {
        if (i<4)
            ddr += m_memory_partition_unit[i]->getTotDramReq();
        else
            gddr += m_memory_partition_unit[i]->getTotDramReq();
    }

    printf("ddr: %u, gddr: %u\n", ddr, gddr);

    if (ddr == 0 && gddr == 0)
        return 0;
    if (ddr != 0)
        return ((float) gddr)/((float)(ddr+gddr))*100.0;
    else return 100;
}

unsigned whichDDRPartition(unsigned long long page_addr, const class memory_config *memConfig)
{
    mem_access_t accessSDDR(MEM_MIGRATE_R, page_addr, 128U, 0);
    const class memory_config* memConfigSDDR = &(memConfig->m_memory_config_types->memory_config_array[0]);
    mem_fetch *mfSDDR = new mem_fetch( accessSDDR, 
            READ_PACKET_SIZE, 
            memConfigSDDR, 0);
    unsigned global_spidSDDR = mfSDDR->get_tlx_addr().chip; 
    delete mfSDDR;
    return global_spidSDDR;
}
