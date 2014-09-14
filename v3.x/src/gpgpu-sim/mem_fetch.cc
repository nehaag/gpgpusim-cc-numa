// Copyright (c) 2009-2011, Tor M. Aamodt
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

#include "mem_fetch.h"
#include "mem_latency_stat.h"
#include "shader.h"
#include "visualizer.h"
#include "gpu-sim.h"
#include "addrdec.h"

uint64_t mem_fetch::sm_next_mf_request_uid=1;
uint64_t mem_fetch::deallocated_tot=0;
uint64_t mem_fetch::deallocated[NUM_MEM_ACCESS_TYPE] = {0,0,0,0,0,0,0,0,0,0,0,0,0};
uint64_t mem_fetch::allocated[NUM_MEM_ACCESS_TYPE] = {0,0,0,0,0,0,0,0,0,0,0,0,0};

mem_fetch::mem_fetch( mem_fetch *mf,
                      const mem_access_t &access)
{
   m_request_uid = sm_next_mf_request_uid++;
   m_access = access;
    if (m_access.get_type() < NUM_MEM_ACCESS_TYPE)
        allocated[access.get_type()]++;
//   m_inst = NULL;
   m_data_size = access.get_size();
   m_ctrl_size = mf->get_ctrl_size();
   m_sid = mf->get_sid();
   m_tpc = mf->get_tpc();
   m_wid = mf->get_wid();
   m_mem_config = mf->get_mem_config();
   m_raw_addr = mf->get_tlx_addr(); 
   m_partition_addr = m_mem_config->m_address_mapping.partition_address(access.get_addr());
   m_type = m_access.is_write()?WRITE_REQUEST:READ_REQUEST;
   m_timestamp = gpu_sim_cycle + gpu_tot_sim_cycle;
   m_timestamp2 = 0;
   m_status = MEM_FETCH_INITIALIZED;
   m_status_change = gpu_sim_cycle + gpu_tot_sim_cycle;
   icnt_flit_size = m_mem_config->icnt_flit_size;

   if (m_mem_config->type == 2) assert(m_raw_addr.sub_partition >= m_mem_config->m_memory_config_types->memory_config_array[0].m_n_mem_sub_partition);
   if (m_mem_config->type == 1) assert(m_raw_addr.sub_partition < m_mem_config->m_memory_config_types->memory_config_array[0].m_n_mem_sub_partition);

}

/* For migration unit packet generation
 */
mem_fetch::mem_fetch( const mem_access_t &access, unsigned ctrl_size, const class memory_config *config, unsigned type) : request_status_vector(28, 0)
{
    if (access.get_addr() == 2152209376) {
        printf("break here");
    }
   m_request_uid = sm_next_mf_request_uid++;
   m_access = access;
    if (m_access.get_type() < NUM_MEM_ACCESS_TYPE)
        allocated[access.get_type()]++;
   m_data_size = access.get_size();
   m_ctrl_size = get_ctrl_size();
   m_sid = 0; // TODO: fake id
   m_tpc = -1;
   m_wid = -1;
   m_mem_config = config;
   unsigned partition_offset = type * (config->m_memory_config_types->memory_config_array[0].m_n_mem_sub_partition);
   config->m_address_mapping.addrdec_tlx_hetero(access.get_addr(), &m_raw_addr, partition_offset);
   m_partition_addr = m_mem_config->m_address_mapping.partition_address(access.get_addr());
   m_type = m_access.is_write()?WRITE_REQUEST:READ_REQUEST;
   m_timestamp = gpu_sim_cycle + gpu_tot_sim_cycle;
   m_timestamp2 = 0;
   m_status = MEM_FETCH_INITIALIZED;
   m_status_change = gpu_sim_cycle + gpu_tot_sim_cycle;
   icnt_flit_size = m_mem_config->icnt_flit_size;

   if (m_mem_config->type == 2) assert(m_raw_addr.sub_partition >= m_mem_config->m_memory_config_types->memory_config_array[0].m_n_mem_sub_partition);
   if (m_mem_config->type == 1) assert(m_raw_addr.sub_partition < m_mem_config->m_memory_config_types->memory_config_array[0].m_n_mem_sub_partition);
}

mem_fetch::mem_fetch( const mem_access_t &access, 
                      const warp_inst_t *inst,
                      unsigned ctrl_size, 
                      unsigned wid,
                      unsigned sid, 
                      unsigned tpc, 
                      const class memory_config *config ) : request_status_vector(28, 0) 
{
   m_request_uid = sm_next_mf_request_uid++;
   m_access = access;
    if (m_access.get_type() < NUM_MEM_ACCESS_TYPE)
        allocated[access.get_type()]++;
   if( inst ) { 
       m_inst = *inst;
       assert( wid == m_inst.warp_id() );
   }
   m_data_size = access.get_size();
   m_ctrl_size = ctrl_size;
   m_sid = sid;
   m_tpc = tpc;
   m_wid = wid;

    if (access.get_addr() == 2152209376)
        printf("break here");

   const class memory_config* config_type = config;
   unsigned type = 0;
//   FOR 3-level address mapping
    unsigned long long addr_temp = access.get_addr();
    unsigned long long line_addr_temp = (access.get_addr() & (~4095UL));
    srand(line_addr_temp);
    unsigned int rand_num = (rand() % 100);
    //Lookup the address in the mem_map generated by the trace
    if (config->m_memory_config_types->enable_addr_limit > 0) {
        if (config->m_memory_config_types->enable_addr_limit == 1) {
            type = (m_map[line_addr_temp]-1);
            m_map_online[line_addr_temp] = type;
            if (type != 1 && type !=0)
                printf("addr: %lld, type: %d", line_addr_temp, type);
        } else if (config->m_memory_config_types->enable_addr_limit == 2) {
            if (m_map_online.count(line_addr_temp))
                type = m_map_online[line_addr_temp];
            else {
                //Perform a bandwidth equal, capacity equal linux standard mapping
                if (rand_num < config->m_memory_config_types->data_ratio)
                    type = 0;   //send to sddr
                else type = 1;  //send to hbm
                m_map_online[line_addr_temp] = type;
            }
        } else if (config->m_memory_config_types->enable_addr_limit == 3) {
            //Capacity limited random allocation in certain ratio
            //l: number of cacheline accesses so far put into hbm
            //lines: total number of cacheline for this workload
            if (m_map_online.count(line_addr_temp)) {
                // second touch to the address
                type = m_map_online[line_addr_temp];
                //trigger migration on second touch
                //migrate(addr1, addr2); //call this function in memory
                //controller to swap the pages starting at these addresses
            } else {
                if (num_lines_hbm < (config->m_memory_config_types->line_ratio/100.0*config->m_memory_config_types->cachelines)) {
                    if (rand_num < config->m_memory_config_types->data_ratio)
                        type = 0;
                    else {
                        type = 1;
                        num_lines_hbm += 1;
                    }
                } else {
                    type = 0;
                }
                m_map_online[line_addr_temp] = type;
            }
        } else if (config->m_memory_config_types->enable_addr_limit == 4) {
            //App annotation placement
            if (m_map_online.count(line_addr_temp))
                type = m_map_online[line_addr_temp];
            else {
                unsigned int tot_hbm_lines = config->m_memory_config_types->line_ratio/100.0*config->m_memory_config_types->cachelines;
                if (num_lines_hbm < tot_hbm_lines) {
                    //calculate for current address its cacheline number in
                    //linear address space and compare with how many can be fit
                    //in HBM, since this is a local first policy it is easier to
                    //do this
//                    /* specifically for xsbench*/
                    if (addr_temp < 2147483648) {
                        type = 1;
                        num_lines_hbm++;
                    } else {
                        unsigned long long cacheline_num = ((addr_temp - 2147483648) & (~127UL))/128;
//                        unsigned long long low_addr = ((addr_temp - 2187403776) & (~127UL))/128;
//                        unsigned long long high_addr = ((addr_temp - 2199701504) & (~127UL))/128;
                        if (cacheline_num <= tot_hbm_lines) {
//                        if (addr_temp >= low_addr && addr_temp <= high_addr) {
                            type = 1;
                            num_lines_hbm++;
                        }
                        else
                            type = 0;
                    }
                 } else {
                    type = 0;
                }
                m_map_online[line_addr_temp] = type;
            }
        }
        assert(type == 1 || type ==0);
        if (type == 0)
           config_type = &(config->m_memory_config_types->memory_config_array[0]);
        else
           config_type = &(config->m_memory_config_types->memory_config_array[1]);
    }

    
/*    
//    if ((access.get_addr() >= 2179483264 && (access.get_addr() < 2182916864)) && config->m_memory_config_types->enable_addr_limit) {
    if ((access.get_addr() >= 2180483840) && config->m_memory_config_types->enable_addr_limit) {
//    if ((access.get_addr() >= 2148532224 && (access.get_addr() < 2149580672)) && config->m_memory_config_types->enable_addr_limit) {
//    if (config->m_memory_config_types->enable_addr_limit) {
//    if (access.get_addr() < 0) {
        //BW-aware mapping for performance:
        //for say 4(DDR3)+8(HBM) = 12 channels in total, we will see 4 bits in
        //the address: bits 8,9,10,11. Based on the value we will reform the
        //address. If these 4 bits value lies from 0-7, then we will use config
        //2 (HBM), else if value is in between 8-11, then use config 1 (DDR3)
        //step1: extract bits 8-11 from the addr
        unsigned long long int address = access.get_addr();
        unsigned int parts = config_type->m_memory_config_types->m_n_mem;
        //assume 8th bit is where channel address starts, and maximum of 16
        //channels for now
        unsigned int channel = ((address & 0x0000000000000f00) >> 8) % 14; //20= 4 + 8*2
//        unsigned int channel = ((address & 0x0000000000000f00) >> 8) % parts;
        unsigned int channel_bits = 2;
        if (channel < config_type->m_memory_config_types->m_n_mem_t1) {
            //pack 2 bits of the address
            channel_bits = LOGB2_32(config_type->m_memory_config_types->m_n_mem_t1);
            type = 0;
            config_type = &(config->m_memory_config_types->memory_config_array[0]);
        } else {
            //pack 3 bit of the address
            channel_bits = LOGB2_32(config_type->m_memory_config_types->m_n_mem_t2);
            type = 1;
            config_type = &(config->m_memory_config_types->memory_config_array[1]);
        }
        //assuming 16 channels we have 4 bits originally in the address
//        unsigned long long int addr_temp_msb = (address & 0xffffffffffffe000) >> (5 - channel_bits);
        unsigned long long int addr_temp_msb = (address & 0xfffffffffffff000) >> (4 - channel_bits);
        unsigned long long int addr_temp_lsb = address & 0x00000000000000ff;
        //correct new address
        addr_temp = addr_temp_msb | ((channel % 8) << 8) | addr_temp_lsb;
//        addr_temp = addr_temp_msb | (channel << 8) | addr_temp_lsb;
//    } else if (access.get_addr() >= 2182916864 && config->m_memory_config_types->enable_addr_limit) {
//    } else if (access.get_addr() >= 2149310720 && config->m_memory_config_types->enable_addr_limit) {
    } else if (config->m_memory_config_types->enable_addr_limit) {
//    } else if ((access.get_addr() >= 2180483840) && config->m_memory_config_types->enable_addr_limit) {
       config_type = &(config->m_memory_config_types->memory_config_array[1]);
       type = 1;
    } else {
       config_type = &(config->m_memory_config_types->memory_config_array[0]);
       type = 0;
    }
*/
   //partition_offset is a global sub partition id, when a mem_fetch object
   //constructor is called for 1st time for an address, irrespective of its
   //actual config, it is assigned config type 1, hence offset needs to be
   //determied accordingly, however when for same object it is called 2nd time,
   //it has its actual config, therefore offset needs to be calculated based on
   //number of sub-partition in 1st type of memory. partition_offset should
   //remain always the same for an address.
   unsigned partition_offset = type * (config->m_memory_config_types->memory_config_array[0].m_n_mem_sub_partition);
   assert(partition_offset <= config->m_memory_config_types->memory_config_array[0].m_n_mem_sub_partition);

   if (type == 1) assert(partition_offset == config->m_memory_config_types->memory_config_array[0].m_n_mem_sub_partition);
   if (type == 0) assert(partition_offset == 0);

   assert(config->type == 1 || config->type == 2);

   config_type->m_address_mapping.addrdec_tlx_hetero(addr_temp, &m_raw_addr, partition_offset);
//   config_type->m_address_mapping.addrdec_tlx_hetero(access.get_addr(),&m_raw_addr, partition_offset);

   if (type == 1) assert(m_raw_addr.sub_partition >= config->m_memory_config_types->memory_config_array[0].m_n_mem_sub_partition);
   if (type == 0) assert(m_raw_addr.sub_partition < config->m_memory_config_types->memory_config_array[0].m_n_mem_sub_partition);

   m_partition_addr = config_type->m_address_mapping.partition_address(addr_temp);
//   m_partition_addr = config_type->m_address_mapping.partition_address(access.get_addr());
   m_type = m_access.is_write()?WRITE_REQUEST:READ_REQUEST;
   m_timestamp = gpu_sim_cycle + gpu_tot_sim_cycle;
   m_timestamp2 = 0;
   m_status = MEM_FETCH_INITIALIZED;
   m_status_change = gpu_sim_cycle + gpu_tot_sim_cycle;
   m_mem_config = config_type;
   if (access.get_addr() == 2187299968) {
       printf("addr: %lld, addr_limit: %lld, config_type: %d\n", access.get_addr(), config->addr_limit, m_mem_config->type);
   }
//   if ((access.get_addr() < config->addr_limit) && config->m_memory_config_types->enable_addr_limit) {
//       assert(m_mem_config->type == 1);
//   }

   icnt_flit_size = config_type->icnt_flit_size;
}

mem_fetch::~mem_fetch()
{
    if (m_access.get_type() < NUM_MEM_ACCESS_TYPE)
        deallocated[m_access.get_type()]++;
    deallocated_tot++;
    m_status = MEM_FETCH_DELETED;
}

#define MF_TUP_BEGIN(X) static const char* Status_str[] = {
#define MF_TUP(X) #X
#define MF_TUP_END(X) };
#include "mem_fetch_status.tup"
#undef MF_TUP_BEGIN
#undef MF_TUP
#undef MF_TUP_END

void mem_fetch::print( FILE *fp, bool print_inst ) const
{
    if( this == NULL ) {
        fprintf(fp," <NULL mem_fetch pointer>\n");
        return;
    }
    fprintf(fp,"  mf: uid=%6u, sid%02u:w%02u, part=%u, ", m_request_uid, m_sid, m_wid, m_raw_addr.chip );
    m_access.print(fp);
    if( (unsigned)m_status < NUM_MEM_REQ_STAT ) 
       fprintf(fp," status = %s (%llu), ", Status_str[m_status], m_status_change );
    else
       fprintf(fp," status = %u??? (%llu), ", m_status, m_status_change );
    if( !m_inst.empty() && print_inst ) m_inst.print(fp);
    else fprintf(fp,"\n");
}

void mem_fetch::set_status( enum mem_fetch_status status, unsigned long long cycle ) 
{
    assert(cycle >= m_status_change);
    request_status_vector[(unsigned)m_status] = cycle - m_status_change;
    m_status = status;
    m_status_change = cycle;
}

bool mem_fetch::isatomic() const
{
   if( m_inst.empty() ) return false;
   return m_inst.isatomic();
}

void mem_fetch::do_atomic()
{
    m_inst.do_atomic( m_access.get_warp_mask() );
}

bool mem_fetch::istexture() const
{
    if( m_inst.empty() ) return false;
    return m_inst.space.get_type() == tex_space;
}

bool mem_fetch::isconst() const
{ 
    if( m_inst.empty() ) return false;
    return (m_inst.space.get_type() == const_space) || (m_inst.space.get_type() == param_space_kernel);
}

/// Returns number of flits traversing interconnect. simt_to_mem specifies the direction
unsigned mem_fetch::get_num_flits(bool simt_to_mem){
	unsigned sz=0;
	// If atomic, write going to memory, or read coming back from memory, size = ctrl + data. Else, only ctrl
	if( isatomic() || (simt_to_mem && get_is_write()) || !(simt_to_mem || get_is_write()) )
		sz = size();
	else
		sz = get_ctrl_size();

	return (sz/icnt_flit_size) + ( (sz % icnt_flit_size)? 1:0);
}

void mem_fetch::printAllocated() {
    printf("allocated : %llu, deallocated: %llu\n", sm_next_mf_request_uid, deallocated);
}


