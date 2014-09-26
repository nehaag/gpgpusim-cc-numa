/// Copyright (c) 2009-2011, Tor M. Aamodt
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

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <list>
#include <set>

#include "../option_parser.h"
#include "mem_fetch.h"
#include "dram.h"
#include "gpu-cache.h"
#include "histogram.h"
#include "l2cache.h"
#include "../statwrapper.h"
#include "../abstract_hardware_model.h"
#include "gpu-sim.h"
#include "shader.h"
#include "mem_latency_stat.h"
#include "l2cache_trace.h"

std::vector <unsigned long long int> latency_breakdown_all_req(10, 0);

mem_fetch * partition_mf_allocator::alloc(new_addr_type addr, mem_access_type type, unsigned size, bool wr ) const 
{
    assert( wr );
    mem_access_t access( type, addr, size, wr );
    mem_fetch *mf = new mem_fetch( access, 
                                   NULL,
                                   WRITE_PACKET_SIZE, 
                                   -1, 
                                   -1, 
                                   -1,
                                   m_memory_config );
        //TODO: for debugging: delete this
        unsigned global_spid = mf->get_sub_partition_id(); 
        const memory_config* config = mf->get_mem_config();
//        assert(config == m_memory_config);
        if (config->type == 2 && (global_spid < config->m_memory_config_types->memory_config_array[0].m_n_mem_sub_partition))
            printf("global_spid: %d, addr: %lld\n", global_spid, mf->get_addr());
        if (config->type == 2) assert(global_spid >= config->m_memory_config_types->memory_config_array[0].m_n_mem_sub_partition);
        if (config->type == 1) assert(global_spid < config->m_memory_config_types->memory_config_array[0].m_n_mem_sub_partition);

    return mf;
}

memory_partition_unit::memory_partition_unit( unsigned partition_id, 
                                              const struct memory_config *config,
                                              class memory_stats_t *stats,
                                              unsigned long int *epoch_number)
: m_id(partition_id), m_config(config), m_stats(stats), m_arbitration_metadata(config), m_epoch_number(epoch_number) 
{
    unsigned dram_id = m_id;
    if (config->type == 2) dram_id -= m_config->m_memory_config_types->memory_config_array[0].m_n_mem;
    m_id_local = dram_id;
    m_dram = new dram_t(dram_id,m_config,m_stats,this);
//   m_dram = new dram_t(m_id,m_config,m_stats,this);

    m_sub_partition = new memory_sub_partition*[m_config->m_n_sub_partition_per_memory_channel]; 
    for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel; p++) {
        unsigned sub_partition_id = m_id * m_config->m_n_sub_partition_per_memory_channel + p; 
        m_sub_partition[p] = new memory_sub_partition(sub_partition_id, m_config, stats); 
    }

    //// Initialize the epoch number
    //epoch_number = 0;

}

memory_partition_unit::~memory_partition_unit() 
{
    delete m_dram; 
    for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel; p++) {
        delete m_sub_partition[p]; 
    } 
    delete[] m_sub_partition; 
}

memory_partition_unit::arbitration_metadata::arbitration_metadata(const struct memory_config *config) 
: m_last_borrower(config->m_n_sub_partition_per_memory_channel - 1), 
  m_private_credit(config->m_n_sub_partition_per_memory_channel, 0), 
  m_shared_credit(0) 
{
    // each sub partition get at least 1 credit for forward progress 
    // the rest is shared among with other partitions 
    m_private_credit_limit = 1; 
    m_shared_credit_limit = config->gpgpu_frfcfs_dram_sched_queue_size 
                            + config->gpgpu_dram_return_queue_size 
                            - (config->m_n_sub_partition_per_memory_channel - 1); 
    if (config->gpgpu_frfcfs_dram_sched_queue_size == 0 
        or config->gpgpu_dram_return_queue_size == 0) 
    {
        m_shared_credit_limit = 0; // no limit if either of the queue has no limit in size 
    }
    assert(m_shared_credit_limit >= 0); 
}

bool memory_partition_unit::arbitration_metadata::has_credits(int inner_sub_partition_id) const 
{
    int spid = inner_sub_partition_id; 
    if (m_private_credit[spid] < m_private_credit_limit) {
        return true; 
    } else if (m_shared_credit_limit == 0 || m_shared_credit < m_shared_credit_limit) {
        return true; 
    } else {
        return false; 
    }
}

void memory_partition_unit::arbitration_metadata::borrow_credit(int inner_sub_partition_id) 
{
    int spid = inner_sub_partition_id; 
    if (m_private_credit[spid] < m_private_credit_limit) {
        m_private_credit[spid] += 1; 
    } else if (m_shared_credit_limit == 0 || m_shared_credit < m_shared_credit_limit) {
        m_shared_credit += 1; 
    } else {
        assert(0 && "DRAM arbitration error: Borrowing from depleted credit!"); 
    }
    m_last_borrower = spid; 
}

void memory_partition_unit::arbitration_metadata::return_credit(int inner_sub_partition_id) 
{
    int spid = inner_sub_partition_id; 
    if (m_private_credit[spid] > 0) {
        m_private_credit[spid] -= 1; 
    } else {
        m_shared_credit -= 1; 
    } 
    assert((m_shared_credit >= 0) && "DRAM arbitration error: Returning more than available credits!"); 
}

void memory_partition_unit::arbitration_metadata::print( FILE *fp ) const 
{
    fprintf(fp, "private_credit = "); 
    for (unsigned p = 0; p < m_private_credit.size(); p++) {
        fprintf(fp, "%d ", m_private_credit[p]); 
    }
    fprintf(fp, "(limit = %d)\n", m_private_credit_limit); 
    fprintf(fp, "shared_credit = %d (limit = %d)\n", m_shared_credit, m_shared_credit_limit); 
}

bool memory_partition_unit::busy() const 
{
    bool busy = false; 
    for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel; p++) {
        if (m_sub_partition[p]->busy()) {
            busy = true; 
        }
    }
    return busy; 
}

void memory_partition_unit::cache_cycle(unsigned cycle) 
{
    for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel; p++) {
        m_sub_partition[p]->cache_cycle(cycle); 
    }
}

void memory_partition_unit::visualizer_print( gzFile visualizer_file ) const 
{
    m_dram->visualizer_print(visualizer_file);
    for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel; p++) {
        m_sub_partition[p]->visualizer_print(visualizer_file); 
    }
}

// determine whether a given subpartition can issue to DRAM 
bool memory_partition_unit::can_issue_to_dram(int inner_sub_partition_id) 
{
    int spid = inner_sub_partition_id; 
    bool sub_partition_contention = m_sub_partition[spid]->dram_L2_queue_full(); 
    bool has_dram_resource = m_arbitration_metadata.has_credits(spid); 

    MEMPART_DPRINTF("sub partition %d sub_partition_contention=%c has_dram_resource=%c\n", 
                    spid, (sub_partition_contention)? 'T':'F', (has_dram_resource)? 'T':'F'); 

    return (has_dram_resource && !sub_partition_contention); 
}

int memory_partition_unit::global_sub_partition_id_to_local_id(int global_sub_partition_id) const
{
    return (global_sub_partition_id - m_id * m_config->m_n_sub_partition_per_memory_channel); 
}

void memory_partition_unit::dram_cycle() 
{ 
    // pop completed memory request from dram and push it to dram-to-L2 queue 
    // of the original sub partition 
    mem_fetch* mf_return = m_dram->return_queue_top();
    if (mf_return) {
        unsigned dest_global_spid = mf_return->get_sub_partition_id(); 
        int dest_spid = global_sub_partition_id_to_local_id(dest_global_spid); 
//        assert(mf_return->get_mem_config() == m_config);
        if (mf_return->get_mem_config() != m_config) {
            printf("WARNING: there might be an error here, as memory packet is not in the correct memory subpartition");
//            dest_spid = dest_global_spid - m_id_local * m_config->m_n_sub_partition_per_memory_channel;
        }
        // TODO: un-comment after testing for blocking
//        assert(m_sub_partition[dest_spid]->get_id() == dest_global_spid); 
        if (!m_sub_partition[dest_spid]->dram_L2_queue_full()) {
            if( mf_return->get_access_type() == L1_WRBK_ACC ) {
                m_sub_partition[dest_spid]->set_done(mf_return); 
                delete mf_return;
            } else {
                m_sub_partition[dest_spid]->dram_L2_queue_push(mf_return);
                mf_return->set_status(IN_PARTITION_DRAM_TO_L2_QUEUE,gpu_sim_cycle+gpu_tot_sim_cycle);
                m_arbitration_metadata.return_credit(dest_spid); 
                MEMPART_DPRINTF("mem_fetch request %p return from dram to sub partition %d\n", mf_return, dest_spid); 
            }
            m_dram->return_queue_pop(); 
        }
    } else {
        m_dram->return_queue_pop(); 
    }
    
    m_dram->cycle(); 
    m_dram->dram_log(SAMPLELOG);   

    if( !m_dram->full() ) {
        // L2->DRAM queue to DRAM latency queue
        // Arbitrate among multiple L2 subpartitions 
        int last_issued_partition = m_arbitration_metadata.last_borrower(); 
        for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel; p++) {
            int spid = (p + last_issued_partition + 1) % m_config->m_n_sub_partition_per_memory_channel; 
            if (!m_sub_partition[spid]->L2_dram_queue_empty() && can_issue_to_dram(spid)) {
                mem_fetch *mf = m_sub_partition[spid]->L2_dram_queue_top();

            //TODO: for debugging: delete this
            unsigned global_spid = mf->get_sub_partition_id(); 
            if (mf->get_mem_config()->type == 2 && global_spid < mf->get_mem_config()->m_memory_config_types->memory_config_array[0].m_n_mem_sub_partition)
                printf("addr: %lld, global_spid: %d", mf->get_addr(), global_spid);
            if (mf->get_mem_config()->type == 2) assert(global_spid >= mf->get_mem_config()->m_memory_config_types->memory_config_array[0].m_n_mem_sub_partition);
            if (mf->get_mem_config()->type == 1) assert(global_spid < mf->get_mem_config()->m_memory_config_types->memory_config_array[0].m_n_mem_sub_partition);


                m_sub_partition[spid]->L2_dram_queue_pop();
                MEMPART_DPRINTF("Issue mem_fetch request %p from sub partition %d to dram\n", mf, spid); 

//                //DRAM memory trace
//                printf("MEM_TRACE: %lld, %d, %lld\n", gpu_sim_cycle+gpu_tot_sim_cycle, mf->is_write(), mf->get_addr());

                dram_delay_t d;
                d.req = mf;
                d.ready_cycle = gpu_sim_cycle+gpu_tot_sim_cycle + m_config->dram_latency;
                m_dram_latency_queue.push_back(d);

                // Add uniques addresses to the cacheline tracking data structure
                // If address is not present then add 0 accesses for all
                // previous epochs and also current
                unsigned long long int cacheline = (mf->get_addr()) & ~(4095ULL);

                // number of epochs this address was not accessed
                int diff = 0;
                // check if an element already exists
                if (num_access_per_cacheline.count(cacheline))
                    diff = *m_epoch_number - (num_access_per_cacheline[cacheline].size() - 4);
                else {
                    diff = *m_epoch_number;

                    // Store the address decoding at dram level in the map
                    const addrdec_t &tlx = mf->get_tlx_addr();
                    num_access_per_cacheline[cacheline].push_back(tlx.bk);
                    num_access_per_cacheline[cacheline].push_back(tlx.row);
                    num_access_per_cacheline[cacheline].push_back(tlx.col);
                    // To store the sum of accesses in all epochs
                    num_access_per_cacheline[cacheline].push_back(0);
                }


                // push 0(s) for all the epochs till now
                for (unsigned int i = 0; i < (diff+1); ++i) {
                    num_access_per_cacheline[cacheline].push_back(0);
                    // updated only once per epoch
                    //if (i == diff)
                    //    reuse_distance_across_epoch[cacheline].push_back(gpu_tot_sim_cycle + gpu_sim_cycle);
                }
                // Increment the acccesses per epoch
                num_access_per_cacheline[cacheline][*m_epoch_number+4] += 1;
                
                // Total number of accesses, summing all the epochs
                num_access_per_cacheline[cacheline][3] += 1;
                if (num_access_per_cacheline[cacheline][3] == 1) {
                    if (mf->get_sub_partition_id() < 8)
                        migrationFinished[cacheline][4] = gpu_sim_cycle + gpu_tot_sim_cycle;
                    else
                        migrationFinished[cacheline][5] = gpu_sim_cycle + gpu_tot_sim_cycle;
                }

                // Track cycle/timestamp when number of accesses to a page becomes 1, 32,
                // 64, 96, 128, 160 and 192
                unsigned long int access_local = num_access_per_cacheline[cacheline][3]/32;
                unsigned access_bin = (access_local >= 6) ? 6 : access_local;
                if(!threshold_cycle.count(cacheline)) {
                    for (int i=0; i<7; i++)
                        threshold_cycle[cacheline].push_back(0);
                } else if (threshold_cycle[cacheline][access_bin] == 0) {
                    threshold_cycle[cacheline][access_bin] = gpu_tot_sim_cycle + gpu_sim_cycle;
                }

                /* Trigger migration
                 * do not migrate if request is already in the correct
                 * portion, currently hardcoded to 2 subpartitions
                 */
                unsigned page_ratio = m_config->m_memory_config_types->page_ratio;
                unsigned long long pages = m_config->m_memory_config_types->pages;
                new_addr_type page_addr = mf->get_addr() & ~(4095ULL);
//                if (enableMigration && !pauseMigration &&
//                         (migrationFinished.size() < page_ratio/100.0*pages)
                if(enableMigration
                        && !pauseMigration
                        && (mf->get_sub_partition_id() < 8)
                        && (mf->get_access_type() != INST_ACC_R)
                        && (rand() % 100 < page_ratio)
                        && (num_access_per_cacheline[page_addr][3] >= migration_threshold))
                {
                    /* Range expansion
                     */
                    unsigned count = 0;
                    for (long long i=(-range_expansion); i<=range_expansion && (count <= max_migrations); i++) {
//                    for (long long i=(2*range_expansion); i>=0; i--) {
                        unsigned long long page_addr_in_range = page_addr + i*4096;
                        if (!migrationQueue.count(page_addr_in_range)
                                && !migrationFinished.count(page_addr_in_range)
                                && page_addr_in_range >= GLOBAL_HEAP_START
                                && page_addr_in_range < 274877906944) {
                            if (!flush_on_migration_enable)
                                migrationQueue[page_addr_in_range] = 0;
                            else
                                migrationQueue[page_addr_in_range] = ((1ULL << 42) - 1ULL);
                            // Timestamp at which a page is marked for migration
                            migrationFinished[page_addr_in_range][0] = gpu_sim_cycle + gpu_tot_sim_cycle;
                            migrationWaitCycle[page_addr_in_range] = 0;
                            /* Determine which partition this request belongs to and
                             * accordingly push in the respective queue. This part is to
                             * be used when we want to enable parallel migrations of the
                             * memory controllers
                             */
                            unsigned partition = whichDDRPartition(page_addr_in_range, mf->get_mem_config());
                            // TODO: generalize 4 to number of DDR memory
                            // controllers
                            assert(partition < 4);
                            
                            sendForMigrationPid[partition].push_back(page_addr_in_range);
                            count++;
                        }
                    }
                }

                // update last access re-use distance stats
                // local update
                //reuse_distance_per_epoch[cacheline].push_back(gpu_tot_sim_cycle + gpu_sim_cycle);

                mf->set_status(IN_PARTITION_DRAM_LATENCY_QUEUE,gpu_sim_cycle+gpu_tot_sim_cycle);
                m_arbitration_metadata.borrow_credit(spid); 
                break;  // the DRAM should only accept one request per cycle 
            }
        }
    }

    // DRAM latency queue
    if( !m_dram_latency_queue.empty() && ( (gpu_sim_cycle+gpu_tot_sim_cycle) >= m_dram_latency_queue.front().ready_cycle ) && !m_dram->full() ) {
        mem_fetch* mf = m_dram_latency_queue.front().req;
        m_dram_latency_queue.pop_front();
        m_dram->push(mf);

        //TODO: for debugging: delete this
        unsigned global_spid = mf->get_sub_partition_id(); 
        if (mf->get_mem_config()->type == 2) assert(global_spid >= mf->get_mem_config()->m_memory_config_types->memory_config_array[0].m_n_mem_sub_partition);
        if (mf->get_mem_config()->type == 1) assert(global_spid < mf->get_mem_config()->m_memory_config_types->memory_config_array[0].m_n_mem_sub_partition);
    }

    if (enableMigration 
            && !pauseMigration
            && !migrationQueue.empty() && flush_on_migration_enable) {
        for (auto &it_pid : sendForMigrationPid) {
            if (it_pid.second.empty())
                continue;
            unsigned long long page_addr_to_migrate = it_pid.second.front();
            auto it = migrationQueue.find(page_addr_to_migrate);
            if (it != migrationQueue.end()) {
                if (it->second != 0 && it->second != (1<<43)) {
                    new_addr_type page_addr = it->first & ~(4095ULL);
                    bool flag = true;
                    for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel; p++) {
                        flag = m_sub_partition[p]->snoop_L2_dram_queue(page_addr);
                    }
                    // Check if any requests are there in DRAM latency queue
                    for (auto &it1 : m_dram_latency_queue) {
                        new_addr_type req_page_addr = it1.req->get_addr() & ~(4095ULL);
                        if (req_page_addr == page_addr)
                            flag = false;
                    }
                    if (flag && checkAllBitsBelowReset(it->second,40)){
                        resetBit(it->second, 40);
                    }
                }
            }
        }
    }
}

void memory_partition_unit::set_done( mem_fetch *mf )
{
    unsigned global_spid = mf->get_sub_partition_id(); 
    int spid = global_sub_partition_id_to_local_id(global_spid);
    if (mf->get_mem_config() != m_config) {
        printf("WARNING: there might be an error here, as memory packet is not in the correct memory subpartition, faulitng addr = %lld\n", mf->get_addr());
//        spid = global_spid - m_id_local * m_config->m_n_sub_partition_per_memory_channel;
    }
//    assert(mf->get_mem_config() == m_config);
//    if (mf->get_mem_config()->type == 1)
//        assert(m_id < mf->get_mem_config()->m_n_mem);
    if (mf->get_mem_config()->type == 2) assert(global_spid >= mf->get_mem_config()->m_memory_config_types->memory_config_array[0].m_n_mem_sub_partition);
    if (mf->get_mem_config()->type == 1) assert(global_spid < mf->get_mem_config()->m_memory_config_types->memory_config_array[0].m_n_mem_sub_partition);
    if (spid >= mf->get_mem_config()->m_n_sub_partition_per_memory_channel || spid < 0) {
        printf("addr: %ld, mem_type: %d, global_spid: %d, spid: %d, mid: %d\n", mf->get_addr(), mf->get_mem_config()->type, global_spid, spid, m_id);
        assert(spid > 0);
        assert(spid < mf->get_mem_config()->m_n_sub_partition_per_memory_channel);
    }
    assert(m_sub_partition[spid]->get_id() == global_spid); 
    if (mf->get_access_type() == L1_WRBK_ACC || mf->get_access_type() == L2_WRBK_ACC) {
        m_arbitration_metadata.return_credit(spid); 
        MEMPART_DPRINTF("mem_fetch request %p return from dram to sub partition %d\n", mf, spid); 
    }
    m_sub_partition[spid]->set_done(mf); 
}

void memory_partition_unit::set_dram_power_stats(unsigned &n_cmd,
                                                 unsigned &n_activity,
                                                 unsigned &n_nop,
                                                 unsigned &n_act,
                                                 unsigned &n_pre,
                                                 unsigned &n_rd,
                                                 unsigned &n_wr,
                                                 unsigned &n_req) const
{
    m_dram->set_dram_power_stats(n_cmd, n_activity, n_nop, n_act, n_pre, n_rd, n_wr, n_req);
}

void memory_partition_unit::print( FILE *fp )
{
    fprintf(fp, "Memory Partition %u: \n", m_id); 
    for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel; p++) {
        m_sub_partition[p]->print(fp); 
    }
    fprintf(fp, "In Dram Latency Queue (total = %zd): \n", m_dram_latency_queue.size()); 
    for (std::list<dram_delay_t>::const_iterator mf_dlq = m_dram_latency_queue.begin(); 
         mf_dlq != m_dram_latency_queue.end(); ++mf_dlq) {
        mem_fetch *mf = mf_dlq->req; 
        fprintf(fp, "Ready @ %llu - ", mf_dlq->ready_cycle); 
        if (mf) 
            mf->print(fp); 
        else 
            fprintf(fp, " <NULL mem_fetch?>\n"); 
    }
    m_dram->print(fp); 

//    /*
    // Unique accesses stats
    fprintf(fp, "Cache line map size: %ld \n", num_access_per_cacheline.size());
    fprintf(fp, "Addr Accesses\n");
    std::map<unsigned long long int, std::vector<unsigned long int> >::iterator it = num_access_per_cacheline.begin();
    std::map<unsigned long long int, std::vector<unsigned long long int> >::iterator it_cycle = threshold_cycle.begin();
    //std::map<unsigned long long int, std::vector<unsigned long int> >::iterator it1 = reuse_distance_per_epoch.begin();
    //std::map<unsigned long long int, std::vector<unsigned long int> >::iterator it2 = reuse_distance_across_epoch.begin();
    unsigned int sum = 0;
    for (; it != num_access_per_cacheline.end(); ++it) {
        fprintf(fp, "%lld ", it->first);
        for (int i=0; i<it_cycle->second.size(); ++i) {
            fprintf(fp, "%lld ", it_cycle->second[i]);
        }
        ++it_cycle;
//        sum = 0;
        for (int i=0; i<it->second.size(); ++i) {
//            if (i>3) 
//                sum += it->second[i];
//            else {
                fprintf(fp, "%ld ", it->second[i]);
//            }
        }
//        fprintf(fp, " %d\n", sum);
        fprintf(fp, "\n");
    }
//    */

//    printf("total latency breakdown: ");
//    for (unsigned i = 10; i < 19; i++) {
//        printf("%llu ", latency_breakdown_all_req[i]);
//    }
//    printf("\n");

    //// re-use distance stats
    //fprintf(fp, "Re-use distance in this kernel\n");
    //it = reuse_distance_per_epoch.begin();
    //for (; it != reuse_distance_per_epoch.end(); ++it) {
    //    fprintf(fp, "%lld: ", it->first);
    //    for (int i=0; i<it->second.size(); ++i)
    //        fprintf(fp, "%ld ", it->second[i]);
    //    fprintf(fp, "\n");

    //    // update the global re-use map
    //    reuse_distance_across_epoch[it->first].push_back(it->second.back());
    //}
    //fprintf(fp, "\n");
    //reuse_distance_per_epoch.clear();

    //fprintf(fp, "Global re-use distance in this kernel\n");
    //it = reuse_distance_across_epoch.begin();
    //for (; it != reuse_distance_across_epoch.end(); ++it) {
    //    fprintf(fp, "%lld: ", it->first);
    //    for (int i=0; i<it->second.size(); ++i)
    //        fprintf(fp, "%ld ", it->second[i]);
    //    fprintf(fp, "\n");
    //}

    //++epoch_number;
}

memory_sub_partition::memory_sub_partition( unsigned sub_partition_id, 
                                            const struct memory_config *config,
                                            class memory_stats_t *stats )
{
    m_id = sub_partition_id;
    m_config=config;
    m_stats=stats;

//    assert(m_id < m_config->m_n_mem_sub_partition); 

    char L2c_name[32];
    snprintf(L2c_name, 32, "L2_bank_%03d", m_id);
    m_L2interface = new L2interface(this);
    m_mf_allocator = new partition_mf_allocator(config);

    if(!m_config->m_L2_config.disabled())
       m_L2cache = new l2_cache(L2c_name,m_config->m_L2_config,m_id,-1,m_L2interface,m_mf_allocator,IN_PARTITION_L2_MISS_QUEUE);

    unsigned int icnt_L2;
    unsigned int L2_dram;
    unsigned int dram_L2;
    unsigned int L2_icnt;
    sscanf(m_config->gpgpu_L2_queue_config,"%u:%u:%u:%u", &icnt_L2,&L2_dram,&dram_L2,&L2_icnt );
    m_icnt_L2_queue = new fifo_pipeline<mem_fetch>("icnt-to-L2",0,icnt_L2); 
    m_L2_dram_queue = new fifo_pipeline<mem_fetch>("L2-to-dram",0,L2_dram);
    m_dram_L2_queue = new fifo_pipeline<mem_fetch>("dram-to-L2",0,dram_L2);
    m_L2_icnt_queue = new fifo_pipeline<mem_fetch>("L2-to-icnt",0,L2_icnt);
    wb_addr=-1;
}

memory_sub_partition::~memory_sub_partition()
{
    delete m_icnt_L2_queue;
    delete m_L2_dram_queue;
    delete m_dram_L2_queue;
    delete m_L2_icnt_queue;
    delete m_L2cache;
    delete m_L2interface;
}

void print_latency_breakdown_stat(mem_fetch* mf) {
    // Print the latency break up map now
    unsigned sum = 0;
    for (unsigned i = 10; i < 19; i++) {
        sum += mf->request_status_vector[i];
    }
    if (sum > 0) {
//        printf("latency breakdown: %llu: ch:%d bk:%d row:%d ", mf->get_addr(), mf->get_tlx_addr().chip, mf->get_tlx_addr().bk, mf->get_tlx_addr().row);
        for (unsigned i = 10; i < 19; i++) {
//            printf("%llu ", mf->request_status_vector[i]);
            latency_breakdown_all_req[i-10] += mf->request_status_vector[i];
        }
//        printf("\n");
    }
}

void memory_sub_partition::cache_cycle( unsigned cycle )
{
    // L2 fill responses
    if( !m_config->m_L2_config.disabled()) {
       if ( m_L2cache->access_ready() && !m_L2_icnt_queue->full() ) {
           mem_fetch *mf = m_L2cache->next_access();
           if(mf->get_access_type() != L2_WR_ALLOC_R){ // Don't pass write allocate read request back to upper level cache
				mf->set_reply();
				mf->set_status(IN_PARTITION_L2_TO_ICNT_QUEUE,gpu_sim_cycle+gpu_tot_sim_cycle);
//                print_latency_breakdown_stat(mf);
				m_L2_icnt_queue->push(mf);
           }else{
				m_request_tracker.erase(mf);
				delete mf;
           }
       }
    }

    // DRAM to L2 (texture) and icnt (not texture)
    if ( !m_dram_L2_queue->empty() ) {
        mem_fetch *mf = m_dram_L2_queue->top();
        if ( !m_config->m_L2_config.disabled() && m_L2cache->waiting_for_fill(mf) ) {
            if (m_L2cache->fill_port_free()) {
                mf->set_status(IN_PARTITION_L2_FILL_QUEUE,gpu_sim_cycle+gpu_tot_sim_cycle);
                m_L2cache->fill(mf,gpu_sim_cycle+gpu_tot_sim_cycle);
                m_dram_L2_queue->pop();
            }
        } else if ( !m_L2_icnt_queue->full() ) {
            mf->set_status(IN_PARTITION_L2_TO_ICNT_QUEUE,gpu_sim_cycle+gpu_tot_sim_cycle);
//            print_latency_breakdown_stat(mf);
            m_L2_icnt_queue->push(mf);
            m_dram_L2_queue->pop();
        }
    }

    // prior L2 misses inserted into m_L2_dram_queue here
    if( !m_config->m_L2_config.disabled() )
       m_L2cache->cycle(true);

    // new L2 texture accesses and/or non-texture accesses
    if ( !m_L2_dram_queue->full() && !m_icnt_L2_queue->empty() ) {
        mem_fetch *mf = m_icnt_L2_queue->top();
        if ( !m_config->m_L2_config.disabled() &&
              ( (m_config->m_L2_texure_only && mf->istexture()) || (!m_config->m_L2_texure_only) )
           ) {
            // L2 is enabled and access is for L2
            bool output_full = m_L2_icnt_queue->full(); 
            bool port_free = m_L2cache->data_port_free(); 
            if ( !output_full && port_free ) {
                std::list<cache_event> events;
                enum cache_request_status status = m_L2cache->access(mf->get_addr(),mf,gpu_sim_cycle+gpu_tot_sim_cycle,events);
                bool write_sent = was_write_sent(events);
                bool read_sent = was_read_sent(events);

//                enum cache_request_status status = HIT;
                if ( status == HIT ) {
                    if( !write_sent ) {
                        // L2 cache replies
                        assert(!read_sent);
                        if( mf->get_access_type() == L1_WRBK_ACC ) {
                            m_request_tracker.erase(mf);
                            delete mf;
                        } else {
                            mf->set_reply();
                            mf->set_status(IN_PARTITION_L2_TO_ICNT_QUEUE,gpu_sim_cycle+gpu_tot_sim_cycle);
//                            print_latency_breakdown_stat(mf);
                            m_L2_icnt_queue->push(mf);
                        }
                        m_icnt_L2_queue->pop();
                    } else {
                        assert(write_sent);
                        m_icnt_L2_queue->pop();
                    }
                } else if ( status != RESERVATION_FAIL ) {
                    // L2 cache accepted request
                    m_icnt_L2_queue->pop();
                } else {
                    assert(!write_sent);
                    assert(!read_sent);
                    // L2 cache lock-up: will try again next cycle
                }
            }
        } else {
            // L2 is disabled or non-texture access to texture-only L2
            mf->set_status(IN_PARTITION_L2_TO_DRAM_QUEUE,gpu_sim_cycle+gpu_tot_sim_cycle);
            m_L2_dram_queue->push(mf);
            m_icnt_L2_queue->pop();

            //TODO: for debugging: delete this
            unsigned global_spid = mf->get_sub_partition_id(); 
            if (mf->get_mem_config()->type == 2) assert(global_spid >= mf->get_mem_config()->m_memory_config_types->memory_config_array[0].m_n_mem_sub_partition);
            if (mf->get_mem_config()->type == 1) assert(global_spid < mf->get_mem_config()->m_memory_config_types->memory_config_array[0].m_n_mem_sub_partition);
        }
    }

    // ROP delay queue
    if( !m_rop.empty() && (cycle >= m_rop.front().ready_cycle) && !m_icnt_L2_queue->full() ) {
        mem_fetch* mf = m_rop.front().req;
        m_rop.pop();
        l1_wb_map.erase(mf->get_request_uid());
        l1_wr_miss_no_wa_map.erase(mf->get_request_uid());
        m_icnt_L2_queue->push(mf);
        mf->set_status(IN_PARTITION_ICNT_TO_L2_QUEUE,gpu_sim_cycle+gpu_tot_sim_cycle);

        //TODO: for debugging: delete this
        unsigned global_spid = mf->get_sub_partition_id(); 
        if (mf->get_mem_config()->type == 2) assert(global_spid >= mf->get_mem_config()->m_memory_config_types->memory_config_array[0].m_n_mem_sub_partition);
        if (mf->get_mem_config()->type == 1) assert(global_spid < mf->get_mem_config()->m_memory_config_types->memory_config_array[0].m_n_mem_sub_partition);
    }

    if (enableMigration 
            && !pauseMigration
            && !migrationQueue.empty() && flush_on_migration_enable) {
        for (auto &it_pid : sendForMigrationPid) {
            if (it_pid.second.empty())
                continue;
            unsigned long long page_addr_to_migrate = it_pid.second.front();
            auto it = migrationQueue.find(page_addr_to_migrate);
            if (it != migrationQueue.end()) {
                if (it->second != 0 && it->second != (1<<43)) {
                    new_addr_type page_addr = it->first & ~(4095ULL);
                    /* check icnt to l2 queue
                    */
                    // Scan through mrqq list and check if there are any request to this page
                    fifo_data<mem_fetch> *head_icnt_L2_queue = m_icnt_L2_queue->fifo_data_top();
                    bool flag = true;
                    while (head_icnt_L2_queue != NULL) {
                        unsigned long long req_page_addr = head_icnt_L2_queue->m_data->get_addr() & ~(4095ULL);
                        if (req_page_addr == page_addr)
                            flag = false;
                        head_icnt_L2_queue = head_icnt_L2_queue->m_next;
                    }
                    if (flag && checkAllBitsBelowReset(it->second,15)){
                        resetBit(it->second, 15);
                    }

                    /* check mshrs of L2 caches
                    */
                    if (m_L2cache->flushOnMigrate(it->first)) {
                        /* if L2 has flushed all the dirty lines and all the pending
                         * reads are done, then clear bit 1 of the second variable of map
                         */
                        unsigned cache_this_id = m_L2cache->cache_id;
                        if (flag && checkAllBitsBelowReset(it->second,16)){
                            resetBit(it->second, 16 + cache_this_id);
                        }
                    }
                }
            }
        }
    }
}

bool memory_sub_partition::full() const
{
    return m_icnt_L2_queue->full();
}

bool memory_sub_partition::L2_dram_queue_empty() const
{
   return m_L2_dram_queue->empty(); 
}

class mem_fetch* memory_sub_partition::L2_dram_queue_top() const
{
   return m_L2_dram_queue->top(); 
}

void memory_sub_partition::L2_dram_queue_pop() 
{
   m_L2_dram_queue->pop(); 
}

bool memory_sub_partition::dram_L2_queue_full() const
{
   return m_dram_L2_queue->full(); 
}

void memory_sub_partition::dram_L2_queue_push( class mem_fetch* mf )
{
   m_dram_L2_queue->push(mf); 
}

void memory_sub_partition::print_cache_stat(unsigned &accesses, unsigned &misses) const
{
    FILE *fp = stdout;
    if( !m_config->m_L2_config.disabled() )
       m_L2cache->print(fp,accesses,misses);
}

void memory_sub_partition::print( FILE *fp ) const
{
    if ( !m_request_tracker.empty() ) {
        fprintf(fp,"Memory Sub Parition %u: pending memory requests:\n", m_id);
        for ( std::set<mem_fetch*>::const_iterator r=m_request_tracker.begin(); r != m_request_tracker.end(); ++r ) {
            mem_fetch *mf = *r;
            if ( mf )
                mf->print(fp);
            else
                fprintf(fp," <NULL mem_fetch?>\n");
        }
    }
    if( !m_config->m_L2_config.disabled() )
       m_L2cache->display_state(fp);
}

void memory_stats_t::visualizer_print( gzFile visualizer_file )
{
   // gzprintf(visualizer_file, "Ltwowritemiss: %d\n", L2_write_miss);
   // gzprintf(visualizer_file, "Ltwowritehit: %d\n",  L2_write_access-L2_write_miss);
   // gzprintf(visualizer_file, "Ltworeadmiss: %d\n", L2_read_miss);
   // gzprintf(visualizer_file, "Ltworeadhit: %d\n", L2_read_access-L2_read_miss);
   if (num_mfs)
      gzprintf(visualizer_file, "averagemflatency: %lld\n", mf_total_lat/num_mfs);
}

void gpgpu_sim::print_dram_stats(FILE *fout) const
{
	unsigned cmd=0;
	unsigned activity=0;
	unsigned nop=0;
	unsigned act=0;
	unsigned pre=0;
	unsigned rd=0;
	unsigned wr=0;
	unsigned req=0;
	unsigned tot_cmd=0;
	unsigned tot_nop=0;
	unsigned tot_act=0;
	unsigned tot_pre=0;
	unsigned tot_rd=0;
	unsigned tot_wr=0;
	unsigned tot_req=0;

	for (unsigned i=0;i<m_memory_config->m_n_mem;i++){
		m_memory_partition_unit[i]->set_dram_power_stats(cmd,activity,nop,act,pre,rd,wr,req);
		tot_cmd+=cmd;
		tot_nop+=nop;
		tot_act+=act;
		tot_pre+=pre;
		tot_rd+=rd;
		tot_wr+=wr;
		tot_req+=req;
	}
    fprintf(fout,"gpgpu_n_dram_reads = %d\n",tot_rd );
    fprintf(fout,"gpgpu_n_dram_writes = %d\n",tot_wr );
    fprintf(fout,"gpgpu_n_dram_activate = %d\n",tot_act );
    fprintf(fout,"gpgpu_n_dram_commands = %d\n",tot_cmd);
    fprintf(fout,"gpgpu_n_dram_noops = %d\n",tot_nop );
    fprintf(fout,"gpgpu_n_dram_precharges = %d\n",tot_pre );
    fprintf(fout,"gpgpu_n_dram_requests = %d\n",tot_req );
}

unsigned memory_sub_partition::flushL2() 
{ 
    if (!m_config->m_L2_config.disabled()) {
        m_L2cache->flush(); 
    }
    return 0; // L2 is read only in this version
}

bool memory_sub_partition::busy() const 
{
    return !m_request_tracker.empty();
}

void memory_sub_partition::push( mem_fetch* req, unsigned long long cycle ) 
{
    if (req) {
        m_request_tracker.insert(req);

        // if a writeback request from l1 has reached l2 then remove it from the
        // l1 writeback map
//#ifdef DEBUG
//        printf("Erasing l1 writeback request as it has reached l2, addr: %llu, uid: %u\n", req->get_addr(), req->get_request_uid());
//#endif
//        l1_wb_map.erase(req->get_request_uid());
//        l1_wr_miss_no_wa_map.erase(req->get_request_uid());
        
        m_stats->memlatstat_icnt2mem_pop(req);
        if( req->istexture() ) {
            m_icnt_L2_queue->push(req);

            //TODO: for debugging: delete this
            unsigned global_spid = req->get_sub_partition_id(); 
            const memory_config* config = req->get_mem_config();
            if (req->get_mem_config()->type == 2) 
                assert(global_spid >= req->get_mem_config()->m_memory_config_types->memory_config_array[0].m_n_mem_sub_partition);
            if (req->get_mem_config()->type == 1) 
                assert(global_spid < req->get_mem_config()->m_memory_config_types->memory_config_array[0].m_n_mem_sub_partition);
            req->set_status(IN_PARTITION_ICNT_TO_L2_QUEUE,gpu_sim_cycle+gpu_tot_sim_cycle);
        } else {
            rop_delay_t r;
            r.req = req;
            r.ready_cycle = cycle + m_config->rop_latency;
            m_rop.push(r);
            req->set_status(IN_PARTITION_ROP_DELAY,gpu_sim_cycle+gpu_tot_sim_cycle);
        }
    }
}

mem_fetch* memory_sub_partition::pop() 
{
    mem_fetch* mf = m_L2_icnt_queue->pop();
    m_request_tracker.erase(mf);
    if ( mf && mf->isatomic() )
        mf->do_atomic();
    if( mf && (mf->get_access_type() == L2_WRBK_ACC || mf->get_access_type() == L1_WRBK_ACC) ) {
        delete mf;
        mf = NULL;
    } 
    return mf;
}

mem_fetch* memory_sub_partition::top() 
{
    mem_fetch *mf = m_L2_icnt_queue->top();
    if( mf && (mf->get_access_type() == L2_WRBK_ACC || mf->get_access_type() == L1_WRBK_ACC) ) {
        m_L2_icnt_queue->pop();
        m_request_tracker.erase(mf);
        delete mf;
        mf = NULL;
    } 
    return mf;
}

void memory_sub_partition::set_done( mem_fetch *mf )
{
    m_request_tracker.erase(mf);
}

void memory_sub_partition::accumulate_L2cache_stats(class cache_stats &l2_stats) const {
    if (!m_config->m_L2_config.disabled()) {
        l2_stats += m_L2cache->get_stats();
    }
}

void memory_sub_partition::get_L2cache_sub_stats(struct cache_sub_stats &css) const{
    if (!m_config->m_L2_config.disabled()) {
        m_L2cache->get_sub_stats(css);
    }
}

void memory_sub_partition::visualizer_print( gzFile visualizer_file )
{
    // TODO: Add visualizer stats for L2 cache 
}

void memory_partition_unit::printNumAccess(unsigned long long addr) {
    if (num_access_per_cacheline.count(addr)) {
        printf("addr: %llu, accesses: %u \n", addr, num_access_per_cacheline[addr][3]);
    }
}

bool memory_sub_partition::snoop_L2_dram_queue(unsigned long long page_addr) {
    // Check if there are any requests in L2->dram queue in the
    // subpartitions
    bool flag = true;
    fifo_data<mem_fetch> *head_L2_dram_queue = m_L2_dram_queue->fifo_data_top();
    while (head_L2_dram_queue != NULL) {
        new_addr_type req_page_addr = head_L2_dram_queue->m_data->get_addr() & ~(4095ULL);
        if (req_page_addr == page_addr)
            flag = false;
        head_L2_dram_queue = head_L2_dram_queue->m_next;
    }
    return flag;
}

void memory_partition_unit::printNumAccessToPage() {
    std::map<unsigned long long int, std::vector<unsigned long int> >::iterator it = num_access_per_cacheline.begin();
    for (; it != num_access_per_cacheline.end(); ++it) {
        printf("addr: %llu, accesses: %u\n", it->first, (it->second)[3]);
    }
}

unsigned memory_partition_unit::getTotDramReq() {
    return m_dram->getTotReq();
}
