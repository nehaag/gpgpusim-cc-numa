// Copyright (c) 2009-2011, Tor M. Aamodt, Wilson W.L. Fung, Ali Bakhoda,
// Ivan Sham, George L. Yuan,
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
#include "gpu-misc.h"
#include "dram.h"
#include "mem_latency_stat.h"
#include "dram_sched.h"
#include "mem_fetch.h"
#include "l2cache.h"

//#define DRAM_VERIFY

#ifdef DRAM_VERIFY
int PRINT_CYCLE = 0;
#endif

template class fifo_pipeline<mem_fetch>;
template class fifo_pipeline<dram_req_t>;



dram_t::dram_t( unsigned int partition_id, const struct memory_config *config, memory_stats_t *stats,
                memory_partition_unit *mp )
{
   id = partition_id;
   m_memory_partition_unit = mp;
   m_stats = stats;
   m_config = config;

   CCDc = 0;
   RRDc = 0;
   RTWc = 0;
   WTRc = 0;

   rw = READ; //read mode is default

	bkgrp = (bankgrp_t**) calloc(sizeof(bankgrp_t*), m_config->nbkgrp);
	bkgrp[0] = (bankgrp_t*) calloc(sizeof(bank_t), m_config->nbkgrp);
	for (unsigned i=1; i<m_config->nbkgrp; i++) {
		bkgrp[i] = bkgrp[0] + i;
	}
	for (unsigned i=0; i<m_config->nbkgrp; i++) {
		bkgrp[i]->CCDLc = 0;
		bkgrp[i]->RTPLc = 0;
	}

   bk = (bank_t**) calloc(sizeof(bank_t*),m_config->nbk);
   bk[0] = (bank_t*) calloc(sizeof(bank_t),m_config->nbk);
   for (unsigned i=1;i<m_config->nbk;i++) 
      bk[i] = bk[0] + i;
   for (unsigned i=0;i<m_config->nbk;i++) {
      bk[i]->state = BANK_IDLE;
      bk[i]->bkgrpindex = i/(m_config->nbk/m_config->nbkgrp);
   }
   prio = 0;  
   rwq = new fifo_pipeline<dram_req_t>("rwq",m_config->CL,m_config->CL+1);
//   mrqq = new fifo_pipeline<dram_req_t>("mrqq",0,32:);
//   Changing size of mrqq to 64 from 32, for taking into account
//   migration
//   requests   
   mrqq = new fifo_pipeline<dram_req_t>("mrqq",0,64);
   returnq = new fifo_pipeline<mem_fetch>("dramreturnq",0,m_config->gpgpu_dram_return_queue_size==0?1024:m_config->gpgpu_dram_return_queue_size); 
   m_frfcfs_scheduler = NULL;
   if ( m_config->scheduler_type == DRAM_FRFCFS )
      m_frfcfs_scheduler = new frfcfs_scheduler(m_config,this,stats);
   n_cmd = 0;
   n_activity = 0;
   n_nop = 0; 
   n_act = 0; 
   n_pre = 0; 
   n_rd = 0;
   n_wr = 0;
   n_req = 0;
   max_mrqs_temp = 0;
   bwutil = 0;
   max_mrqs = 0;
   ave_mrqs = 0;

   // migration counters
   n_req_migration_read = 0;
   n_req_migration_write = 0;
   n_req_actual = 0;

   // migration count vectors
   num_migration_read.push_back(0);
   num_migration_write.push_back(0);
   num_actual.push_back(0);

   for (unsigned i=0;i<10;i++) {
      dram_util_bins[i]=0;
      dram_eff_bins[i]=0;
   }
   last_n_cmd = last_n_activity = last_bwutil = 0;

   n_cmd_partial = 0;
   n_activity_partial = 0;
   n_nop_partial = 0;  
   n_act_partial = 0;  
   n_pre_partial = 0;  
   n_req_partial = 0;
   ave_mrqs_partial = 0;
   bwutil_partial = 0;

   if ( queue_limit() )
      mrqq_Dist = StatCreate("mrqq_length",1, queue_limit());
   else //queue length is unlimited; 
      mrqq_Dist = StatCreate("mrqq_length",1,64); //track up to 64 entries

   cycle_count = 0;
   migrateReqCountR = 0;
   migrateReqCountW = 0;
   migrationTriggered = 0;
   pendingMigration = false;
}

bool dram_t::full() const 
{
    if(m_config->scheduler_type == DRAM_FRFCFS ){
        if(m_config->gpgpu_frfcfs_dram_sched_queue_size == 0 ) return false;
//        return m_frfcfs_scheduler->num_pending() >= m_config->gpgpu_frfcfs_dram_sched_queue_size;
        return ((m_frfcfs_scheduler->num_pending() >= m_config->gpgpu_frfcfs_dram_sched_queue_size) || (mrqq->full()));
    }
   else return mrqq->full();
}

unsigned dram_t::que_length() const
{
   unsigned nreqs = 0;
   if (m_config->scheduler_type == DRAM_FRFCFS ) {
      nreqs = m_frfcfs_scheduler->num_pending();
   } else {
      nreqs = mrqq->get_length();
   }
   return nreqs;
}

bool dram_t::returnq_full() const
{
   return returnq->full();
}

unsigned int dram_t::queue_limit() const 
{ 
   return m_config->gpgpu_frfcfs_dram_sched_queue_size; 
}


dram_req_t::dram_req_t( class mem_fetch *mf )
{
   txbytes = 0;
   dqbytes = 0;
   data = mf;

   const addrdec_t &tlx = mf->get_tlx_addr();

   bk  = tlx.bk; 
   row = tlx.row; 
   col = tlx.col; 
   nbytes = mf->get_data_size();

   timestamp = gpu_tot_sim_cycle + gpu_sim_cycle;
   addr = mf->get_addr();
   insertion_time = (unsigned) gpu_sim_cycle;
   rw = data->get_is_write()?WRITE:READ;
}

void dram_t::incrementVectors() {
    num_migration_read.push_back(0);
    num_migration_write.push_back(0);
    num_actual.push_back(0);
}

void dram_t::push( class mem_fetch *data ) 
{
    unsigned long long page_addr = data->get_addr() & ~(4095ULL);
    globalPageCount[page_addr][last_updated_at]++;

    if (data->get_addr() == 2152209376)
        printf("break here \n");
    if (id != data->get_tlx_addr().chip) {
        printf("WARNING: addr: %lld, id = %d, chip = %d, access_type: %d, uid = %u, timestamp= %u\n", data->get_addr(), id, data->get_tlx_addr().chip, data->get_access_type(), data->get_request_uid(), data->get_timestamp());
       /* Magically complete the request
        * TODO: COMMENT this part, after testing the overhead of migration
        */
        if (enableMigration && !flush_on_migration_enable) {
            fakeMigration(data);
            return;
        }
    }
    // TODO: enable the assert when done with the overhead testing
    assert(id == data->get_tlx_addr().chip); // Ensure request is in correct memory partition

   dram_req_t *mrq = new dram_req_t(data);
   data->set_status(IN_PARTITION_MC_INTERFACE_QUEUE,gpu_sim_cycle+gpu_tot_sim_cycle);
   mrqq->push(mrq);

   // if a writeback datauest from l2 has reached memory controller then remove it from the
   // l2 writeback map
//#ifdef DEBUG
//   printf("Erasing l2 writeback datauest as it has reached memory controller, addr: %llu, uid: %u\n", data->get_addr(), data->get_request_uid());
//#endif
   l2_wb_map.erase(data->get_request_uid());

   // stats...
   n_req += 1;
   n_req_partial += 1;
   if ( m_config->scheduler_type == DRAM_FRFCFS ) {
      unsigned nreqs = m_frfcfs_scheduler->num_pending();
      if ( nreqs > max_mrqs_temp)
         max_mrqs_temp = nreqs;
   } else {
      max_mrqs_temp = (max_mrqs_temp > mrqq->get_length())? max_mrqs_temp : mrqq->get_length();
   }
   m_stats->memlatstat_dram_access(data);

   if (data->get_access_type() == MEM_MIGRATE_R) {
       n_req_migration_read += 1;
       num_migration_read.back()++;
   } else if (data->get_access_type() == MEM_MIGRATE_W) {
       n_req_migration_write += 1;
       num_migration_write.back()++;
   } else {
       n_req_actual += 1;
       num_actual.back()++;
   }

   // for every request in MC queue, do
   unsigned row = data->get_tlx_addr().row;
   unsigned bank = data->get_tlx_addr().bk;
   request_dist[bank][row]++;
}

void dram_t::print_req_dist_stats() {
    // print the stats to the file
    std::map<unsigned, std::map<unsigned, unsigned long long> >::iterator it = request_dist.begin();
    unsigned sum_channel = 0;
    for (it; it != request_dist.end(); it++) {
//        printf("channel:%d, bank:%d, ", id, it->first);
        unsigned sum = 0;
        std::map<unsigned, unsigned long long>::iterator it2 = it->second.begin();
        for (it2; it2 != it->second.end(); it2++) {
//            printf("%d:%llu ", it2->first, it2->second);
            sum += it2->second;
        }
//        printf("total:%lld\n", sum);
        sum_channel += sum;
    }
    if (sum_channel != 0)
//        printf("cycle:%lld, channel:%d total: %lld\n", cycle_count, id, sum_channel);
    // reset request_distribution
    request_dist.clear();
}

void dram_t::scheduler_fifo()
{
   if (!mrqq->empty()) {
      unsigned int bkn;
      dram_req_t *head_mrqq = mrqq->top();
      head_mrqq->data->set_status(IN_PARTITION_MC_BANK_ARB_QUEUE,gpu_sim_cycle+gpu_tot_sim_cycle);
      bkn = head_mrqq->bk;
      if (!bk[bkn]->mrq) 
         bk[bkn]->mrq = mrqq->pop();
   }
}


#define DEC2ZERO(x) x = (x)? (x-1) : 0;
#define SWAP(a,b) a ^= b; b ^= a; a ^= b;

void dram_t::fakeMigration(class mem_fetch *data) {
    data->set_status(IN_PARTITION_MC_RETURNQ,gpu_sim_cycle+gpu_tot_sim_cycle);
    if( data->get_access_type() != L1_WRBK_ACC && data->get_access_type() != L2_WRBK_ACC && data->get_access_type() != MEM_MIGRATE_R && data->get_access_type() != MEM_MIGRATE_W) {
        data->set_reply();
        returnq->push(data);
    } else if (data->get_access_type() == MEM_MIGRATE_R) {
        migrateReqCountR++;
        //TODO:after 32 read reqs are done => data is in mem controller now
        //and we need to write this page to the destination memory
        //controller
        if (migrateReqCountR == 32) {
            //TODO: pass last parameter mem_type correctly here
            this->migratePage(destAddr, 0, destDramCtlr, 1, memConfigRemote, NULL, destMemType);
            migrateReqCountR = 0;
        }
        m_memory_partition_unit->set_done(data);
        delete data;
    } else if (data->get_access_type() == MEM_MIGRATE_W) {
        migrateReqCountW++;
        if (migrateReqCountW == 32) {
            //TODO:Send migration unit the call back that migration is done
            //for this memroy controller unit, Write Part!
            migrateReqCountW = 0;
            unsigned long long page_addr = data->get_addr() & ~(4095ULL);
            migrationQueue.erase(page_addr);
            migrationWaitCycle.erase(page_addr);
            // Timestamp at which front page's migration is complete
            migrationFinished[page_addr][3] = gpu_sim_cycle + gpu_tot_sim_cycle;
            sendForMigration.remove(page_addr);
            // Timestamp at which front page is blocked until
            // migration is completed
            migrationFinished[sendForMigration.front()][1] = gpu_sim_cycle + gpu_tot_sim_cycle;
            // Determine the source partition of the request and hence
            // remove the request from the respective queues
            //                      unsigned partition = whichDDRPartition(page_addr);
            //                      readyForNextMigration[partition] = true;
            //                      sendForMigrationPid[partition].remove(page_addr);
            readyForNextMigration = true;
        }
        m_memory_partition_unit->set_done(data);
        delete data;
    } else {
        m_memory_partition_unit->set_done(data);
        delete data;
    }
}

void dram_t::cycle()
{
    cycle_count++;
    if ((cycle_count % 10000) == 0) 
        print_req_dist_stats();
   if( !returnq->full() ) {
       dram_req_t *cmd = rwq->pop();
       if( cmd ) {
#ifdef DRAM_VIEWCMD 
           printf("\tDQ: BK%d Row:%03x Col:%03x", cmd->bk, cmd->row, cmd->col + cmd->dqbytes);
#endif
           cmd->dqbytes += m_config->dram_atom_size; 
           if (cmd->dqbytes >= cmd->nbytes) {
              mem_fetch *data = cmd->data; 
              data->set_status(IN_PARTITION_MC_RETURNQ,gpu_sim_cycle+gpu_tot_sim_cycle); 
              if( data->get_access_type() != L1_WRBK_ACC && data->get_access_type() != L2_WRBK_ACC && data->get_access_type() != MEM_MIGRATE_R && data->get_access_type() != MEM_MIGRATE_W) {
                 data->set_reply();
                 returnq->push(data);
              } else if (data->get_access_type() == MEM_MIGRATE_R) {
                  migrateReqCountR++;
                  //TODO:after 32 read reqs are done => data is in mem controller now
                  //and we need to write this page to the destination memory
                  //controller
                  if (migrateReqCountR == 32) {
                      //TODO: pass last parameter mem_type correctly here
                      this->migratePage(destAddr, 0, destDramCtlr, 1, memConfigRemote, NULL, destMemType);
                      migrateReqCountR = 0;
                  }
                 m_memory_partition_unit->set_done(data);
                 delete data;
              } else if (data->get_access_type() == MEM_MIGRATE_W) {
                  migrateReqCountW++;
                  if (migrateReqCountW == 32) {
                      //TODO:Send migration unit the call back that migration is done
                      //for this memroy controller unit, Write Part!
                      migrateReqCountW = 0;
                      unsigned long long page_addr = data->get_addr() & ~(4095ULL);
                      migrationQueue.erase(page_addr);
                      migrationWaitCycle.erase(page_addr);
                      // Timestamp at which front page's migration is complete
                      migrationFinished[page_addr][3] = gpu_sim_cycle + gpu_tot_sim_cycle;
                      sendForMigration.remove(page_addr);
                      // Determine the source partition of the request and hence
                      // remove the request from the respective queues
//                      unsigned partition = whichDDRPartition(page_addr);
//                      readyForNextMigration[partition] = true;
//                      sendForMigrationPid[partition].remove(page_addr);
                      readyForNextMigration = true;
                  }
                 m_memory_partition_unit->set_done(data);
                 delete data;
              } else {
                 m_memory_partition_unit->set_done(data);
                 delete data;
              }
              delete cmd;
           }
#ifdef DRAM_VIEWCMD 
           printf("\n");
#endif
       }
   }

   /* check if the upcoming request is on an idle bank */
   /* Should we modify this so that multiple requests are checked? */

   switch (m_config->scheduler_type) {
   case DRAM_FIFO: scheduler_fifo(); break;
   case DRAM_FRFCFS: scheduler_frfcfs(); break;
	default:
		printf("Error: Unknown DRAM scheduler type\n");
		assert(0);
   }
   if ( m_config->scheduler_type == DRAM_FRFCFS ) {
      unsigned nreqs = m_frfcfs_scheduler->num_pending();
      if ( nreqs > max_mrqs) {
         max_mrqs = nreqs;
      }
      ave_mrqs += nreqs;
      ave_mrqs_partial += nreqs;
   } else {
      if (mrqq->get_length() > max_mrqs) {
         max_mrqs = mrqq->get_length();
      }
      ave_mrqs += mrqq->get_length();
      ave_mrqs_partial +=  mrqq->get_length();
   }


   unsigned k=m_config->nbk;
   bool issued = false;

   // check if any bank is ready to issue a new read
   for (unsigned i=0;i<m_config->nbk;i++) {
      unsigned j = (i + prio) % m_config->nbk;
	  unsigned grp = j>>m_config->bk_tag_length;
      if (bk[j]->mrq) { //if currently servicing a memory request
          bk[j]->mrq->data->set_status(IN_PARTITION_DRAM,gpu_sim_cycle+gpu_tot_sim_cycle);
         // correct row activated for a READ
         if ( !issued && !CCDc && !bk[j]->RCDc &&
              !(bkgrp[grp]->CCDLc) &&
              (bk[j]->curr_row == bk[j]->mrq->row) && 
              (bk[j]->mrq->rw == READ) && (WTRc == 0 )  &&
              (bk[j]->state == BANK_ACTIVE) &&
              !rwq->full() ) {
            if (rw==WRITE) {
               rw=READ;
               rwq->set_min_length(m_config->CL);
            }
            rwq->push(bk[j]->mrq);
            bk[j]->mrq->txbytes += m_config->dram_atom_size; 
            CCDc = m_config->tCCD;
            bkgrp[grp]->CCDLc = m_config->tCCDL;
            RTWc = m_config->tRTW;
            bk[j]->RTPc = m_config->BL/m_config->data_command_freq_ratio;
            bkgrp[grp]->RTPLc = m_config->tRTPL;
            issued = true;
            n_rd++;
            bwutil += m_config->BL/m_config->data_command_freq_ratio;
            bwutil_partial += m_config->BL/m_config->data_command_freq_ratio;
            bk[j]->n_access++;
#ifdef DRAM_VERIFY
            PRINT_CYCLE=1;
            printf("\tcycle: %llu, channel: %d, \tRD  Bk:%d Row:%03x Col:%03x \n",
                   cycle_count, id, j, bk[j]->curr_row,
                   bk[j]->mrq->col + bk[j]->mrq->txbytes - m_config->dram_atom_size);
#endif            
            // transfer done
            if ( !(bk[j]->mrq->txbytes < bk[j]->mrq->nbytes) ) {
               bk[j]->mrq = NULL;
            }
         } else
            // correct row activated for a WRITE
            if ( !issued && !CCDc && !bk[j]->RCDWRc &&
                 !(bkgrp[grp]->CCDLc) &&
                 (bk[j]->curr_row == bk[j]->mrq->row)  && 
                 (bk[j]->mrq->rw == WRITE) && (RTWc == 0 )  &&
                 (bk[j]->state == BANK_ACTIVE) &&
                 !rwq->full() ) {
            if (rw==READ) {
               rw=WRITE;
               rwq->set_min_length(m_config->WL);
            }
            rwq->push(bk[j]->mrq);

            bk[j]->mrq->txbytes += m_config->dram_atom_size; 
            CCDc = m_config->tCCD;
            bkgrp[grp]->CCDLc = m_config->tCCDL;
            WTRc = m_config->tWTR; 
            bk[j]->WTPc = m_config->tWTP; 
            issued = true;
            n_wr++;
            bwutil += m_config->BL/m_config->data_command_freq_ratio;
            bwutil_partial += m_config->BL/m_config->data_command_freq_ratio;
#ifdef DRAM_VERIFY
            PRINT_CYCLE=1;
            printf("\tcycle: %llu, channel: %d, \tWR  Bk:%d Row:%03x Col:%03x \n",
                   cycle_count, id, j, bk[j]->curr_row, 
                   bk[j]->mrq->col + bk[j]->mrq->txbytes - m_config->dram_atom_size);
#endif  
            // transfer done 
            if ( !(bk[j]->mrq->txbytes < bk[j]->mrq->nbytes) ) {
               bk[j]->mrq = NULL;
            }
         }

         else
            // bank is idle
            if ( !issued && !RRDc && 
                 (bk[j]->state == BANK_IDLE) &&
                 !bk[j]->RPc && !bk[j]->RCc ) {
#ifdef DRAM_VERIFY
            PRINT_CYCLE=1;
            printf("\tcycle: %llu, channel: %d, \tACT BK:%d NewRow:%03x From:%03x \n",
                   cycle_count, id, j,bk[j]->mrq->row,bk[j]->curr_row);
#endif
            // activate the row with current memory request 
            bk[j]->curr_row = bk[j]->mrq->row;
            bk[j]->state = BANK_ACTIVE;
            RRDc = m_config->tRRD;
            bk[j]->RCDc = m_config->tRCD;
            bk[j]->RCDWRc = m_config->tRCDWR;
            bk[j]->RASc = m_config->tRAS;
            bk[j]->RCc = m_config->tRC;
            prio = (j + 1) % m_config->nbk;
            issued = true;
            n_act_partial++;
            n_act++;
         }

         else
            // different row activated
            if ( (!issued) && 
                 (bk[j]->curr_row != bk[j]->mrq->row) &&
                 (bk[j]->state == BANK_ACTIVE) && 
                 (!bk[j]->RASc && !bk[j]->WTPc && 
				  !bk[j]->RTPc &&
				  !bkgrp[grp]->RTPLc) ) {
            // make the bank idle again
            bk[j]->state = BANK_IDLE;
            bk[j]->RPc = m_config->tRP;
            prio = (j + 1) % m_config->nbk;
            issued = true;
            n_pre++;
            n_pre_partial++;
#ifdef DRAM_VERIFY
            PRINT_CYCLE=1;
            printf("cycle: %llu, channel: %d, \tPRE BK:%d Row:%03x \n", cycle_count, id, j,bk[j]->curr_row);
#endif
         }
      } else {
         if (!CCDc && !RRDc && !RTWc && !WTRc && !bk[j]->RCDc && !bk[j]->RASc
             && !bk[j]->RCc && !bk[j]->RPc  && !bk[j]->RCDWRc) k--;
         bk[j]->n_idle++;
      }
   }
   if (!issued) {
      n_nop++;
      n_nop_partial++;
#ifdef DRAM_VIEWCMD
      printf("\tNOP                        ");
#endif
   }
   if (k) {
      n_activity++;
      n_activity_partial++;
   }
   n_cmd++;
   n_cmd_partial++;

   // decrements counters once for each time dram_issueCMD is called
   DEC2ZERO(RRDc);
   DEC2ZERO(CCDc);
   DEC2ZERO(RTWc);
   DEC2ZERO(WTRc);
   for (unsigned j=0;j<m_config->nbk;j++) {
      DEC2ZERO(bk[j]->RCDc);
      DEC2ZERO(bk[j]->RASc);
      DEC2ZERO(bk[j]->RCc);
      DEC2ZERO(bk[j]->RPc);
      DEC2ZERO(bk[j]->RCDWRc);
      DEC2ZERO(bk[j]->WTPc);
      DEC2ZERO(bk[j]->RTPc);
   }
   for (unsigned j=0; j<m_config->nbkgrp; j++) {
	   DEC2ZERO(bkgrp[j]->CCDLc);
	   DEC2ZERO(bkgrp[j]->RTPLc);
   }

#ifdef DRAM_VISUALIZE
   visualize();
#endif
   assert(n_cmd == cycle_count);

   /* Check if there are any pending migration requests, if yes then send them
    * if possible
    */
   if (pendingMigration)
       resumeMigration();

   /* Wait for in-flight outstanding requests to clear up from the memory
    * controller and then only flag for migration
    */
    if (enableMigration && !migrationQueue.empty() && flush_on_migration_enable) {
//        std::map<unsigned long long, uint64_t>::iterator it = migrationQueue.begin();
//        for (; it != migrationQueue.end(); ++it) {
        unsigned long long page_addr_to_migrate = sendForMigration.front();
        std::map<unsigned long long, uint64_t>::iterator it = migrationQueue.find(page_addr_to_migrate);
        if (it != migrationQueue.end()) {
            if (it->second != 0 && it->second != (1<<43)) {
                new_addr_type page_addr = it->first & ~(4095ULL);
                if (outstandingRequest(it->first) == 3) {
                    /* if L2 has flushed all the dirty lines and all the pending
                     * reads are done, then clear bit 1 of the second variable of map
                     */
                    if (checkAllBitsBelowReset(it->second,41))
                        resetBit(it->second, 41);
                }
            }
        }
    }
}

unsigned dram_t::outstandingRequest(new_addr_type page_addr)
{
    // Scan through mrqq list and check if there are any request to this page
    fifo_data<dram_req_t> *head_mrqq = mrqq->fifo_data_top();
    while (head_mrqq != NULL) {
        unsigned long long mrqq_page_addr = head_mrqq->m_data->addr & ~(4095ULL);
        if (mrqq_page_addr == page_addr)
            return 2;
        head_mrqq = head_mrqq->m_next;
    }
    return 3;
}

//if mrq is being serviced by dram, gets popped after CL latency fulfilled
class mem_fetch* dram_t::return_queue_pop() 
{
    return returnq->pop();
}

class mem_fetch* dram_t::return_queue_top() 
{
    return returnq->top();
}

void dram_t::print( FILE* simFile) const
{
   unsigned i;
   fprintf(simFile,"DRAM[%d]: %d bks, busW=%d BL=%d CL=%d, ", 
           id, m_config->nbk, m_config->busW, m_config->BL, m_config->CL );
   fprintf(simFile,"tRRD=%d tCCD=%d, tRCD=%d tRAS=%d tRP=%d tRC=%d\n",
           m_config->tCCD, m_config->tRRD, m_config->tRCD, m_config->tRAS, m_config->tRP, m_config->tRC );
   fprintf(simFile,"n_cmd=%d n_nop=%d n_act=%d n_pre=%d n_req=%d n_rd=%d n_write=%d bw_util=%.4g\n",
           n_cmd, n_nop, n_act, n_pre, n_req, n_rd, n_wr,
           (float)bwutil/n_cmd);
   
   // Migration stats
   fprintf(simFile, "n_migration_read = %u, n_migration_write = %u\n", n_req_migration_read, n_req_migration_write);
   fprintf(simFile, "n_migration = %u, n_actual = %u\n", n_req_migration_read + n_req_migration_write, n_req_actual);
   unsigned int tot = n_req_migration_read + n_req_migration_write + n_req_actual;
   if (tot != n_req)
       fprintf(simFile, "ERROR!!!!, check stats not equal\n");
   printMigrationStats(simFile);


   fprintf(simFile,"n_activity=%d dram_eff=%.4g\n",
           n_activity, (float)bwutil/n_activity);
   for (i=0;i<m_config->nbk;i++) {
      fprintf(simFile, "bk%d: %da %di ",i,bk[i]->n_access,bk[i]->n_idle);
   }
   fprintf(simFile, "\n");
   fprintf(simFile, "dram_util_bins:");
   for (i=0;i<10;i++) fprintf(simFile, " %d", dram_util_bins[i]);
   fprintf(simFile, "\ndram_eff_bins:");
   for (i=0;i<10;i++) fprintf(simFile, " %d", dram_eff_bins[i]);
   fprintf(simFile, "\n");
   if(m_config->scheduler_type== DRAM_FRFCFS) 
       fprintf(simFile, "mrqq: max=%d avg=%g\n", max_mrqs, (float)ave_mrqs/n_cmd);
}

void dram_t::visualize() const
{
   printf("RRDc=%d CCDc=%d mrqq.Length=%d rwq.Length=%d\n", 
          RRDc, CCDc, mrqq->get_length(),rwq->get_length());
   for (unsigned i=0;i<m_config->nbk;i++) {
      printf("BK%d: state=%c curr_row=%03x, %2d %2d %2d %2d %p ", 
             i, bk[i]->state, bk[i]->curr_row,
             bk[i]->RCDc, bk[i]->RASc,
             bk[i]->RPc, bk[i]->RCc,
             bk[i]->mrq );
      if (bk[i]->mrq)
         printf("txf: %d %d", bk[i]->mrq->nbytes, bk[i]->mrq->txbytes);
      printf("\n");
   }
   if ( m_frfcfs_scheduler ) 
      m_frfcfs_scheduler->print(stdout);
}

void dram_t::print_stat( FILE* simFile ) 
{
   fprintf(simFile,"DRAM (%d): n_cmd=%d n_nop=%d n_act=%d n_pre=%d n_req=%d n_rd=%d n_write=%d bw_util=%.4g ",
           id, n_cmd, n_nop, n_act, n_pre, n_req, n_rd, n_wr,
           (float)bwutil/n_cmd);

   fprintf(simFile, "n_migration_read = %u, n_migration_write = %u\n", n_req_migration_read, n_req_migration_write);
   fprintf(simFile, "n_migration = %u, n_actual = %u\n", n_req_migration_read + n_req_migration_write, n_req_actual);
   unsigned int tot = n_req_migration_read + n_req_migration_write + n_req_actual;
   if (tot != n_req)
       fprintf(simFile, "ERROR!!!!, check stats not equal\n");
   printMigrationStats(simFile);
   
   fprintf(simFile, "mrqq: %d %.4g mrqsmax=%d ", max_mrqs, (float)ave_mrqs/n_cmd, max_mrqs_temp);
   fprintf(simFile, "\n");
   fprintf(simFile, "dram_util_bins:");
   for (unsigned i=0;i<10;i++) fprintf(simFile, " %d", dram_util_bins[i]);
   fprintf(simFile, "\ndram_eff_bins:");
   for (unsigned i=0;i<10;i++) fprintf(simFile, " %d", dram_eff_bins[i]);
   fprintf(simFile, "\n");
   max_mrqs_temp = 0;
}

void dram_t::visualizer_print( gzFile visualizer_file )
{
   // dram specific statistics
   gzprintf(visualizer_file,"dramncmd: %u %u\n",id, n_cmd_partial);  
   gzprintf(visualizer_file,"dramnop: %u %u\n",id,n_nop_partial);
   gzprintf(visualizer_file,"dramnact: %u %u\n",id,n_act_partial);
   gzprintf(visualizer_file,"dramnpre: %u %u\n",id,n_pre_partial);
   gzprintf(visualizer_file,"dramnreq: %u %u\n",id,n_req_partial);
   gzprintf(visualizer_file,"dramavemrqs: %u %u\n",id,
            n_cmd_partial?(ave_mrqs_partial/n_cmd_partial ):0);

   // utilization and efficiency
   gzprintf(visualizer_file,"dramutil: %u %u\n",  
            id,n_cmd_partial?100*bwutil_partial/n_cmd_partial:0);
   gzprintf(visualizer_file,"drameff: %u %u\n", 
            id,n_activity_partial?100*bwutil_partial/n_activity_partial:0);

   // reset for next interval
   bwutil_partial = 0;
   n_activity_partial = 0;
   ave_mrqs_partial = 0; 
   n_cmd_partial = 0;
   n_nop_partial = 0;
   n_act_partial = 0;
   n_pre_partial = 0;
   n_req_partial = 0;

   // dram access type classification
   for (unsigned j = 0; j < m_config->nbk; j++) {
      gzprintf(visualizer_file,"dramglobal_acc_r: %u %u %u\n", id, j, 
               m_stats->mem_access_type_stats[GLOBAL_ACC_R][id][j]);
      gzprintf(visualizer_file,"dramglobal_acc_w: %u %u %u\n", id, j, 
               m_stats->mem_access_type_stats[GLOBAL_ACC_W][id][j]);
      gzprintf(visualizer_file,"dramlocal_acc_r: %u %u %u\n", id, j, 
               m_stats->mem_access_type_stats[LOCAL_ACC_R][id][j]);
      gzprintf(visualizer_file,"dramlocal_acc_w: %u %u %u\n", id, j, 
               m_stats->mem_access_type_stats[LOCAL_ACC_W][id][j]);
      gzprintf(visualizer_file,"dramconst_acc_r: %u %u %u\n", id, j, 
               m_stats->mem_access_type_stats[CONST_ACC_R][id][j]);
      gzprintf(visualizer_file,"dramtexture_acc_r: %u %u %u\n", id, j, 
               m_stats->mem_access_type_stats[TEXTURE_ACC_R][id][j]);
   }
}


void dram_t::set_dram_power_stats(	unsigned &cmd,
									unsigned &activity,
									unsigned &nop,
									unsigned &act,
									unsigned &pre,
									unsigned &rd,
									unsigned &wr,
									unsigned &req) const{

	// Point power performance counters to low-level DRAM counters
	cmd = n_cmd;
	activity = n_activity;
	nop = n_nop;
	act = n_act;
	pre = n_pre;
	rd = n_rd;
	wr = n_wr;
	req = n_req;
}

unsigned int dram_t::migratePage(unsigned long long int source_addr, unsigned long long int dest_addr, dram_t *dest_dram_ctrl, unsigned int req_type, const class memory_config *mem_config_local, const class memory_config *mem_config_remote, unsigned mem_type_func) 
{
    assert(req_type == 0 || req_type == 1);

    //determine which dram controller request needs to be sent, if it is a read
    //request, then it is local controller, else if it is a write request then
    //it is destination controller
    if (req_type == 0)   //0: Read request
        dram_ctrl = this;
    else
        dram_ctrl = dest_dram_ctrl;

    dram_ctrl->migrationTriggered = 1;
    dram_ctrl->destDramCtlr = dest_dram_ctrl;
    dram_ctrl->sourceAddr = source_addr;
    dram_ctrl->destAddr = dest_addr;
    dram_ctrl->memConfigLocal = mem_config_local;
    dram_ctrl->memConfigRemote = mem_config_remote;
    dram_ctrl->destMemType = 1U ^ mem_type_func;
    dram_ctrl->mem_type = mem_type_func;
    dram_ctrl->reqType = req_type;

    //Push 32 read request to the channel to a given source address
    int i;
    for (i=0; i<32; i++) {
        if (dram_ctrl->sendMigrationRequest(source_addr + (i) * 128ULL))
            continue;
        else {
            dram_ctrl->pendingMigration = true;
            dram_ctrl->numRequestPending = 32 - i;
            dram_ctrl->isWritePending = reqType;
            break;
        }
    }
    return i;
}

void dram_t::resumeMigration() {
    for (int i=(32 - numRequestPending); i < 32; i++) {
         if (sendMigrationRequest(sourceAddr + (i) * 128ULL))
            continue;
        else {
            pendingMigration = true;
            numRequestPending = 32 - i;
            isWritePending = reqType;
            return;
        }
    }
    pendingMigration = false;
}

bool dram_t::sendMigrationRequest(unsigned long long addr) {
   if (!full() && !mrqq->full()) {
       mem_fetch *mf;
       //create a new mf for the next packet
       if (reqType == 0) {
          mem_access_t access(MEM_MIGRATE_R, addr, 128U, 0);
           mf = new mem_fetch( access, 
                               READ_PACKET_SIZE,
                               memConfigLocal, mem_type);
       } else {
         mem_access_t access(MEM_MIGRATE_W, addr, 128U, 0);
         mf = new mem_fetch( access, 
                             WRITE_PACKET_SIZE,
                             memConfigLocal, mem_type);
       }

       //push the request in the memory controller
       push(mf);
       return true;
   } else {
       return false; 
   }
}

void dram_t::printMigrationStats( FILE* simFile) const
{
    unsigned int i;
    assert (num_migration_read.size() == num_migration_write.size());
    assert (num_migration_read.size() == num_actual.size());

    fprintf(simFile, "Migration reads distribution with time \n");
    for (i=0; i<num_migration_read.size(); i++) {
        fprintf(simFile, "%u %u %u %u\n", i, num_migration_read[i], num_migration_write[i], num_actual[i]);
    }
}

unsigned dram_t::whichDDRPartition(unsigned long long page_addr) {
    mem_access_t accessSDDR(MEM_MIGRATE_R, page_addr, 128U, 0);
    const class memory_config* memConfigSDDR = &(memConfigLocal->m_memory_config_types->memory_config_array[0]);
    mem_fetch *mfSDDR = new mem_fetch( accessSDDR, 
            READ_PACKET_SIZE, 
            memConfigSDDR, 0);
    //    unsigned global_spidSDDR = mfSDDR->get_sub_partition_id(); 
    unsigned global_spidSDDR = mfSDDR->get_tlx_addr().chip; 
    delete mfSDDR;
    return global_spidSDDR;
}

unsigned dram_t::getTotReq() {
    return n_req;
}
