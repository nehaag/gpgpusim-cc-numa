#ifndef MEM_MIGRATE_H
#define MEM_MIGRATE_H

#include <iostream>
#include <fstream>
#include <list>
#include <stdio.h>
#include <time.h>


#include "dram.h"
#include "mem_fetch.h"
#include "addrdec.h"
#include "l2cache.h"

typedef unsigned long long int mem_addr;

class migrate{
    public:
        /*SDDR Page address to send to HBM */
        mem_addr addrMigrateToHBM;
        /*HBM victim page to send to SDDR */
        mem_addr addrMigrateToSDDR;
        const memory_config *memConfigHBM;
        const memory_config *memConfigSDDR;

        /*
         * memory partition unitt pointers to determine which dram to send data
         * to
         */
        class memory_partition_unit **mMemoryPartitionUnit;

        /*Constructor */
        migrate(const class memory_config *config_SDDR, const class memory_config *config_HBM, class memory_partition_unit **m_memory_partition_unit);
        /*Migrate addresses as trigerred by the policy
         * Swap pages of BO and CO memory
         */
        void migratePage(mem_addr addrToHBM, mem_addr addrToSDDR);
        /* Migrate addresses from CO -> BO memory
         * unidirectional
         * overloaded function
         */
        void migratePage(mem_addr addrToHBM);

        /*Select a victim page in HBM to be migrated to SDDR */
        mem_addr selectHBMVictim();

        /*Monitor the accesses to all pages in the memory */
        void monitorPages();
};

//class migrationReq {
//   /* Data members */
//   unsigned long long addr;
//   const class memory_config *memConfigSource;
//   const class memory_config *memConfigDest;
//   dram_t *sourceCtrl;
//   dram_t *destCtrl;
//   /* request type 0 means its a read, 1 == write */
//   bool reqType;
//   unsigned totalNumReqs;
//   unsigned numReqsInMigration;
//   unsigned numReqsPendingForMigration;
//   unsigned numReqsMigrationDone;
//   /* state can take 4 values:
//    * 1: Waiting for sending atleast 1 cacheline for migration
//    * 2: Sent more than 1 but less than 32 requests for migration
//    * 3: Sent all 32 req for migration
//    * 4: Migration complete
//    */
//   unsigned state;
//
//   /* Member functions */
//   migrationReq(unsigned long long req_addr, const class memory_config *mem_config_source, const class memory_config *mem_config_dest, dram_t * source_ctrl, dram_t *dest_ctrl, bool req_type) {
//        addr = addr;
//        memConfigSource = mem_config_source;
//        memConfigDest = mem_config_dest;
//        sourceCtrl = source_ctrl;
//        destCtrl = dest_ctrl;
//        reqType = req_type;
//        totalNumReqs = 32;
//        numReqsInMigration = 0;
//        numReqsPendingForMigration = 32;
//        numReqsMigrationDone = 0;
//        state = 1;
//   }
//   unsigned sendMigrationRequest();
//};
#endif
