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
#endif
