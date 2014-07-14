#include "mem_fetch.h"
#include "mem_latency_stat.h"
#include "shader.h"
#include "visualizer.h"
#include "gpu-sim.h"
#include "addrdec.h"
#include "migrate.h"



migrate::migrate(const class memory_config *config_SDDR, const class memory_config *config_HBM, class memory_partition_unit **m_memory_partition_unit) {
    addrMigrateToHBM = 0;
    addrMigrateToSDDR = 0;
    memConfigSDDR = config_SDDR;
    memConfigHBM = config_HBM;
    mMemoryPartitionUnit = m_memory_partition_unit;
}

void migrate::migratePage(mem_addr addrToHBM, mem_addr addrToSDDR) {
    //TODO: assign memConfigSDDR and memConfigHBM approporiately
    addrMigrateToHBM = addrToHBM;
    addrMigrateToSDDR = addrToSDDR;

    /*Determine the page address */
    //TODO: page size is kept as 2kB, need to change it to 4kB
    mem_addr pageAddrToHBM = (addrToHBM >> 11ULL) << 11ULL;
    mem_addr pageAddrToSDDR = (addrToSDDR >> 11ULL) << 11ULL;

    /*
     * Build the packet mem_fetch and determine the global_spid for the packet
     */
    /*SDDR request */
    mem_access_t accessSDDR(MEM_MIGRATE_R, pageAddrToHBM, 128U, 0);
    mem_fetch *mfSDDR = new mem_fetch( accessSDDR, 
                                   READ_PACKET_SIZE, 
                                   memConfigSDDR, 0);
    unsigned global_spidSDDR = mfSDDR->get_sub_partition_id() - 1; 

    /*HBM request */
    mem_access_t accessHBM(MEM_MIGRATE_R, pageAddrToSDDR, 128U, 0);
    mem_fetch *mfHBM = new mem_fetch( accessHBM, 
                                   READ_PACKET_SIZE, 
                                   memConfigHBM, 1);
    unsigned global_spidHBM = mfHBM->get_sub_partition_id() - 1;

    /*Determine the dram controller pointer */
    class dram_t *m_dramSDDR = mMemoryPartitionUnit[global_spidSDDR]->get_dram();
    class dram_t *m_dramHBM = mMemoryPartitionUnit[global_spidHBM]->get_dram();

    /* 
     * Delete the mem_fetch packets as new will be created in the respective
     * controllers, we created them earlier to get the global_spid of the
     * respective memory technology
     */
    delete mfSDDR;
    delete mfHBM;
    
    /*Send the page migration request to respective DRAM controllers */
    unsigned int numReqHBM = m_dramHBM->migratePage(pageAddrToSDDR,
            pageAddrToHBM, m_dramSDDR, 0, memConfigHBM, memConfigSDDR, 1);
    unsigned int numReqSDDR = m_dramSDDR->migratePage(pageAddrToHBM,
            pageAddrToSDDR, m_dramHBM, 0, memConfigSDDR, memConfigHBM, 0);

    //TODO:if numReqHBM and numReqSDDR != 16, then need to call migrate page
    //again

    /*Update the global structure for page mapping */
    m_map_online[addrToHBM]  = 1;
    m_map[addrToHBM]  = 1;
    m_map_online[addrToSDDR]  = 0;
    m_map[addrToSDDR]  = 0;
}

mem_addr migrate::selectHBMVictim() {
    return 0;
}

void migrate::monitorPages() {
}
