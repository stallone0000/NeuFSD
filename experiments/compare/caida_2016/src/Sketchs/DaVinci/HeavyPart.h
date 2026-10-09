#ifndef _HEAVYPART_H_
#define _HEAVYPART_H_

#include "common/param.h"

template<int bucket_num>
class HeavyPart
{
public:
    int num_bucket;
    int hp_memory_access_counter;
    alignas(64) Bucket* buckets;
    HeavyPart(int _num_bucket);
    ~HeavyPart();

    void clear();

    int insert(uint8_t *key, uint8_t *swap_key, uint32_t &swap_val, uint32_t f = 1);
    int quick_insert(uint8_t *key, uint32_t f = 1);

    int query(uint8_t *key);

    int get_memory_usage();
    int get_bucket_num();
private:
    int CalculateFP(uint8_t *key, uint32_t &fp);
};
template class HeavyPart<2400>;

#endif //_HEAVYPART_H_