#ifndef STREAMMEASUREMENTSYSTEM_MRAC_H
#define STREAMMEASUREMENTSYSTEM_MRAC_H

#include "../../common/BOBHash32.h"
#include "../../common/EMFSD.h"
#include "../../common/wmrd_calculator.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

using std::string;
double sample_p = 1;

template <uint32_t key_length, uint32_t memory_in_bytes> class MRAC {
        static constexpr uint32_t w = memory_in_bytes / 4;
        uint32_t counters[w];
        BOBHash32 *bob_hash;

        double est_cardinality = 0;

        EMFSD *em_fsd_algo = NULL;

      public:
        string name;
        int biggest_counter = 0;

        MRAC() {
                srand(time(0));
                memset(counters, 0, sizeof(counters));
                bob_hash = new BOBHash32(rand() % MAX_PRIME32);

                std::stringstream name_buffer;
                name_buffer << "MRAC@" << memory_in_bytes;
                name = name_buffer.str();
        }

        void insert(uint8_t *item) {
                uint32_t pos =
                    bob_hash->run((const char *)item, key_length) % w;
                counters[pos] += 1;
                biggest_counter = biggest_counter < counters[pos] ? counters[pos] : biggest_counter;
        }

        int query(uint8_t *item) {
                uint32_t pos =
                    bob_hash->run((const char *)item, key_length) % w;
                    return counters[pos];
        }

        void collect_fsd() {
                em_fsd_algo = new EMFSD();
                em_fsd_algo->set_counters(w, counters);
        }



        void next_epoch() { em_fsd_algo->next_epoch(); }

        void get_distribution(vector<double> &dist_est) {
                // ADD
                collect_fsd();
                puts("collect finished");
                for (int i=1; i<=10; i++){
                        printf("epoch %d\n", i);
                        next_epoch();
                        cout << "next epoch finished" << endl;
                }
                dist_est = em_fsd_algo->ns;
        }

        double get_cardinality() {
                //        if (est_cardinality == 0) {
                //            calc_distribution();
                //        }
                return em_fsd_algo->n_sum;
        }
};

#endif // STREAMMEASUREMENTSYSTEM_MRAC_H
