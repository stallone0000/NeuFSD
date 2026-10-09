#include "../common/BOBHash32.h"
#include <string>
#include <iostream>
#include "../../common_func.h"
using namespace std;

enum Type
{
    CM,
    CU,
    half_CU,
    Count
};

void cardPrintTowerInfo(int width1, int mask1, int width2, int mask2, int width3, int mask3, int threshold, Type type, int maximum) {
    const int width = 55;
    const std::string line(width, '-');
    const std::string space(width , ' ');

    std::cout << "╭" << line << "╮" << std::endl;
    std::cout << "│" << space << "│" << std::endl;
    
    std::string title = "Tower Filter";
    int titleLength = title.length();
    int titlePadding = (width - titleLength) / 2;
    std::cout << "│" << std::string(titlePadding, ' ') << title << std::string(width - titlePadding - titleLength, ' ') << "│" << std::endl;
    
    std::cout << "│" << space << "│" << std::endl;
    std::cout << "│   Parameters:" << std::string(width - 14, ' ') << "│" << std::endl;
    
    std::printf("│     Width1              = %-7d%*s│\n", width1, width - 34, "");
    std::printf("│     mask1               = %-7d%*s│\n", mask1, width - 34, "");
    std::printf("│     Width2              = %-7d%*s│\n", width2, width - 34, "");
    std::printf("│     mask2               = %-7d%*s│\n", mask2, width - 34, "");
    std::printf("│     Width3              = %-7d%*s│\n", width3, width - 34, "");
    std::printf("│     mask3               = %-7d%*s│\n", mask3, width - 34, "");
    std::printf("│     threshold           = %-7d%*s│\n", threshold, width - 34, "");
    std::printf("│     version             = %-7d%*s│\n", type, width - 34, "");
    std::printf("│     maximum             = %-7d%*s│\n", maximum, width - 34, "");
    
    std::cout << "│" << space << "│" << std::endl;
    std::cout << "╰" << line << "╯" << std::endl;
}

class Counters {
public:
    uint8_t mask1 = 3;
    uint16_t mask2 = 15;
    int mem;
    uint8_t *counters;
    int width;
    int counter_w; // counter width in bits

    Counters(int WIDTH, int COUNTER_W) {
        width = WIDTH;
        counter_w = COUNTER_W;
        // Calculate memory required based on counter width
        mem = (WIDTH * COUNTER_W + 7) / 8; // ceil to nearest byte
        counters = new uint8_t[mem];
        memset(counters, 0, mem);
    }

    ~Counters() {
        delete[] counters;
    }

    uint32_t index(int idx) {
        int byteIndex = (idx * counter_w) / 8;
        int bitOffset = (idx * counter_w) % 8;
        switch (counter_w) {
            case 1:
            case 2:
            case 4: {
                uint32_t mask = (1 << counter_w) - 1;
                return (counters[byteIndex] >> bitOffset) & mask;
            }
            case 8:
                return counters[idx];
            default:
                return 0;
        }
    }

    void increment(int idx) {
        int byteIndex = (idx * counter_w) / 8;
        int bitOffset = (idx * counter_w) % 8;
        uint32_t mask = (1 << counter_w) - 1;
        uint32_t value = (counters[byteIndex] >> bitOffset) & mask;

        if (value < mask) { // Ensure the counter does not overflow
            value++;
            counters[byteIndex] &= ~(mask << bitOffset); // Clear the bits where the counter resides
            counters[byteIndex] |= value << bitOffset; // Set the new value
        }
    }
    
    void increment(int idx, int sign) {
        int byteIndex = (idx * counter_w) / 8;
        int bitOffset = (idx * counter_w) % 8;
        uint32_t mask = (1 << counter_w) - 1;
        uint32_t value = (counters[byteIndex] >> bitOffset) & mask;
        
        value += sign;
        
        counters[byteIndex] &= ~(mask << bitOffset); // Clear the bits where the counter resides
        counters[byteIndex] |= value << bitOffset; // Set the new value
    }
};


class TowerSketch
{
public:
    Counters *line;
    BOBHash32 *hash;
    BOBHash32 *hash_sign;
    Type type;
    int idx[3];
    int mem;
    int maximum;
    uint8_t mask1 = 3;
    uint16_t mask2 = 15;
    uint16_t mask3 = 255;
    uint32_t threshold;
    TowerSketch(int w_d, Type _type = CM, uint32_t T = ELE_THRESHOLD, int _init = INIT)
    {
        mem = w_d * 4;
        threshold = T;
        line = new Counters[3]{{mem, 2}, {mem / 2, 4}, {mem / 4, 8}};
        type = _type;
        if(type == Count){
            maximum = (1 << (line[2].counter_w - 1)) - 1;
        }
        else{
            maximum =  (1 << (line[2].counter_w)) - 1;
        }
        hash = new BOBHash32[3];
        hash_sign = new BOBHash32[3];
        hash[0].initialize(_init);
        hash[1].initialize(_init + 1);
        hash[2].initialize(_init + 2);
        hash_sign[0].initialize(_init + 3);
        hash_sign[1].initialize(_init + 4);
        hash_sign[2].initialize(_init + 5);
        mask2 = T;
        
        cardPrintTowerInfo(mem, mask1, mem/2, mask2, mem/4, mask3, T, _type, maximum);
    }
    ~TowerSketch()
    {
        delete[] line;
    }
    void clear()
    {
        delete[] line;
    }
    void add_val(int line_index, int index, int val){
        line[line_index].increment(index, val);
    }
    bool insert(const char *key, int f = 1)
    {
        if (type == CM)
            return insertCM(key, f);
        else if (type == CU)
            return insertCU(key, f);
        else if (type == Count)
            return insertCount(key, f);
        else
            return inserthalf_CU(key, f);
    }
    bool insertCount(const char *key, int f = 1) {
        uint32_t checking_id = 2298415734;
        bool printout = 0;
        if(checking_id == *(uint32_t*)key)
            printout = 1;
        if(printout)
            std::cout << "------------------Enter tower's insertCount with key = " << *(uint32_t*)key << "------------------" << std::endl;
        bool flag = false;
        idx[0] = hash[0].run(key, 4) % line[0].width;
        idx[1] = hash[1].run(key, 4) % line[1].width;
        int sign0 = (hash_sign[0].run(key, 4) & 1) == 1 ? 1 : -1;
        int sign1 = (hash_sign[1].run(key, 4) & 1) == 1 ? 1 : -1;
        int32_t val0 = line[0].index(idx[0]);
        int32_t val1 = line[1].index(idx[1]);
        if(printout)
            std::cout << "idx[0]: " << idx[0] << " idx[1]: " << idx[1] << " val0: " << val0 << " val1: " << val1 << " sign0: " << sign0 << " sign1: " << sign1 << std::endl;

        // 判断第一个计数器的值
        int32_t mask0 = (1 << line[0].counter_w) - 1;
        int32_t signBit0 = 1 << (line[0].counter_w - 1);
        bool isNegative0 = ((val0 & signBit0) != 0);
        bool sameSign0 = (sign0 == 1 && !isNegative0) || (sign0 == -1 && isNegative0);
        if (sameSign0) {
            if (sign0 == 1) {
                if ((val0 & (mask0 >> 1)) != (mask0 >> 1)) {
                    flag = true;
                    if(printout){
                        std::cout << "Have compared " << (val0 & (mask0 >> 1)) << " and " << (mask0 >> 1) << std::endl;
                        std::cout << "Same sign and Incrementing counter 0 with 1" << std::endl;
                    }
                    line[0].increment(idx[0], sign0);
                }
            } else {
                if (val0 != signBit0) {
                    flag = true;
                    if(printout){
                        std::cout << "Have compared " << val0 << " and " << signBit0 << std::endl;
                        std::cout << "Same sign and Incrementing counter 0 with -1" << std::endl;
                    }
                    line[0].increment(idx[0], sign0);
                }
            }
        } else {
            if(printout)
                std::cout << "Incrementing counter 0 with " << sign0 << std::endl;
            line[0].increment(idx[0], sign0);
        }


        // 判断第二个计数器的值
        int32_t mask1 = (1 << line[1].counter_w) - 1;
        int32_t signBit1 = 1 << (line[1].counter_w - 1);
        bool isNegative1 = (val1 & signBit1) != 0;
        bool sameSign1 = (sign1 == 1 && !isNegative1) || (sign1 == -1 && isNegative1);
        if (sameSign1) {
            if (sign1 == 1) {
                if ((val1 & (mask1 >> 1)) != (mask1 >> 1)) {
                    flag = true;
                    if(printout){
                        std::cout << "Have compared " << (val1 & (mask1 >> 1)) << " and " << (mask1 >> 1) << std::endl;
                        std::cout << "Same sign and Incrementing counter 1 with 1" << std::endl;
                    }
                    line[1].increment(idx[1], sign1);
                }
            } else {
                if (val1 != signBit1) {
                    flag = true;
                    if(printout){
                        std::cout << "Have compared " << val1 << " and " << signBit1 << std::endl;
                        std::cout << "Same sign and Incrementing counter 1 with -1" << std::endl;
                    }
                    line[1].increment(idx[1], sign1);
                }
            }
        } else {
            if(printout)
                std::cout << "Incrementing counter 1 with " << sign1 << std::endl;
            line[1].increment(idx[1], sign1);
        }
        int32_t val0_a = line[0].index(idx[0]);
        int32_t val1_a = line[1].index(idx[1]);

        // 判断是否小于阈值
        if(printout){
            std::cout << "After inserting: val0: " << val0_a << " val1: " << val1_a << std::endl;
            std::cout << "------------------Leave tower's insertCount with flag = " << flag << "------------------" << std::endl;
        }
        

        return flag;
    }
    bool insertCM(const char *key, int f = 1)
    {
        
        bool flag = false;
        idx[0] = hash[0].run(key, 4) % line[0].width;
        idx[1] = hash[1].run(key, 4) % line[1].width;
        idx[2] = hash[2].run(key, 4) % line[2].width;
        uint32_t val0 = line[0].index(idx[0]);
        uint32_t val1 = line[1].index(idx[1]);
        uint32_t val2 = line[2].index(idx[2]);
        if (val0 < mask1){ //threshold?
            flag = true;
        }
        if (val1 < mask2){ //threshold?
            flag = true;
        }
        if (val2 < mask3){ //threshold?
            flag = true;
        }
        if (val0 != mask1)
            line[0].increment(idx[0]);
        if (val1 != mask2)
            line[1].increment(idx[1]);
        if (val2 != mask3)
            line[2].increment(idx[2]);
        return flag;
    }
    bool insertCU(const char *key, int f = 1)
    {
        bool flag = false;
        idx[0] = hash[0].run(key, 4) % line[0].width;
        idx[1] = hash[1].run(key, 4) % line[1].width;
        idx[2] = hash[2].run(key, 4) % line[2].width;
        uint32_t val0 = line[0].index(idx[0]);
        uint32_t val1 = line[1].index(idx[1]);
        uint32_t val2 = line[2].index(idx[2]);
        if (val0 < threshold)
            flag = true;
        if (val0 == mask1)
            val0 = UINT32_MAX;
        if (val1 < threshold)
        {
            flag = true;
        }
        if (val1 == mask2)
            val1 = UINT32_MAX;
        if (val2 < threshold)
        {
            flag = true;
        }
        if (val2 == mask3)
            val2 = UINT32_MAX;
        uint32_t min_val = std::min(val0, std::min(val1, val2));
        if (min_val == UINT32_MAX)
            return false;
        if (val0 == min_val)
        {
            line[0].increment(idx[0]);
        }
        if (val1 == min_val)
        {
            line[1].increment(idx[1]);
        }
        if (val2 == min_val)
        {
            line[2].increment(idx[2]);
        }

        return flag;
    }
    bool inserthalf_CU(const char *key, int f = 1)
    {
        bool flag = false;
        idx[0] = hash[0].run(key, 4) % line[0].width;
        idx[1] = hash[1].run(key, 4) % line[1].width;
        idx[2] = hash[2].run(key, 4) % line[2].width;
        uint32_t val0 = line[0].index(idx[0]);
        uint32_t val1 = line[1].index(idx[1]);
        uint32_t val2 = line[2].index(idx[2]);
        if (val0 != mask1)
        {
            flag = true;
            line[0].increment(idx[0]);
        }
        else
            val0 = UINT32_MAX;
        if (val1 <= val0)
        {
            if (val1 < threshold)
                line[1].increment(idx[1]);
        }
        if (val1 < threshold)
            flag = true;
        if (val2 < threshold)
        {
            flag = true;
            line[2].increment(idx[2]);
        }
        return flag;
    }
    int query(const char *key)
    {
        if(type == Count){
            idx[0] = hash[0].run(key, 4) % line[0].width;
            idx[1] = hash[1].run(key, 4) % line[1].width;
            idx[2] = hash[2].run(key, 4) % line[2].width;
            int32_t val0 = line[0].index(idx[0]);
            int32_t val1 = line[1].index(idx[1]);
            int32_t val2 = line[2].index(idx[2]);

            // 判断第一个计数器的值
            int32_t mask0 = (1 << line[0].counter_w) - 1;
            int32_t signBit0 = 1 << (line[0].counter_w - 1);
            bool isNegative0 = (val0 & signBit0) != 0;

            if (val0 == signBit0 - 1 || val0 == signBit0)
                val0 = 1 << 30;
            int ret = min(abs(val0), min(abs(val1), abs(val2)));
            return ret;
        }
        else{
            idx[0] = hash[0].run(key, 4) % line[0].width;
            idx[1] = hash[1].run(key, 4) % line[1].width;
            idx[2] = hash[2].run(key, 4) % line[2].width;
            uint32_t val0 = line[0].index(idx[0]);
            uint32_t val1 = line[1].index(idx[1]);
            uint32_t val2 = line[2].index(idx[2]);
            if (val0 == mask1)
                val0 = 1 << 30;
            if (val1 == mask2)
                val1 = 1 << 30;
            int ret = min(val0, min(val1, val2));
            return ret;
        }
    }
    bool query_if_overflow(const char *key)
    {
        idx[0] = hash[0].run(key, 4) % line[0].width;
        idx[1] = hash[1].run(key, 4) % line[1].width;
        idx[2] = hash[2].run(key, 4) % line[2].width;
        uint32_t val0 = line[0].index(idx[0]);
        uint32_t val1 = line[1].index(idx[1]);
        uint32_t val2 = line[2].index(idx[2]);
        if (val0 >= mask1 && val1 >= mask2 && val2 >= mask3)
            return true;
        return false;
    }
    uint32_t query8bit(const char *key)
    {
        idx[2] = hash[2].run(key, 4) % line[2].width;
        return line[2].index(idx[2]);
    }
    int get_cardinality()
    {
        int empty = 0;
        for (int i = 0; i < line[2].width; i++)
        {
            if (!line[2].index(i))
            {
                empty++;
            }
        }
        return (int)((double)line[2].width * log((double)line[2].width / (double)empty));
    }
    void get_entropy(int & tot, double & entr){
        int mice_dist[256] = {0};
        for(int i = 0; i < line[2].width ; i++){
            mice_dist[line[2].index(i)]++;
        }
        for(int i= 1;i<256;i++){
            tot += mice_dist[i] * i;
            entr += mice_dist[i] * i * log2(i);
        }
    }

    void printCountersToCSV(const std::string& filePath = "outputs/tower_counters.csv") {
        std::ofstream file(filePath);
        if (!file.is_open()) {
            std::cerr << "Failed to open file: " << filePath << std::endl;
            return;
        }

        // Write CSV headers
        file << "CounterSet,CounterIndex,CounterValue\n";

        // Iterate over first set of counters
        for (int i = 0; i < line[0].width; ++i) {
            uint32_t value = line[0].index(i);
            file << "0," << i << "," << value << "\n"; // 0 indicates the first set of counters
        }

        // Iterate over second set of counters
        for (int i = 0; i < line[1].width; ++i) {
            uint32_t value = line[1].index(i);
            file << "1," << i << "," << value << "\n"; // 1 indicates the second set of counters
        }

        // Iterate over third set of counters
        for (int i = 0; i < line[2].width; ++i) {
            uint32_t value = line[2].index(i);
            file << "2," << i << "," << value << "\n"; // 2 indicates the third set of counters
        }

        file.close();
        std::cout << "Counters printed to CSV file: " << filePath << std::endl;
    }
};
