#ifndef EVICTION_H
#define EVICTION_H

#include "../../common/BOBHash32.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <stdexcept>

using std::string;
using std::vector;

constexpr unsigned char MAX_TWO_BIT_VALUE = 3; // 2^2 - 1
constexpr int BITS_PER_ELEMENT = 2;
constexpr int ELEMENTS_PER_CHAR = 8 / BITS_PER_ELEMENT; // 每个unsigned char能存储的2-bit元素数量

class TwoBitArray {
private:
    std::vector<unsigned char> storage;
    size_t total_elements;

public:
    // 构造函数，指定总共需要存储的2-bit元素数量
    explicit TwoBitArray(size_t num_two_bit_elements = 0) : total_elements(num_two_bit_elements) {
        if (num_two_bit_elements == 0) {
            return;
        }
        size_t num_chars = static_cast<size_t>(
            std::ceil(static_cast<double>(num_two_bit_elements) / ELEMENTS_PER_CHAR)
        );
        storage.assign(num_chars, 0); // 初始化所有字节为0
    }

    // 内部辅助类，用于实现 arr[index] = value 语法
    class ElementProxy {
    private:
        TwoBitArray& array_ref;
        size_t element_index;
    
    public:
        ElementProxy(TwoBitArray& arr, size_t idx) : array_ref(arr), element_index(idx) {}

        ElementProxy& operator=(unsigned char value) {
            array_ref.setElement(element_index, value);
            return *this;
        }

        operator unsigned char() const {
            return array_ref.getElement(element_index);
        }
    };

    ElementProxy operator[](size_t index) {
        if (index >= total_elements) {
            throw std::out_of_range("Array index out of bounds.");
        }
        return ElementProxy(*this, index);
    }

    unsigned int operator[](size_t index) const {
        if (index >= total_elements) {
            throw std::out_of_range("Array index out of bounds.");
        }
        return static_cast<unsigned int>(getElement(index));
    }

    void setElement(size_t element_index, unsigned char value) {
        if (value > MAX_TWO_BIT_VALUE) {
            throw std::invalid_argument("Value for a 2-bit unit must be between 0 and 3.");
        }

        size_t char_index = element_index / ELEMENTS_PER_CHAR;
        int sub_index_in_char = element_index % ELEMENTS_PER_CHAR; // 0, 1, 2, or 3

        int offset = sub_index_in_char * BITS_PER_ELEMENT; // 0, 2, 4, or 6
        unsigned char clear_mask = ~(MAX_TWO_BIT_VALUE << offset);

        storage[char_index] &= clear_mask;
        storage[char_index] |= (value << offset);
    }

    unsigned char getElement(size_t element_index) const {
        size_t char_index = element_index / ELEMENTS_PER_CHAR;
        int sub_index_in_char = element_index % ELEMENTS_PER_CHAR;

        int offset = sub_index_in_char * BITS_PER_ELEMENT;
        return (storage[char_index] >> offset) & MAX_TWO_BIT_VALUE;
    }

    size_t size() const {
        return total_elements;
    }

    void printInternalStorage() const {
        std::cout << "Internal storage (bytes):" << std::endl;
        for (size_t i = 0; i < storage.size(); ++i) {
            std::cout << "Byte " << i << ": ";
            for (int bit = 7; bit >= 0; --bit) {
                std::cout << ((storage[i] >> bit) & 1);
                if (bit > 0 && bit % BITS_PER_ELEMENT == 0 && bit < 7) {
                    std::cout << " ";
                }
            }
            std::cout << std::endl;
        }
    }
};

template <uint32_t key_length, uint32_t memory_in_bytes> 
class Eviction {
private:
    static constexpr uint32_t z = 2;
    static constexpr uint32_t w = memory_in_bytes / z * 4;
    static constexpr uint32_t W = z * w;

    TwoBitArray counters[W];
    uint32_t g[4]; // 0, 1, 2, >2
    double theta[4];
    uint32_t total;
    BOBHash32 *bob_hash;

public:
    string name;

    Eviction() : total(0) {
        srand(time(0));
        bob_hash = new BOBHash32(rand() % MAX_PRIME32);
        
        // 初始化计数器数组
        for (uint32_t i = 0; i < W; ++i) {
            counters[i] = TwoBitArray(1);
        }
        
        g[0] = W;
        g[1] = g[2] = g[3] = 0;

        std::stringstream name_buffer;
        name_buffer << "Eviction@" << memory_in_bytes;
        name = name_buffer.str();
    }

    ~Eviction() {
        delete bob_hash;
    }

    void insert(uint8_t *item) {
        uint32_t pos = bob_hash->run(reinterpret_cast<const char*>(item), key_length) % W;
        
        // 驱逐逻辑
        if (pos < w && counters[pos][0] == 0) {
            g[counters[pos + w][0]]--;
        }
        
        if (pos < w || counters[pos - w][0] == 0) {
            if (counters[pos][0] < 3) {
                g[counters[pos][0]]--;
                counters[pos][0] = counters[pos][0] + 1;
                g[counters[pos][0]]++;
            }
        }
    }

    double h(uint32_t j, uint32_t w_param = 1, uint32_t m = 0) {
        cout << "j: " << j << ", w_param: " << w_param << ", m: " << m << endl;
        if (j == 0 && m == 0) {
            return std::exp(-1.0);
        }
        if (w_param > j || w_param * m > j) {
            return 0.0;
        }
        
        double res = 0.0;
        double frac = 1.0;
        
        for (int r = 0; r <= static_cast<int>(m) && r <= static_cast<int>(j / w_param); ++r) {
            if (r > 0) {
                frac *= r;
            }
            double tmp = std::pow(theta[w_param], r) / frac * h(j - r * w_param, w_param + 1, m - r);
            res += tmp;
        }
        return res;
    }

    int query(uint8_t *item) {
        uint32_t pos = bob_hash->run(reinterpret_cast<const char*>(item), key_length) % W;
        return counters[pos][0];
    }

    void get_distribution(vector<double> &dist_est) {
        double L = -std::log(1.0 * g[0] / (g[0] + g[1] + g[2] + g[3]));
        printf("L: %lf\n", L);
        
        theta[1] = 1.0 * g[1] / g[0] / L;
        theta[2] = (1.0 * g[2] / g[0]) / L - 1.0 * L * h(2, 1, 2);
        theta[3] = (1.0 * g[3] / g[0]) / L - 1.0 * L * h(3, 1, 2) - 1.0 * L * L * h(3, 1, 3);
        
        uint32_t len = static_cast<uint32_t>(dist_est.size());
        double sum = 0.0;
        
        for (uint32_t i = 1; i < len; ++i) {
            sum += dist_est[i];
        }
        
        double theta_sum = 1;
        if (theta_sum > 0) {
            double p1 = theta[1] / theta_sum;
            double p2 = theta[2] / theta_sum;
            
            dist_est[1] = p1 * sum;
            dist_est[2] = p2 * sum;
        }
    }
    void output() {
        std::cout << "Eviction:" << std::endl;
        std::cout << "W: " << W << std::endl;
        std::cout << "w: " << w << std::endl;
        std::cout << "z: " << z << std::endl;
        std::cout << "g[0]: " << g[0] << std::endl;
        std::cout << "g[1]: " << g[1] << std::endl;
        std::cout << "g[2]: " << g[2] << std::endl;
        std::cout << "g[3]: " << g[3] << std::endl;
        std::cout << "theta[1]: " << theta[1] << std::endl;
        std::cout << "theta[2]: " << theta[2] << std::endl;
        std::cout << "theta[3]: " << theta[3] << std::endl;
        std::cout << "total: " << total << std::endl;
    }
};

#endif // STREAMMEASUREMENTSYSTEM_MRAC_H