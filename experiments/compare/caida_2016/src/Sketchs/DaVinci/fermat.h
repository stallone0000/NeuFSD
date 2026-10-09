#ifndef _FERMAT_H_
#define _FERMAT_H_

#include <iostream>
#include <cstdint>
#include <unordered_map>
#include <queue>
#include <cstring>
// #include "util/BOBHash32.h"
#include "../common/BOBHash32.h"
#include "util/mod.h"
#include "util/prime.h"
#include "util/murmur3.h"
#include <vector>
#include <algorithm>
#include <variant>
#include <cassert>
using namespace std;

#define HASH_TO_SIGN(hash_value) (((hash_value) & 1) ? -1 : 1)
#define DEBUG_F 0

// fingprint no used

// use a 16-bit prime, so 2 * a mod PRIME will not overflow
static const uint32_t PRIME_ID = MAXPRIME[32]; //PRIME for Fermat_sketch
static const uint32_t PRIME_ID_IDP_CNTPM = MAXPRIME[32]; //PRIME for Fermat_sketch
static const uint32_t PRIME_ID_COUNT = MAXPRIME[31]; //PRIME for Fermat_count
static const uint32_t PRIME_FING = MAXPRIME[32];

inline uint64_t checkTable(uint64_t pos)
{
    return powMod32(pos, PRIME_ID - 2, PRIME_ID);
}

inline uint64_t checkTable_count(uint64_t pos, uint32_t prime = PRIME_ID_IDP_CNTPM)
{
    return powMod32(pos, prime - 2, prime);
}


using DataVariant = std::variant<std::unordered_map<int, int>, std::unordered_map<unsigned int, int>>;

class Fermat
{
    // bool use_fing;


public:

    int pure_cnt;
    unordered_map<int32_t, int> insertedflows;

    

    virtual void clear_look_up_table() = 0;
    virtual void create_array() = 0;
    virtual void clear_array() = 0;

    virtual void Insert(uint32_t flow_id, uint32_t cnt) = 0;
    // virtual void Insert() = 0;
    virtual void Insert_one(uint32_t flow_id) = 0;

    virtual void Delete_in_one_bucket(int row, int col, int pure_row, int pure_col, int sign = 1) = 0;

    // virtual bool verify(int row, int col, uint32_t &flow_id, uint32_t &fing) = 0;
    virtual int verify(int row, int col, uint32_t &flow_id, uint32_t &fing) = 0;
    virtual int united_verify(int row, int col, uint32_t &flow_id, uint32_t &fing, TowerSketch* tower) {
        cout << "united_verify() is not implemented in root class!" << endl;
        return -1;
    }

    virtual void display() = 0;
    virtual int query(const char *key) = 0;
    virtual int undecoded_query(const char *key) = 0;
    // virtual bool Decode(unordered_map<uint32_t, int> &result);
    // virtual bool Decode(unordered_map<int32_t, int> &result);
    virtual bool Decode(DataVariant& data) = 0;
    virtual bool united_decode(DataVariant& data, TowerSketch* tower){
        cout << "Decode() is not implemented in root class!" << endl;
        return false;
    }
    virtual int get_id(int n_array, int n) = 0;
    virtual int get_counter(int n_array, int n) = 0;

    virtual int set_id(int n_array, int n, int value) = 0;
    virtual int set_counter(int n_array, int n, int value) = 0;

    virtual int get_array_num() = 0;
    virtual int get_entry_num() = 0;

    virtual int query_from_cpy(const char *key){
        cout << "query_from_cpy() is not implemented in root class!" << endl;
        return -1;
    }
    virtual int query_after_decoding(const char *key){
        cout << "query_after_decoding() is not implemented in root class!" << endl;
        return -1;
    }
    virtual int undecoded_query_before_decoding(const char *key){
        cout << "undecoded_query_before_decoding() is not implemented in root class!" << endl;
        return -1;
    }

    bool cpy_counters(){ 
        cout << "cpy_counters() is not implemented!" << endl;
        return false;
    }

    virtual bool cpy_counters_to_pos(int32_t ***countercpy){
        cout << "cpy_counters_to() is not implemented!" << endl;
        return false;
    }
    
    virtual int query_array(const char *key, int array_index){
        cout << "query_array() is not implemented in root class!" << endl;
        return -1;
    }

    virtual ~Fermat() {};
};
class Fermat_Sketch : public Fermat
{
    int array_num;
    int entry_num;
    int decodeflag = 0;
    // hash
    BOBHash32 *hash;
    BOBHash32 *hash_fp;

    uint32_t *table;

    bool use_fing;
    uint32_t **id;
    uint32_t **counter;
    uint32_t **fingerprint;
    uint32_t **idcpy, **fingcpy, **countercpy;

public:


    void clear_look_up_table() override
    {
        delete[] table;
    }

    void create_array() override
    {
        pure_cnt = 0;
        // id
        id = new uint32_t *[array_num];
        for (int i = 0; i < array_num; ++i)
        {
            id[i] = new uint32_t[entry_num];
            memset(id[i], 0, entry_num * sizeof(uint32_t));
        }
        // fingerprint
        if (use_fing)
        {
            fingerprint = new uint32_t *[array_num];
            for (int i = 0; i < array_num; ++i)
            {
                fingerprint[i] = new uint32_t[entry_num];
                memset(fingerprint[i], 0, entry_num * sizeof(uint32_t));
            }
        }

        // counter
        counter = new uint32_t *[array_num];
        for (int i = 0; i < array_num; ++i)
        {
            counter[i] = new uint32_t[entry_num];
            memset(counter[i], 0, entry_num * sizeof(uint32_t));
        }
    }

    void clear_array() override
    {
        for (int i = 0; i < array_num; ++i)
            delete[] id[i];
        delete[] id;

        if (fingerprint)
        {
            for (int i = 0; i < array_num; ++i)
                delete[] fingerprint[i];
            delete[] fingerprint;
        }

        for (int i = 0; i < array_num; ++i)
            delete[] counter[i];
        delete[] counter;
    }

    Fermat_Sketch(int _a, int _e, bool _fing, uint32_t _init) : array_num(_a), entry_num(_e), use_fing(_fing), fingerprint(nullptr), hash_fp(nullptr)
    {
        cout << "You are running Fermat Sketch version. Prime for ID: " << PRIME_ID <<endl;
        create_array();
        // hash
        if (use_fing)
            hash_fp = new BOBHash32(_init);
        hash = new BOBHash32[array_num];
        for (int i = 0; i < array_num; ++i)
        {
            hash[i].initialize(_init + (10 * i) + 1);
        }
    }

    Fermat_Sketch(int _memory, bool _fing, uint32_t _init) : use_fing(_fing), fingerprint(nullptr), hash_fp(nullptr)
    {
        printf("You are running Fermat Sketch version.\n");
        array_num = 3;
        if (use_fing)
            entry_num = _memory / (array_num * 12);
        else
            entry_num = _memory / (array_num * 8);
        create_array();

        // hash
        if (use_fing)
            hash_fp = new BOBHash32(_init);
        hash = new BOBHash32[array_num];
        for (int i = 0; i < array_num; ++i)
            hash[i].initialize(_init + i + 1);
    }

    ~Fermat_Sketch() override
    {
        clear_array();
        clear_look_up_table();
        if (hash_fp)
            delete hash_fp;
        delete[] hash;
    }

    void Insert(uint32_t flow_id, uint32_t cnt) override
    {
        insertedflows[flow_id]+=cnt;
        if (use_fing)
        {
            uint32_t fing = hash_fp->run((char *)&flow_id, sizeof(uint32_t));
            for (int i = 0; i < array_num; ++i)
            {
                uint32_t pos = hash[i].run((char *)&flow_id, sizeof(uint32_t)) % entry_num;
                id[i][pos] = (mulMod32(flow_id, cnt, PRIME_ID) + (uint64_t)id[i][pos]) % PRIME_ID;
                fingerprint[i][pos] = ((uint64_t)fingerprint[i][pos] + mulMod32(fing, cnt, PRIME_FING)) % PRIME_FING;
                counter[i][pos] += cnt;
            }
        }
        else
        {
            for (int i = 0; i < array_num; ++i)
            {
                uint32_t pos = hash[i].run((char *)&flow_id, sizeof(uint32_t)) % entry_num;
                id[i][pos] = (mulMod32(flow_id, cnt, PRIME_ID) + (uint64_t)id[i][pos]) % PRIME_ID;
                counter[i][pos] += cnt;
            }
        }
    }

    void Insert_one(uint32_t flow_id) override
    {
        // flow_id should < PRIME_ID
        if (use_fing)
        {
            uint32_t fing = hash_fp->run((char *)&flow_id, sizeof(uint32_t)) % PRIME_FING;
            for (int i = 0; i < array_num; ++i)
            {
                uint32_t pos = hash[i].run((char *)&flow_id, sizeof(uint32_t)) % entry_num;
                id[i][pos] = ((uint64_t)id[i][pos] + (uint64_t)(flow_id % PRIME_ID)) % PRIME_ID;
                fingerprint[i][pos] = ((uint64_t)fingerprint[i][pos] + (uint64_t)fing) % PRIME_FING;
                counter[i][pos]++;
            }
        }
        else
        {
            for (int i = 0; i < array_num; ++i)
            {
                uint32_t pos = hash[i].run((char *)&flow_id, sizeof(uint32_t)) % entry_num;
                id[i][pos] = ((uint64_t)id[i][pos] + (uint64_t)(flow_id % PRIME_ID)) % PRIME_ID;
                counter[i][pos]++;
            }
        }
    }

    void Delete_in_one_bucket(int row, int col, int pure_row, int pure_col, int sign = 1) override
    {
        id[row][col] = ((uint64_t)PRIME_ID + (uint64_t)id[row][col] - (uint64_t)id[pure_row][pure_col]) % PRIME_ID;
        if (use_fing)
            fingerprint[row][col] = ((uint64_t)PRIME_FING + (uint64_t)fingerprint[row][col] - (uint64_t)fingerprint[pure_row][pure_col]) % PRIME_FING;
        counter[row][col] -= counter[pure_row][pure_col];
        
    }

    int verify(int row, int col, uint32_t &flow_id, uint32_t &fing) override
    {
#if DEBUG_F
        ++pure_cnt;
#endif
        if (counter[row][col] & 0x80000000)
        {
            uint64_t temp = checkTable(~counter[row][col] + 1);
            flow_id = mulMod32(PRIME_ID - id[row][col], temp, PRIME_ID);
        }
        else
        {
            uint64_t temp = checkTable(counter[row][col]);
            flow_id = mulMod32(id[row][col], temp, PRIME_ID);
        }
        if (use_fing)
        {
            fing = powMod32(counter[row][col], PRIME_FING - 2, PRIME_FING);
            fing = mulMod32(fingerprint[row][col], fing, PRIME_FING);
        }
        if (!(hash[row].run((char *)&flow_id, sizeof(int32_t)) % entry_num == col))
            return false;
        if (use_fing && !(hash_fp->run((char *)&flow_id, sizeof(uint32_t)) % PRIME_FING == fing))
            return false;
        return true;
    }

    void display() override
    {
        cout << " --- display --- " << endl;
        for (int i = 0; i < array_num; ++i)
        {
            for (int j = 0; j < entry_num; ++j)
            {
                if (counter[i][j])
                {
                    cout << i << "," << j << ":" << counter[i][j] << endl;
                }
            }
        }
    }
    int query(const char *key) override
    {
        uint32_t flow_id = *(uint32_t *)key;
        uint32_t ret = 1 << 30;
        if (decodeflag)
        {
            for (int i = 0; i < array_num; ++i)
            {
                uint32_t pos = hash[i].run((char *)&flow_id, sizeof(uint32_t)) % entry_num;
                ret = min(counter[i][pos], ret);
            }
        }
        return (int)ret;
    }
    int undecoded_query(const char *key) override
    {
        uint32_t flow_id = *(uint32_t *)key;
        uint32_t ret = 1 << 30;
        
        for (int i = 0; i < array_num; ++i)
        {
            uint32_t pos = hash[i].run((char *)&flow_id, sizeof(uint32_t)) % entry_num;
            ret = min(counter[i][pos], ret);
        }
        
        return (int)ret;
    }
    int undecoded_query_before_decoding(const char *key) override {
        cout << "Undecoded query of fermat_sketch is not implemented!" << endl;
        return -1;
    }
    bool Decode(DataVariant& data) override
    {
        auto* mapPtr = std::get_if<std::unordered_map<unsigned int, int>>(&data);
        if (!mapPtr) {
            return false;  // return false if type doesn't match
        }
        
        auto& result = *mapPtr;
        idcpy = new uint32_t *[array_num];
        for (int i = 0; i < array_num; i++)
        {
            idcpy[i] = new uint32_t[entry_num];
            for (int j = 0; j < entry_num; j++)
                idcpy[i][j] = id[i][j];
        }
        if (use_fing)
        {
            fingcpy = new uint32_t *[array_num];
            for (int i = 0; i < array_num; i++)
            {
                fingcpy[i] = new uint32_t[entry_num];
                for (int j = 0; j < entry_num; j++)
                    fingcpy[i][j] = fingerprint[i][j];
            }
        }
        countercpy = new uint32_t *[array_num];
        for (int i = 0; i < array_num; i++)
        {
            countercpy[i] = new uint32_t[entry_num];
            for (int j = 0; j < entry_num; j++)
                countercpy[i][j] = counter[i][j];
        }
        decodeflag = 1;
        queue<int> *candidate = new queue<int>[array_num];
        uint32_t flow_id = 0;
        uint32_t fing = 0;

        // first round
        for (int i = 0; i < array_num; ++i)
            for (int j = 0; j < entry_num; ++j)
            {
                if (counter[i][j] == 0)
                {
                    continue;
                }
                else if (verify(i, j, flow_id, fing))
                {
                    // find pure bucket
                    if (result.count(flow_id) != 0)
                    {
                        result[flow_id] += counter[i][j];
                    }
                    else
                    {
                        result[flow_id] = counter[i][j];
                    }
                    // delete flow from other rows
                    for (int t = 0; t < array_num; ++t)
                    {
                        if (t == i)
                            continue;
                        uint32_t pos = hash[t].run((char *)&flow_id, sizeof(uint32_t)) % entry_num;
                        Delete_in_one_bucket(t, pos, i, j);
                        candidate[t].push(pos);
                    }
                    Delete_in_one_bucket(i, j, i, j);
                }
            }

        bool pause;
        int acc = 0;
        while (true)
        {
            acc++;
            pause = true;
            for (int i = 0; i < array_num; ++i)
            {
                if (!candidate[i].empty())
                    pause = false;
                while (!candidate[i].empty())
                {
                    int check = candidate[i].front();
                    candidate[i].pop();
                    if (counter[i][check] == 0)
                    {
                        continue;
                    }
                    else if (verify(i, check, flow_id, fing))
                    {
                        // find pure bucket
                        if (result.count(flow_id) != 0)
                        {
                            result[flow_id] += counter[i][check];
                        }
                        else
                        {
                            result[flow_id] = counter[i][check];
                        }
                        // delete flow from other rows
                        for (int t = 0; t < array_num; ++t)
                        {
                            if (t == i)
                                continue;
                            uint32_t pos = hash[t].run((char *)&flow_id, sizeof(uint32_t)) % entry_num;
                            Delete_in_one_bucket(t, pos, i, check);
                            candidate[t].push(pos);
                        }
                        Delete_in_one_bucket(i, check, i, check);
                    }
                }
            }
            if (pause){
                printf("Break because pause!\n");
                break;

            }
            if (acc > 10000)
                printf("Break because acc is too big!\n");
                break;
        }
        printf("Get out of while in decode in fermat.h\n");

        delete[] candidate;
        bool flag = true;
        for (int i = 0; i < array_num; ++i)
            for (int j = 0; j < entry_num; ++j)
                if (counter[i][j] != 0)
                {
                    flag = false;
                }
        for (auto p : result)
        {
            if (p.second == 0)
            {
                result.erase(p.first);
            }
        }
        return flag;
    }
    int get_id(int n_array, int n){
        if(n_array >=0 && n_array <= array_num && n >= 0 && n <= entry_num){
            return id[n_array][n];
        }
        else{
            cout << "get_id() out of range!" << endl;
            assert(0);
        }
    }
    int get_counter(int n_array, int n){
        if(n_array >=0 && n_array <= array_num && n >= 0 && n <= entry_num){
            return counter[n_array][n];
        }
        else{
            cout << "get_counter() out of range!" << endl;
            assert(0);
        }
    }
    int set_id(int n_array, int n, int value){
        if(n_array >=0 && n_array <= array_num && n >= 0 && n <= entry_num){
            id[n_array][n] = value;
            return 0;
        }
        else{
            cout << "set_id() out of range! " << n_array << "/" << array_num << " " << n << "/" << entry_num << endl;
            assert(0);
        }
    }
    int set_counter(int n_array, int n, int value){
        if(n_array >=0 && n_array <= array_num && n >= 0 && n <= entry_num){
            counter[n_array][n] = value;
            return 0;
        }
        else{
            cout << "set_counter() out of range!" << endl;
            assert(0);
        }
    }
    int get_array_num() override{
        return array_num;
    }
    int get_entry_num() override{
        return entry_num;
    }
};

class Fermat_Count : public Fermat
{
    int array_num;
    int entry_num;
    int decodeflag = 0;
    // hash
    BOBHash32 *hash;
    BOBHash32 *hash_fp;

    uint32_t *table;

    bool use_fing;
    // arrays
    // int array_num;
    // int entry_num;
    int32_t **id;
    int32_t **fingerprint;
    int32_t **counter;
    int32_t **idcpy, **countercpy;
    int32_t **fingcpy;
    // int decodeflag = 0;
    // // hash
    // BOBHash32 *hash;
    // BOBHash32 *hash_fp;
    BOBHash32 *hash_sign;

public:

    void clear_look_up_table() override
    {
        delete[] table;
    }

    void create_array() override
    {
        pure_cnt = 0;
        // id
        id = new int32_t *[array_num];
        for (int i = 0; i < array_num; ++i)
        {
            id[i] = new int32_t[entry_num];
            memset(id[i], 0, entry_num * sizeof(int32_t));
        }
        // fingerprint
        if (use_fing)
        {
            fingerprint = new int32_t *[array_num];
            for (int i = 0; i < array_num; ++i)
            {
                fingerprint[i] = new int32_t[entry_num];
                memset(fingerprint[i], 0, entry_num * sizeof(int32_t));
            }
        }

        // counter
        counter = new int32_t *[array_num];
        for (int i = 0; i < array_num; ++i)
        {
            counter[i] = new int32_t[entry_num];
            memset(counter[i], 0, entry_num * sizeof(int32_t));
        }
    }

    void clear_array() override
    {
        for (int i = 0; i < array_num; ++i)
            delete[] id[i];
        delete[] id;

        if (fingerprint)
        {
            for (int i = 0; i < array_num; ++i)
                delete[] fingerprint[i];
            delete[] fingerprint;
        }

        for (int i = 0; i < array_num; ++i)
            delete[] counter[i];
        delete[] counter;
    }

    Fermat_Count(int _a, int _e, bool _fing, uint32_t _init) : array_num(_a), entry_num(_e), use_fing(_fing), fingerprint(nullptr), hash_fp(nullptr)
    {
        printf("You are running Fermat Count version. Prime for ID: %d\n", PRIME_ID_COUNT);
        create_array();
        // hash
        if (use_fing)
            hash_fp = new BOBHash32(_init);
        hash = new BOBHash32[array_num];
        hash_sign = new BOBHash32[array_num];
        
        for (int i = 0; i < array_num; ++i)
        {
            hash[i].initialize(_init + (10 * i) + 1);
            hash_sign[i].initialize(_init + (17 * i) + 1);
        }
    }

    Fermat_Count(int _memory, bool _fing, uint32_t _init) : use_fing(_fing), fingerprint(nullptr), hash_fp(nullptr)
    {
        printf("You are running Fermat Count version. _memory = %d\n", _memory);
        array_num = 3;
        if (use_fing)
            entry_num = _memory / (array_num * 12);
        else
            entry_num = _memory / (array_num * 8);

        create_array();

        // hash
        if (use_fing)
            hash_fp = new BOBHash32(_init);
        hash = new BOBHash32[array_num];
        hash_sign = new BOBHash32[array_num];
        for (int i = 0; i < array_num; ++i){

            hash[i].initialize(_init + i + 1);
            hash_sign[i].initialize(_init + (17 * i) + 1);
        }
    }

    ~Fermat_Count() override
    {
        clear_array();
        clear_look_up_table();
        if (hash_fp)
            delete hash_fp;
        delete[] hash;
    }

    int get_sign(int32_t flow_id, int i){
        uint32_t pos = hash[i].run((char *)&flow_id, sizeof(uint32_t)) % entry_num;
        uint32_t kk = hash_sign[i].run((char *)&flow_id, sizeof(uint32_t));
        int sign = HASH_TO_SIGN(kk);
        return sign;
    }
    void Insert(uint32_t flow_id, uint32_t cnt) override
    {     
        int flow_id_ = (int)flow_id;
        insertedflows[flow_id_]+=cnt;
        if (use_fing)
        {
            uint32_t fing = hash_fp->run((char *)&flow_id, sizeof(uint32_t));
            for (int i = 0; i < array_num; ++i)
            {
                uint32_t pos = hash[i].run((char *)&flow_id_, sizeof(uint32_t)) % entry_num;
                uint32_t kk = hash_sign[i].run((char *)&flow_id_, sizeof(uint32_t));
                int sign = HASH_TO_SIGN(kk);
                if(sign > 0){
                    id[i][pos] = ((int64_t)id[i][pos] + (int64_t)mulMod32_Cnt(flow_id_, cnt, PRIME_ID)) % PRIME_ID;
                    fingerprint[i][pos] = ((int64_t)fingerprint[i][pos] + mulMod32_Cnt(fing, cnt, PRIME_FING)) % PRIME_FING;
                    counter[i][pos] += cnt;
                }
                else{
                    id[i][pos] = ((int64_t)id[i][pos] - (int64_t)mulMod32_Cnt(flow_id_, cnt, PRIME_ID)) % PRIME_ID;
                    fingerprint[i][pos] = ((int64_t)fingerprint[i][pos] - mulMod32_Cnt(fing, cnt, PRIME_FING)) % PRIME_FING;
                    counter[i][pos] -= cnt;
                }
            }
        }
        else
        {
            for (int i = 0; i < array_num; ++i)
            {
                uint32_t pos = hash[i].run((char *)&flow_id_, sizeof(uint32_t)) % entry_num;
                uint32_t kk = hash_sign[i].run((char *)&flow_id_, sizeof(uint32_t));
                int sign = HASH_TO_SIGN(kk);
                if(sign > 0){
                    id[i][pos] = ((int64_t)id[i][pos] + (int64_t)mulMod32_Cnt(flow_id_, cnt, PRIME_ID_COUNT)) % (int64_t)PRIME_ID_COUNT;
                    counter[i][pos] += cnt;
                }
                else{
                    id[i][pos] = ((int64_t)id[i][pos] - (int64_t)mulMod32_Cnt(flow_id_, cnt, PRIME_ID_COUNT)) % (int64_t)PRIME_ID_COUNT;
                    counter[i][pos] -= cnt;
                }
            }
        }
    }

    void Insert_one(uint32_t flow_id) override
    {
        // flow_id should < PRIME_ID
        if (use_fing)
        {
            uint32_t fing = hash_fp->run((char *)&flow_id, sizeof(uint32_t)) % PRIME_FING;
            for (int i = 0; i < array_num; ++i)
            {
                uint32_t pos = hash[i].run((char *)&flow_id, sizeof(uint32_t)) % entry_num;
                id[i][pos] = ((uint64_t)id[i][pos] + (uint64_t)(flow_id % PRIME_ID)) % PRIME_ID;
                fingerprint[i][pos] = ((int64_t)fingerprint[i][pos] + (uint64_t)fing) % PRIME_FING;
                counter[i][pos]++;
            }
        }
        else
        {
            for (int i = 0; i < array_num; ++i)
            {
                uint32_t pos = hash[i].run((char *)&flow_id, sizeof(uint32_t)) % entry_num;
                id[i][pos] = ((uint64_t)id[i][pos] + (uint64_t)(flow_id % PRIME_ID)) % PRIME_ID;
                counter[i][pos]++;
            }
        }
    }

    void Delete_in_one_bucket(int row, int col, int pure_row, int pure_col, int sign = 1) override
    {
        id[row][col] = ((int64_t)PRIME_ID_COUNT + (int64_t)id[row][col] - (int64_t)id[pure_row][pure_col]) % PRIME_ID_COUNT;
        if (use_fing)
            fingerprint[row][col] = ((int64_t)PRIME_FING + (int64_t)fingerprint[row][col] - (int64_t)fingerprint[pure_row][pure_col]) % PRIME_FING;
        counter[row][col] -= counter[pure_row][pure_col];
    }
    int verify(int row, int col, uint32_t &flow_id, uint32_t &fing)
    {
#if DEBUG_F
        ++pure_cnt;
#endif
        int32_t cnt_value = counter[row][col];
        int32_t id_value = id[row][col];
        {
            uint64_t temp = checkTable_count(abs(cnt_value));
            flow_id = mulMod32_Cnt(id_value, temp, PRIME_ID_COUNT);
        }
        if (use_fing)
        {
            fing = powMod32(cnt_value, PRIME_FING - 2, PRIME_FING);
            fing = mulMod32(fingerprint[row][col], fing, PRIME_FING);
        }
        int mapto = hash[row].run((char *)&flow_id, sizeof(int32_t)) % entry_num;
        if (!(mapto == col))
            return false;
        if (use_fing && !(hash_fp->run((char *)&flow_id, sizeof(int32_t)) % PRIME_FING == fing))
            return false;
        return true;
    }

    void display() override
    {
        cout << " --- display --- " << endl;
        for (int i = 0; i < array_num; ++i)
        {
            for (int j = 0; j < entry_num; ++j)
            {
                if (counter[i][j])
                {
                    cout << i << "," << j << ":" << counter[i][j] << endl;
                }
            }
        }
    }
    int query(const char *key) override
    {
        uint32_t flow_id = *(uint32_t *)key;
        uint32_t ret = 1 << 30;
        if (decodeflag)
        {
            for (int i = 0; i < array_num; ++i)
            {
                uint32_t pos = hash[i].run((char *)&flow_id, sizeof(uint32_t)) % entry_num;
                if(ret > counter[i][pos]) return counter[i][pos];
                else return ret;
            }
        }
        return (int)ret;
    }

    int undecoded_query(const char *key) override // Must be used after decode() is used.
    {
        uint32_t flow_id = *(uint32_t *)key;
        std::vector<int32_t> values;
        std::vector<int32_t> values_from_changed_counters;

        for (int i = 0; i < array_num; ++i)
        {
            uint32_t pos = hash[i].run((char *)&flow_id, sizeof(uint32_t)) % entry_num;
            
            uint32_t kk = hash_sign[i].run((char *)&flow_id, sizeof(uint32_t));
            int sign = HASH_TO_SIGN(kk);
            values.push_back(countercpy[i][pos]*sign);
            cout << countercpy[i][pos]*sign << " ";
        }

        cout << "The size of the values in fermat_count: " << values.size() << endl;

        // find the median value
        size_t median_index = values.size() / 2;
        std::nth_element(values.begin(), values.begin() + median_index, values.end());
        int32_t median = values[median_index];

        if (values.size() % 2 == 0) {
            int32_t next_median = *std::max_element(values.begin(), values.begin() + median_index);
            median = (median + next_median) / 2;
        }

        return (int)median;
    }
    
    bool Decode(DataVariant& data) override
    {
        auto* mapPtr = std::get_if<std::unordered_map<int32_t, int>>(&data);
        if (!mapPtr) {
            return false;  // 如果类型不匹配，则直接返回 false
        }
        
        auto& result = *mapPtr;

        cout << "The size of the converted Variant map in fermat_count: " << result.size() << endl;
        idcpy = new int32_t *[array_num];
        for (int i = 0; i < array_num; i++)
        {
            idcpy[i] = new int32_t[entry_num];
            for (int j = 0; j < entry_num; j++)
                idcpy[i][j] = id[i][j];
        }
        if (use_fing)
        {
            fingcpy = new int32_t *[array_num];
            for (int i = 0; i < array_num; i++)
            {
                fingcpy[i] = new int32_t[entry_num];
                for (int j = 0; j < entry_num; j++)
                    fingcpy[i][j] = fingerprint[i][j];
            }
        }
        countercpy = new int32_t *[array_num];
        for (int i = 0; i < array_num; i++)
        {
            countercpy[i] = new int32_t[entry_num];
            for (int j = 0; j < entry_num; j++)
                countercpy[i][j] = counter[i][j];
        }
        decodeflag = 1;
        queue<int> *candidate = new queue<int>[array_num];
        int32_t flow_id = 0;
        int32_t fing = 0;
        for (int i = 0; i < array_num; ++i){
            uint32_t kk = hash_sign[i].run((char *)&flow_id, sizeof(uint32_t));
            int sign = HASH_TO_SIGN(kk);
            for (int j = 0; j < entry_num; ++j)
            {
                uint32_t temp_flow_id = 0;
                uint32_t temp_fin = 0;
                if (counter[i][j] == 0)
                {
                    continue;
                }
                else if (verify(i, j, temp_flow_id, temp_fin))
                {
                    // find pure bucket
                    flow_id = (int32_t)temp_flow_id;
                    fing = (int32_t)temp_fin;
                    if (result.count(flow_id) != 0)
                    {
                        result[flow_id] += abs(counter[i][j]);
                    }
                    else
                    {
                        result[flow_id] = abs(counter[i][j]);
                    }
                    // delete flow from other rows
                    
                    for (int t = 0; t < array_num; ++t)
                    {
                        if (t == i)
                            continue;
                        uint32_t pos = hash[t].run((char *)&flow_id, sizeof(uint32_t)) % entry_num;
                        
                        Delete_in_one_bucket(t, pos, i, j, sign);
                        candidate[t].push(pos);
                    }
                    Delete_in_one_bucket(i, j, i, j, sign);
                }
            }
        }

        bool pause;
        int acc = 0;
        while (true)
        {
            acc++;
            pause = true;
            for (int i = 0; i < array_num; ++i)
            {
                uint32_t kk = hash_sign[i].run((char *)&flow_id, sizeof(uint32_t));
                int sign = HASH_TO_SIGN(kk);
                if (!candidate[i].empty())
                    pause = false;
                while (!candidate[i].empty())
                {
                    int check = candidate[i].front();
                    candidate[i].pop();
                    uint32_t temp_flow_id = 0;
                    uint32_t temp_fin = 0;
                    if (counter[i][check] == 0)
                    {
                        continue;
                    }
                    else if (verify(i, check, temp_flow_id, temp_fin))
                    {
                        // find pure bucket
                        flow_id = (int32_t)temp_flow_id;
                        fing = (int32_t)temp_fin;
                        if (result.count(flow_id) != 0)
                        {
                            result[flow_id] += abs(counter[i][check]);
                        }
                        else
                        {
                            result[flow_id] = abs(counter[i][check]);
                        }
                        // delete flow from other rows
                        for (int t = 0; t < array_num; ++t)
                        {
                            if (t == i)
                                continue;
                            uint32_t pos = hash[t].run((char *)&flow_id, sizeof(uint32_t)) % entry_num;
                            Delete_in_one_bucket(t, pos, i, check, sign);
                            candidate[t].push(pos);
                        }
                        Delete_in_one_bucket(i, check, i, check, sign);
                    }
                }
            }
            if (pause){
                printf("Break because pauce! acc = %d.\n", acc);
                break;
            }
            if (acc > 100000){
                printf("Break because acc is too big!\n");
                break;
            }
        }

        delete[] candidate;
        bool flag = true;
        for (int i = 0; i < array_num; ++i)
            for (int j = 0; j < entry_num; ++j)
                if (counter[i][j] != 0)
                {
                    flag = false;
                }
        for (auto p : result)
        {
            if (p.second == 0)
            {
                result.erase(p.first);
            }
        }
        return flag;
    }
    int get_id(int n_array, int n){
        if(n_array >=0 && n_array <= array_num && n >= 0 && n <= entry_num){
            return id[n_array][n];
        }
        else{
            cout << "get_id() out of range!" << endl;
            assert(0);
        }
    }
    int get_counter(int n_array, int n){
        if(n_array >=0 && n_array <= array_num && n >= 0 && n <= entry_num){
            return counter[n_array][n];
        }
        else{
            cout << "get_counter() out of range!" << endl;
            assert(0);
        }
    }

    int set_id(int n_array, int n, int value){
        if(n_array >=0 && n_array <= array_num && n >= 0 && n <= entry_num){
            id[n_array][n] = value;
            return 1;
        }
        else{
            cout << "set_id() out of range!" << endl;
            assert(0);
            return 0;
        }
    }
    int set_counter(int n_array, int n, int value){
        if(n_array >=0 && n_array <= array_num && n >= 0 && n <= entry_num){
            counter[n_array][n] = value;
            return 1;
        }
        else{
            cout << "set_counter() out of range!" << endl;
            assert(0);
            return 0;
        }
    }
    int get_array_num() override{
        return array_num;
    }
    int get_entry_num() override{
        return entry_num;
    }
};

//ID uses + only and cnt uses + and -.
class Fermat_Count_IDP_CNTPM : public Fermat
{
    int array_num;
    int entry_num;
    int decodeflag = 0;
    // hash
    BOBHash32 *hash;
    BOBHash32 *hash_fp;

    uint32_t *table;

    bool use_fing;
    // arrays
    int32_t **fingerprint;
    uint32_t **idcpy;
    int32_t **countercpy;
    int32_t **fingcpy;
\
    BOBHash32 *hash_sign;

public:
    uint32_t **id;
    int32_t **counter;

    int get_array_num() override{
        return array_num;
    }
    int get_entry_num() override{
        return entry_num;
    }

    void clear_look_up_table() override
    {
        delete[] table;
    }

    void create_array() override
    {
        pure_cnt = 0;
        // id
        id = new uint32_t *[array_num];
        for (int i = 0; i < array_num; ++i)
        {
            id[i] = new uint32_t[entry_num];
            memset(id[i], 0, entry_num * sizeof(uint32_t));
        }
        // fingerprint
        if (use_fing)
        {
            fingerprint = new int32_t *[array_num];
            for (int i = 0; i < array_num; ++i)
            {
                fingerprint[i] = new int32_t[entry_num];
                memset(fingerprint[i], 0, entry_num * sizeof(int32_t));
            }
        }

        // counter
        counter = new int32_t *[array_num];
        for (int i = 0; i < array_num; ++i)
        {
            counter[i] = new int32_t[entry_num];
            memset(counter[i], 0, entry_num * sizeof(int32_t));
        }
    }

    void clear_array() override
    {
        for (int i = 0; i < array_num; ++i)
            delete[] id[i];
        delete[] id;

        if (fingerprint)
        {
            for (int i = 0; i < array_num; ++i)
                delete[] fingerprint[i];
            delete[] fingerprint;
        }

        for (int i = 0; i < array_num; ++i)
            delete[] counter[i];
        delete[] counter;
    }

    Fermat_Count_IDP_CNTPM(int _a, int _e, bool _fing, uint32_t _init) : array_num(_a), entry_num(_e), use_fing(_fing), fingerprint(nullptr), hash_fp(nullptr)
    {
        create_array();
        // hash
        if (use_fing)
            hash_fp = new BOBHash32(_init);
        hash = new BOBHash32[array_num];
        hash_sign = new BOBHash32[array_num];
        
        for (int i = 0; i < array_num; ++i)
        {
            hash[i].initialize(_init + (10 * i) + 1);
            hash_sign[i].initialize(_init + (17 * i) + 1);
        }
    }

    Fermat_Count_IDP_CNTPM(int _memory, bool _fing, uint32_t _init) : use_fing(_fing), fingerprint(nullptr), hash_fp(nullptr)
    {
        array_num = 3;
        if (use_fing)
            entry_num = _memory / (array_num * 12);
        else
            entry_num = _memory / (array_num * 8);

        create_array();

        // hash
        if (use_fing)
            hash_fp = new BOBHash32(_init);
        hash = new BOBHash32[array_num];
        hash_sign = new BOBHash32[array_num];
        for (int i = 0; i < array_num; ++i){

            hash[i].initialize(_init + i + 1);
            hash_sign[i].initialize(_init + (17 * i) + 1);
        }

        cout << "Finish initiating Fermat_Count_IDP_CNTPM with memory = " << _memory << " array_num = " << array_num << " entry_num = " << entry_num << endl;
    }

    ~Fermat_Count_IDP_CNTPM() override
    {
        clear_array();
        clear_look_up_table();
        if (hash_fp)
            delete hash_fp;
        delete[] hash;
    }

    int get_sign(int32_t flow_id, int i){
        uint32_t pos = hash[i].run((char *)&flow_id, sizeof(uint32_t)) % entry_num;
        uint32_t kk = hash_sign[i].run((char *)&flow_id, sizeof(uint32_t));
        int sign = HASH_TO_SIGN(kk);
        return sign;
    }

    void Insert(uint32_t flow_id, uint32_t cnt) override {    
        uint32_t flow_id_ = flow_id;
        int cnt_sign = (int)cnt>0?1:-1;
        uint32_t abscnt = abs((int)(cnt));
        insertedflows[flow_id_]+=cnt;
        if (use_fing) {
            uint32_t fing = hash_fp->run((char *)&flow_id, sizeof(uint32_t));
            for (int i = 0; i < array_num; ++i) {
                uint32_t pos = hash[i].run((char *)&flow_id_, sizeof(uint32_t)) % entry_num;
                uint32_t kk = hash_sign[i].run((char *)&flow_id_, sizeof(uint32_t));
                int sign = HASH_TO_SIGN(kk);
                id[i][pos] = ((uint64_t)id[i][pos] + (uint64_t)mulMod32(flow_id_, cnt, PRIME_ID)) % PRIME_ID;
                if(sign > 0){
                    fingerprint[i][pos] = ((int64_t)fingerprint[i][pos] + mulMod32(fing, cnt, PRIME_FING)) % PRIME_FING;
                    counter[i][pos] += cnt;
                }
                else{
                    fingerprint[i][pos] = ((int64_t)fingerprint[i][pos] - mulMod32(fing, cnt, PRIME_FING)) % PRIME_FING;
                    counter[i][pos] -= cnt;
                }
            }
        }
        else
        {
            for (int i = 0; i < array_num; ++i)
            {
                uint32_t pos = hash[i].run((char *)&flow_id_, sizeof(uint32_t)) % entry_num;
                uint32_t kk = hash_sign[i].run((char *)&flow_id_, sizeof(uint32_t));
                int sign = HASH_TO_SIGN(kk);
                if(cnt_sign > 0){
                    id[i][pos] = ((uint64_t)id[i][pos] + (uint64_t)mulMod32(flow_id_, abscnt, PRIME_ID_IDP_CNTPM)) % (uint64_t)PRIME_ID_IDP_CNTPM;
                }
                else{
                    id[i][pos] = ((uint64_t)PRIME_ID_IDP_CNTPM + (uint64_t)id[i][pos] - (uint64_t)mulMod32_Cnt(flow_id_, abscnt, PRIME_ID_IDP_CNTPM)) % (uint64_t)PRIME_ID_IDP_CNTPM;
                }
                if(sign > 0){
                    counter[i][pos] += cnt;
                }
                else{
                    counter[i][pos] -= cnt;
                }
            }
        }
    }

    void Insert_one(uint32_t flow_id) override
    {
        // flow_id should < PRIME_ID
        if (use_fing)
        {
            uint32_t fing = hash_fp->run((char *)&flow_id, sizeof(uint32_t)) % PRIME_FING;
            for (int i = 0; i < array_num; ++i)
            {
                uint32_t pos = hash[i].run((char *)&flow_id, sizeof(uint32_t)) % entry_num;
                id[i][pos] = ((uint64_t)id[i][pos] + (uint64_t)(flow_id % PRIME_ID)) % PRIME_ID;
                fingerprint[i][pos] = ((int64_t)fingerprint[i][pos] + (uint64_t)fing) % PRIME_FING;
                counter[i][pos]++;
            }
        }
        else
        {
            for (int i = 0; i < array_num; ++i)
            {
                uint32_t pos = hash[i].run((char *)&flow_id, sizeof(uint32_t)) % entry_num;
                id[i][pos] = ((uint64_t)id[i][pos] + (uint64_t)(flow_id % PRIME_ID)) % PRIME_ID;
                counter[i][pos]++;
            }
        }
    }

    void Delete_in_one_bucket(int row, int col, int pure_row, int pure_col, int sign = 1) override
    {
        id[row][col] = ((uint64_t)PRIME_ID_IDP_CNTPM + (uint64_t)id[row][col] - (uint64_t)id[pure_row][pure_col]) % PRIME_ID_IDP_CNTPM;
        // cout << "ID After minus: " << id[row][col] << endl;
        if (use_fing)
            fingerprint[row][col] = ((int64_t)PRIME_FING + (int64_t)fingerprint[row][col] - (int64_t)fingerprint[pure_row][pure_col]) % PRIME_FING;
        counter[row][col] -= counter[pure_row][pure_col];
    }

    int verify(int row, int col, uint32_t &flow_id, uint32_t &fing)
    {
        
#if DEBUG_F
        ++pure_cnt;
#endif
        uint32_t checked_id = 3458834590;
        int32_t cnt_value = counter[row][col];
        uint32_t id_value = id[row][col];
        uint64_t temp = 0;
        if (cnt_value & 0x80000000){
            temp = checkTable_count((~cnt_value + 1), PRIME_ID_IDP_CNTPM);
            flow_id = mulMod32(id_value, temp, PRIME_ID_IDP_CNTPM);
        }
        else{
            temp = checkTable(abs(cnt_value));
            flow_id = mulMod32(id_value, temp, PRIME_ID_IDP_CNTPM);
        }
        
        if (use_fing)
        {
            fing = powMod32(cnt_value, PRIME_FING - 2, PRIME_FING);
            fing = mulMod32(fingerprint[row][col], fing, PRIME_FING);
        }
        int mapto = hash[row].run((char *)&flow_id, sizeof(int32_t)) % entry_num;
        if (!(mapto == col)){
            flow_id = mulMod32(PRIME_ID_IDP_CNTPM - id_value, temp, PRIME_ID_IDP_CNTPM);
            mapto = hash[row].run((char *)&flow_id, sizeof(int32_t)) % entry_num;
            if (!(mapto == col)){
                return false;
            }
            else{
                return 2;
            }

        }
        if (use_fing && !(hash_fp->run((char *)&flow_id, sizeof(int32_t)) % PRIME_FING == fing)){
            cout << "fing is wrong!" << endl;
            return false;
        }
        return 1;
    }

    void display() override
    {
        cout << " --- display --- " << endl;
        for (int i = 0; i < array_num; ++i)
        {
            for (int j = 0; j < entry_num; ++j)
            {
                if (counter[i][j])
                {
                    cout << i << "," << j << ":" << counter[i][j] << endl;
                }
            }
        }
    }
    
    int query(const char *key) override
    {
        uint32_t flow_id = *(uint32_t *)key;
        return abs(counter[0][hash[0].run((char *)&flow_id, sizeof(uint32_t)) % entry_num]);
        uint32_t ret = 1 << 30;
        if (decodeflag)
        {
            for (int i = 0; i < array_num; ++i)
            {
                uint32_t pos = hash[i].run((char *)&flow_id, sizeof(uint32_t)) % entry_num;
                if(ret > counter[i][pos]) return counter[i][pos];
                else return ret;
            }
        }
        return (int)ret;
    }

    int query_array(const char *key, int array_index) override
    {
        uint32_t flow_id = *(uint32_t *)key;
        return abs(counter[array_index][hash[array_index].run((char *)&flow_id, sizeof(uint32_t)) % entry_num]);
        uint32_t ret = 1 << 30;
        if (decodeflag)
        {
            uint32_t pos = hash[array_index].run((char *)&flow_id, sizeof(uint32_t)) % entry_num;
            return counter[array_index][pos];
        }
        return (int)ret;
    }

    int undecoded_query(const char *key) override
    {
        //if countercpy is not defined
        if (countercpy == NULL)
        {
            cout << "countercpy is not defined!" << endl;
            if(cpy_counters()){
                cout << "countercpy defined!" << endl;
            }
            else{
                cout << "countercpy define failed!" << endl;
                return -1;
            }
        }
        uint32_t flow_id = *(uint32_t *)key;
        std::vector<int32_t> values;
        std::vector<int32_t> values_from_changed_counters;

        for (int i = 0; i < array_num; ++i)
        {
            uint32_t pos = hash[i].run((char *)&flow_id, sizeof(uint32_t)) % entry_num;
            
            uint32_t kk = hash_sign[i].run((char *)&flow_id, sizeof(uint32_t));
            int sign = HASH_TO_SIGN(kk);
            values.push_back(countercpy[i][pos]*sign);
            values_from_changed_counters.push_back(counter[i][pos]*sign);
        }

        // find median value
        size_t median_index = values.size() / 2;
        std::nth_element(values.begin(), values.begin() + median_index, values.end());
        std::nth_element(values_from_changed_counters.begin(), values_from_changed_counters.begin() + median_index, values_from_changed_counters.end());
        int32_t median = values[median_index];
        int32_t median_from = values_from_changed_counters[median_index];

        if (values.size() % 2 == 0) {
            int32_t next_median = *std::max_element(values.begin(), values.begin() + median_index);
            median = (median + next_median) / 2;
        }

        return (int)median;
    }
    int undecoded_query_before_decoding(const char *key) override
    {
        uint32_t flow_id = *(uint32_t *)key;
        std::vector<int32_t> values;

        for (int i = 0; i < array_num; ++i)
        {
            uint32_t pos = hash[i].run((char *)&flow_id, sizeof(uint32_t)) % entry_num;
            
            uint32_t kk = hash_sign[i].run((char *)&flow_id, sizeof(uint32_t));
            int sign = HASH_TO_SIGN(kk);
            values.push_back(counter[i][pos]*sign);
            cout << counter[i][pos]*sign << " ";
        }
        
        size_t median_index = values.size() / 2;
        std::nth_element(values.begin(), values.begin() + median_index, values.end());
        int32_t median = values[median_index];

        if (values.size() % 2 == 0) {
            int32_t next_median = *std::max_element(values.begin(), values.begin() + median_index);
            median = (median + next_median) / 2;
        }

        return (int)median;
    }

    bool cpy_counters_to_pos(int32_t ***counterdst){
        *counterdst = new int32_t *[array_num];
        for (int i = 0; i < array_num; i++)
        {
            (*counterdst)[i] = new int32_t[entry_num];
            for (int j = 0; j < entry_num; j++)
                (*counterdst)[i][j] = abs(counter[i][j]);
        }
        return true;
    }
    bool cpy_counters(){
        idcpy = new uint32_t *[array_num];
        for (int i = 0; i < array_num; i++)
        {
            idcpy[i] = new uint32_t[entry_num];
            for (int j = 0; j < entry_num; j++)
                idcpy[i][j] = id[i][j];
        }
        if (use_fing)
        {
            fingcpy = new int32_t *[array_num];
            for (int i = 0; i < array_num; i++)
            {
                fingcpy[i] = new int32_t[entry_num];
                for (int j = 0; j < entry_num; j++)
                    fingcpy[i][j] = fingerprint[i][j];
            }
        }
        countercpy = new int32_t *[array_num];
        for (int i = 0; i < array_num; i++)
        {
            countercpy[i] = new int32_t[entry_num];
            for (int j = 0; j < entry_num; j++)
                countercpy[i][j] = counter[i][j];
        }
        return true;
    }
    int get_sign(int array_index, char* flow_id, int size = sizeof(uint32_t)){
        uint32_t kk = hash_sign[array_index].run(flow_id, size);
        int sign = HASH_TO_SIGN(kk);
        return sign;
    }
    bool Decode(DataVariant& data) override
    {
        uint32_t checked_id = 3458834590;
        auto* mapPtr = std::get_if<std::unordered_map<int32_t, int>>(&data);
        if (!mapPtr) {
            return false;  // 如果类型不匹配，则直接返回 false
        }
        
        auto& result = *mapPtr;

        idcpy = new uint32_t *[array_num];
        for (int i = 0; i < array_num; i++)
        {
            idcpy[i] = new uint32_t[entry_num];
            for (int j = 0; j < entry_num; j++)
                idcpy[i][j] = id[i][j];
        }
        if (use_fing)
        {
            fingcpy = new int32_t *[array_num];
            for (int i = 0; i < array_num; i++)
            {
                fingcpy[i] = new int32_t[entry_num];
                for (int j = 0; j < entry_num; j++)
                    fingcpy[i][j] = fingerprint[i][j];
            }
        }
        countercpy = new int32_t *[array_num];
        for (int i = 0; i < array_num; i++)
        {
            countercpy[i] = new int32_t[entry_num];
            for (int j = 0; j < entry_num; j++)
                countercpy[i][j] = counter[i][j];
        }
        decodeflag = 1;
        queue<int> *candidate = new queue<int>[array_num];
        int32_t flow_id = 0;
        int32_t fing = 0;
        // first round
        for (int i = 0; i < array_num; ++i){
            int sign = 0;
            bool sign_cnt_fetch_pos = 0;
            for (int j = 0; j < entry_num; ++j)
            {
                uint32_t temp_flow_id = 0;
                uint32_t temp_fin = 0;

                if (counter[i][j] == 0)
                {
                    continue;
                }
                else if (verify(i, j, temp_flow_id, temp_fin) == 1)
                {
                    // find pure bucket
                    sign = get_sign(i, (char *)&temp_flow_id, sizeof(uint32_t));
                    if(id[i][j] == checked_id){
                        cout << "First(1)Find a pure bucket!" << "i: " << i << " j: " << j << " counter[i][j] = " << counter[i][j] << " id[i][j] = " << id[i][j] << " sign = " << sign << endl;
                    }
                    flow_id = (int32_t)temp_flow_id;
                    fing = (int32_t)temp_fin;
                    if (result.count(flow_id) != 0)
                    {
                        result[flow_id] += abs(counter[i][j]);
                    }
                    else
                    {
                        result[flow_id] = abs(counter[i][j]);
                    }
                    if(id[i][j] == checked_id){
                        cout << "result[" << (uint32_t)flow_id << "] = " << result[flow_id] << endl;
                    }
                    // delete flow from other rows
                    for (int t = 0; t < array_num; ++t)
                    {
                        if (t == i)
                            continue;
                        uint32_t pos = hash[t].run((char *)&flow_id, sizeof(uint32_t)) % entry_num;
                        
                        Delete_in_one_bucket(t, pos, i, j, sign);
                        candidate[t].push(pos);
                    }
                    Delete_in_one_bucket(i, j, i, j, sign);
                }
                else if (verify(i, j, temp_flow_id, temp_fin) == 2){
                    sign = get_sign(i, (char *)&temp_flow_id, sizeof(uint32_t));
                    if(id[i][j] == checked_id){
                        cout << "First(2)Find a pure bucket!" << "i: " << i << " j: " << j << " counter[i][j] = " << counter[i][j] << " id[i][j] = " << id[i][j] << " sign = " << sign << endl;
                    }
                    flow_id = (int32_t)temp_flow_id;
                    fing = (int32_t)temp_fin;
                    if (result.count(flow_id) != 0)
                    {
                        result[flow_id] += sign * counter[i][j];
                    }
                    else
                    {
                        result[flow_id] = sign * counter[i][j];
                    }
                    // delete flow from other rows
                    for (int t = 0; t < array_num; ++t)
                    {
                        if (t == i)
                            continue;
                        uint32_t pos = hash[t].run((char *)&flow_id, sizeof(uint32_t)) % entry_num;
                        
                        Delete_in_one_bucket(t, pos, i, j, sign);
                        candidate[t].push(pos);
                    }
                    Delete_in_one_bucket(i, j, i, j, sign);
                }                
            }
        }

        bool pause;
        int acc = 0;
        while (true)
        {
            acc++;
            pause = true;
            for (int i = 0; i < array_num; ++i)
            {
                int sign = get_sign(i, (char *)&flow_id, sizeof(uint32_t));
                if (!candidate[i].empty())
                    pause = false;
                while (!candidate[i].empty())
                {
                    int check = candidate[i].front();
                    candidate[i].pop();
                    uint32_t temp_flow_id = 0;
                    uint32_t temp_fin = 0;
                    if (counter[i][check] == 0)
                    {
                        continue;
                    }
                    else if (verify(i, check, temp_flow_id, temp_fin) == 1)
                    {
                        sign = get_sign(i, (char *)&temp_flow_id, sizeof(uint32_t));
                        if(id[i][check] == checked_id){
                            cout << "(1)Find a pure bucket!" << "i: " << i << " check: " << check << " counter[i][check] = " << counter[i][check] << " id[i][check] = " << " sign = " << sign << endl;
                        }
                        // find pure bucket
                        flow_id = (int32_t)temp_flow_id;
                        fing = (int32_t)temp_fin;
                        if (result.count(flow_id) != 0)
                        {
                            result[flow_id] += abs(counter[i][check]);
                            
                        }
                        else
                        {
                            result[flow_id] = abs(counter[i][check]);
                            if(counter[i][check] != abs(counter[i][check])) cout<<counter[i][check]<<" "<<abs(counter[i][check])<<" ";
                            
                        }
                        // delete flow from other rows
                        for (int t = 0; t < array_num; ++t)
                        {
                            if (t == i)
                                continue;
                            uint32_t pos = hash[t].run((char *)&flow_id, sizeof(uint32_t)) % entry_num;
                            Delete_in_one_bucket(t, pos, i, check, sign);
                            candidate[t].push(pos);
                        }
                        Delete_in_one_bucket(i, check, i, check, sign);
                    }
                    else if (verify(i, check, temp_flow_id, temp_fin) == 2)
                    {
                        sign = get_sign(i, (char *)&temp_flow_id, sizeof(uint32_t));
                        // find pure bucket
                        if(id[i][check] == checked_id){
                            cout << "(2)Find a pure bucket!" << "i: " << i << " check: " << check << " counter[i][j] = " << counter[i][check] << " id[i][check] = " << " sign = " << sign << endl;
                        }
                        flow_id = (int32_t)temp_flow_id;
                        fing = (int32_t)temp_fin;
                        if (result.count(flow_id) != 0)
                        {
                            result[flow_id] += sign * counter[i][check];
                            
                        }
                        else
                        {
                            result[flow_id] = sign * counter[i][check];
                            
                        }
                        // delete flow from other rows
                        for (int t = 0; t < array_num; ++t)
                        {
                            if (t == i)
                                continue;
                            uint32_t pos = hash[t].run((char *)&flow_id, sizeof(uint32_t)) % entry_num;
                            Delete_in_one_bucket(t, pos, i, check, sign);
                            candidate[t].push(pos);
                        }
                        Delete_in_one_bucket(i, check, i, check, sign);
                    }
                }
            }
            if (pause){
                printf("Break because pauce! acc = %d.\n", acc);
                break;
            }
            if (acc > 100000){
                printf("Break because acc is too big!\n");
                break;
            }
        }

        delete[] candidate;
        bool flag = true;
        for (int i = 0; i < array_num; ++i)
            for (int j = 0; j < entry_num; ++j)
                if (counter[i][j] != 0)
                {
                    flag = false;
                }
        for (auto p : result)
        {
            if (p.second == 0)
            {
                result.erase(p.first);
            }
        }
        return flag;
    }
    int get_id(int n_array, int n){
        if(n_array >=0 && n_array <= array_num && n >= 0 && n <= entry_num){
            return id[n_array][n];
        }
        else{
            cout << "get_id() out of range!" << endl;
            assert(0);
        }
    }
    int get_counter(int n_array, int n){
        if(n_array >=0 && n_array <= array_num && n >= 0 && n <= entry_num){
            return counter[n_array][n];
        }
        else{
            cout << "get_counter() out of range!" << endl;
            assert(0);
        }
    }

    int set_id(int n_array, int n, int value){
        if(n_array >=0 && n_array <= array_num && n >= 0 && n <= entry_num){
            id[n_array][n] = (uint32_t)value;
            return 1;
        }
        else{
            cout << "Hahahaha, set_id() out of range!" << endl;
            cout << "n_array: " << n_array << " n: " << n << " value: " << value << endl;
            assert(0);
            return 0;
        }
    }

    int set_counter(int n_array, int n, int value){
        if(n_array >=0 && n_array <= array_num && n >= 0 && n <= entry_num){
            counter[n_array][n] = value;
            return 1;
        }
        else{
            cout << "set_counter() out of range!" << endl;
            assert(0);
            return 0;
        }
    }
};

//ID uses + only and cnt uses + and -.
class Fermat_Count_IDP_CNTPM_48bits : public Fermat
{
    int array_num;
    int entry_num;
    int decodeflag = 0;
    // hash
    BOBHash32 *hash;
    BOBHash32 *hash_fp;

    uint32_t *table;

    bool use_fing;
    // arrays
    int32_t **fingerprint;
    uint32_t **idcpy;
    int16_t **countercpy;
    int32_t **fingcpy;
\
    BOBHash32 *hash_sign;

public:
    uint32_t **id;
    int16_t **counter;

    int get_array_num() override{
        return array_num;
    }
    int get_entry_num() override{
        return entry_num;
    }

    void clear_look_up_table() override
    {
        delete[] table;
    }

    void create_array() override
    {
        pure_cnt = 0;
        // id
        id = new uint32_t *[array_num];
        for (int i = 0; i < array_num; ++i)
        {
            id[i] = new uint32_t[entry_num];
            memset(id[i], 0, entry_num * sizeof(uint32_t));
        }
        // fingerprint
        if (use_fing)
        {
            fingerprint = new int32_t *[array_num];
            for (int i = 0; i < array_num; ++i)
            {
                fingerprint[i] = new int32_t[entry_num];
                memset(fingerprint[i], 0, entry_num * sizeof(int32_t));
            }
        }

        // counter
        counter = new int16_t *[array_num];
        for (int i = 0; i < array_num; ++i)
        {
            counter[i] = new int16_t[entry_num];
            memset(counter[i], 0, entry_num * sizeof(int16_t));
        }
    }

    void clear_array() override
    {
        for (int i = 0; i < array_num; ++i){
            delete[] id[i];
            id[i] = nullptr;
        }
        delete[] id;
        id = nullptr;

        if (fingerprint)
        {
            for (int i = 0; i < array_num; ++i){
                delete[] fingerprint[i];
                fingerprint[i] = nullptr;
            }
            delete[] fingerprint;
            fingerprint = nullptr;
        }

        for (int i = 0; i < array_num; ++i){
            delete[] counter[i];
            counter[i] = nullptr;
        }
        delete[] counter;
        counter = nullptr;
    }

    Fermat_Count_IDP_CNTPM_48bits(int _a, int _e, bool _fing, uint32_t _init) : array_num(_a), entry_num(_e), use_fing(_fing), fingerprint(nullptr), hash_fp(nullptr)
    {
        create_array();
        // hash
        if (use_fing)
            hash_fp = new BOBHash32(_init);
        hash = new BOBHash32[array_num];
        hash_sign = new BOBHash32[array_num];
        
        for (int i = 0; i < array_num; ++i)
        {
            hash[i].initialize(_init + (10 * i) + 1);
            hash_sign[i].initialize(_init + (17 * i) + 1);
        }
    }

    Fermat_Count_IDP_CNTPM_48bits(int _memory, bool _fing, uint32_t _init) : use_fing(_fing), fingerprint(nullptr), hash_fp(nullptr)
    {
        array_num = 3;
        if (use_fing)
            entry_num = _memory / (array_num * 12);
        else
            entry_num = _memory / (array_num * 6);

        create_array();

        // hash
        if (use_fing)
            hash_fp = new BOBHash32(_init);
        hash = new BOBHash32[array_num];
        hash_sign = new BOBHash32[array_num];
        for (int i = 0; i < array_num; ++i){

            hash[i].initialize(_init + i + 1);
            hash_sign[i].initialize(_init + (17 * i) + 1);
        }

        cout << "Finish initiating Fermat_Count_IDP_CNTPM_48bits with memory = " << _memory << " array_num = " << array_num << " entry_num = " << entry_num << endl;
    }

    ~Fermat_Count_IDP_CNTPM_48bits() override
    {
        clear_array();
        // clear_look_up_table();
        if (hash_fp){
            delete hash_fp;
            hash_fp = nullptr;
        }
        if(hash){
            delete[] hash;
            hash = nullptr;
        }
        // delete[] hash;
    }

    int get_sign(int32_t flow_id, int i){
        uint32_t pos = hash[i].run((char *)&flow_id, sizeof(uint32_t)) % entry_num;
        uint32_t kk = hash_sign[i].run((char *)&flow_id, sizeof(uint32_t));
        int sign = HASH_TO_SIGN(kk);
        return sign;
    }

    void Insert(uint32_t flow_id, uint32_t cnt) override {    
        uint32_t flow_id_ = flow_id;
        int cnt_sign = (int)cnt>0?1:-1;
        uint32_t abscnt = abs((int)(cnt));
        insertedflows[flow_id_]+=cnt;
        if (use_fing) {
            uint32_t fing = hash_fp->run((char *)&flow_id, sizeof(uint32_t));
            for (int i = 0; i < array_num; ++i) {
                uint32_t pos = hash[i].run((char *)&flow_id_, sizeof(uint32_t)) % entry_num;
                uint32_t kk = hash_sign[i].run((char *)&flow_id_, sizeof(uint32_t));
                int sign = HASH_TO_SIGN(kk);
                id[i][pos] = ((uint64_t)id[i][pos] + (uint64_t)mulMod32(flow_id_, cnt, PRIME_ID)) % PRIME_ID;
                if(sign > 0){
                    fingerprint[i][pos] = ((int64_t)fingerprint[i][pos] + mulMod32(fing, cnt, PRIME_FING)) % PRIME_FING;
                    counter[i][pos] += cnt;
                }
                else{
                    fingerprint[i][pos] = ((int64_t)fingerprint[i][pos] - mulMod32(fing, cnt, PRIME_FING)) % PRIME_FING;
                    counter[i][pos] -= cnt;
                }
            }
        }
        else
        {
            for (int i = 0; i < array_num; ++i)
            {
                uint32_t pos = hash[i].run((char *)&flow_id_, sizeof(uint32_t)) % entry_num;
                uint32_t kk = hash_sign[i].run((char *)&flow_id_, sizeof(uint32_t));
                int sign = HASH_TO_SIGN(kk);
                if(cnt_sign > 0){
                    id[i][pos] = ((uint64_t)id[i][pos] + (uint64_t)mulMod32(flow_id_, abscnt, PRIME_ID_IDP_CNTPM)) % (uint64_t)PRIME_ID_IDP_CNTPM;
                }
                else{
                    id[i][pos] = ((uint64_t)PRIME_ID_IDP_CNTPM + (uint64_t)id[i][pos] - (uint64_t)mulMod32_Cnt(flow_id_, abscnt, PRIME_ID_IDP_CNTPM)) % (uint64_t)PRIME_ID_IDP_CNTPM;
                }
                if(sign > 0){
                    counter[i][pos] += cnt;
                }
                else{
                    counter[i][pos] -= cnt;
                }
            }
        }
    }

    void Insert_one(uint32_t flow_id) override
    {
        // flow_id should < PRIME_ID
        if (use_fing)
        {
            uint32_t fing = hash_fp->run((char *)&flow_id, sizeof(uint32_t)) % PRIME_FING;
            for (int i = 0; i < array_num; ++i)
            {
                uint32_t pos = hash[i].run((char *)&flow_id, sizeof(uint32_t)) % entry_num;
                id[i][pos] = ((uint64_t)id[i][pos] + (uint64_t)(flow_id % PRIME_ID)) % PRIME_ID;
                fingerprint[i][pos] = ((int64_t)fingerprint[i][pos] + (uint64_t)fing) % PRIME_FING;
                counter[i][pos]++;
            }
        }
        else
        {
            for (int i = 0; i < array_num; ++i)
            {
                uint32_t pos = hash[i].run((char *)&flow_id, sizeof(uint32_t)) % entry_num;
                id[i][pos] = ((uint64_t)id[i][pos] + (uint64_t)(flow_id % PRIME_ID)) % PRIME_ID;
                counter[i][pos]++;
            }
        }
    }

    void Delete_in_one_bucket(int row, int col, int pure_row, int pure_col, int sign = 1) override
    {
        id[row][col] = ((uint64_t)PRIME_ID_IDP_CNTPM + (uint64_t)id[row][col] - (uint64_t)id[pure_row][pure_col]) % PRIME_ID_IDP_CNTPM;
        // cout << "ID After minus: " << id[row][col] << endl;
        if (use_fing)
            fingerprint[row][col] = ((int64_t)PRIME_FING + (int64_t)fingerprint[row][col] - (int64_t)fingerprint[pure_row][pure_col]) % PRIME_FING;
        counter[row][col] -= sign*abs(counter[pure_row][pure_col]);
    }

    int verify(int row, int col, uint32_t &flow_id, uint32_t &fing)
    {
        
#if DEBUG_F
        ++pure_cnt;
#endif
        uint32_t checked_id = -1;
        int32_t cnt_value = counter[row][col];
        uint32_t id_value = id[row][col];
        uint64_t temp = 0;
        if (cnt_value & 0x80000000){
            temp = checkTable_count((~cnt_value + 1), PRIME_ID_IDP_CNTPM);
            flow_id = mulMod32(id_value, temp, PRIME_ID_IDP_CNTPM);
        }
        else{
            temp = checkTable(abs(cnt_value));
            flow_id = mulMod32(id_value, temp, PRIME_ID_IDP_CNTPM);
        }
        bool printCondition = (flow_id == checked_id || col == 7326 || col == 4446 || col == 6509 || col == 3236 || col == 1085 || col == 1348);

        if(printCondition){
            cout << "flow_id is checked_id! counter[row][col] = " << counter[row][col] << " id[row][col] = " << id[row][col] << "row = " << row << " col = " << col << endl;
        }
        
        if (use_fing)
        {
            fing = powMod32(cnt_value, PRIME_FING - 2, PRIME_FING);
            fing = mulMod32(fingerprint[row][col], fing, PRIME_FING);
        }
        int mapto = hash[row].run((char *)&flow_id, sizeof(int32_t)) % entry_num;
        if (!(mapto == col)){
            flow_id = mulMod32(PRIME_ID_IDP_CNTPM - id_value, temp, PRIME_ID_IDP_CNTPM);
            mapto = hash[row].run((char *)&flow_id, sizeof(int32_t)) % entry_num;
            if (!(mapto == col)){
                if(printCondition)
                cout << "mapto is wrong! row = " << row << " col = " << col << " mapto = " << mapto << endl;
                return false;
            }
            else{
                if(printCondition)
                cout << "mapto is right by using reverse! row = " << row << " col = " << col << " mapto = " << mapto << endl;
                return 2;
            }
        }
        if (use_fing && !(hash_fp->run((char *)&flow_id, sizeof(int32_t)) % PRIME_FING == fing)){
            cout << "fing is wrong!" << endl;
            return false;
        }
        if(printCondition)
        cout << "mapto is right! row = " << row << " col = " << col << " mapto = " << mapto << endl;
        return 1;
    }

    int united_verify(int row, int col, uint32_t &flow_id, uint32_t &fing, TowerSketch* tower) override
    {
        
#if DEBUG_F
        ++pure_cnt;
#endif
        int returnflag = -1;
        uint32_t checked_id = 0;
        int32_t cnt_value = counter[row][col];
        uint32_t id_value = id[row][col];
        uint64_t temp = 0;
        if (cnt_value & 0x80000000){
            temp = checkTable_count((~cnt_value + 1), PRIME_ID_IDP_CNTPM);
            flow_id = mulMod32(id_value, temp, PRIME_ID_IDP_CNTPM);
        }
        else{
            temp = checkTable(abs(cnt_value));
            flow_id = mulMod32(id_value, temp, PRIME_ID_IDP_CNTPM);
        }

        bool printCondition = 0;//(flow_id == checked_id || col == 7326 || col == 4446 || col == 6509 || col == 3236 || col == 1085 || col == 1348);

        if(printCondition){
            cout << "flow_id is checked_id! counter[row][col] = " << counter[row][col] << " id[row][col] = " << id[row][col] << "row = " << row << " col = " << col << endl;
        }
        
        if (use_fing)
        {
            fing = powMod32(cnt_value, PRIME_FING - 2, PRIME_FING);
            fing = mulMod32(fingerprint[row][col], fing, PRIME_FING);
        }
        int mapto = hash[row].run((char *)&flow_id, sizeof(int32_t)) % entry_num;
        if (!(mapto == col)){
            flow_id = mulMod32(PRIME_ID_IDP_CNTPM - id_value, temp, PRIME_ID_IDP_CNTPM);
            mapto = hash[row].run((char *)&flow_id, sizeof(int32_t)) % entry_num;
            if (!(mapto == col)){
                if(printCondition)
                cout << "mapto is wrong! row = " << row << " col = " << col << " mapto = " << mapto << endl;
                // return false;
                returnflag = 0;
                goto verify_return;
            }
            else{
                if(printCondition)
                cout << "mapto is right by using reverse! row = " << row << " col = " << col << " mapto = " << mapto << endl;
                // return 2;
                if(!tower->query_if_overflow((char *)&flow_id)){
                    returnflag = 0;
                    goto verify_return;
                }
                returnflag = 2;
                goto verify_return;
            }
        }
        if (use_fing && !(hash_fp->run((char *)&flow_id, sizeof(int32_t)) % PRIME_FING == fing)){
            cout << "fing is wrong!" << endl;
            // return false;
            returnflag = 0;
            goto verify_return;
        }
        if(!tower->query_if_overflow((char *)&flow_id)){
            returnflag = 0;
            goto verify_return;
        }
        
        if(printCondition)
            cout << "mapto is right! row = " << row << " col = " << col << " mapto = " << mapto << endl;
        // return 1;
        returnflag = 1;
verify_return:

        if(tower->query_if_overflow((char *)&flow_id) && returnflag == 0){
            cout << "[[[[[[[[[[[[[Cnt_value = " << cnt_value << " id_value = " << id_value << " temp = " << temp << " flow_id = " << flow_id << endl;
            cout << "[[[[[[[[[[[[[Tower overflow said yes! But verify value is " << returnflag << endl;
        }
        else if(!tower->query_if_overflow((char *)&flow_id) && returnflag != 0){
            cout << "[[[[[[[[[[[[[Cnt_value = " << cnt_value << " id_value = " << id_value << " temp = " << temp << " flow_id = " << flow_id << endl;
            cout << "[[[[[[[[[[[[[Tower overflow said no! But verify value is " << returnflag << endl;
        }

        return returnflag;
    }

    void display() override
    {
        cout << " --- display --- " << endl;
        for (int i = 0; i < array_num; ++i)
        {
            for (int j = 0; j < entry_num; ++j)
            {
                if (counter[i][j])
                {
                    cout << i << "," << j << ":" << counter[i][j] << endl;
                }
            }
        }
    }
    
    int query(const char *key) override
    {
        uint32_t flow_id = *(uint32_t *)key;
        return abs(counter[0][hash[0].run((char *)&flow_id, sizeof(uint32_t)) % entry_num]);
        uint32_t ret = 1 << 30;
        if (decodeflag)
        {
            for (int i = 0; i < array_num; ++i)
            {
                uint32_t pos = hash[i].run((char *)&flow_id, sizeof(uint32_t)) % entry_num;
                if(ret > counter[i][pos]) return counter[i][pos];
                else return ret;
            }
        }
        return (int)ret;
    }

    int query_array(const char *key, int array_index) override
    {
        uint32_t flow_id = *(uint32_t *)key;
        return abs(counter[array_index][hash[array_index].run((char *)&flow_id, sizeof(uint32_t)) % entry_num]);
        uint32_t ret = 1 << 30;
        if (decodeflag)
        {
            uint32_t pos = hash[array_index].run((char *)&flow_id, sizeof(uint32_t)) % entry_num;
            return counter[array_index][pos];
        }
        return (int)ret;
    }

    int undecoded_query(const char *key) override
    {
        //if countercpy is not defined
        if (countercpy == NULL)
        {
            cout << "countercpy is not defined!" << endl;
            if(cpy_counters()){
                cout << "countercpy defined!" << endl;
            }
            else{
                cout << "countercpy define failed!" << endl;
                return -1;
            }
        }
        uint32_t flow_id = *(uint32_t *)key;
        std::vector<int32_t> values;
        std::vector<int32_t> values_from_changed_counters;

        for (int i = 0; i < array_num; ++i)
        {
            uint32_t pos = hash[i].run((char *)&flow_id, sizeof(uint32_t)) % entry_num;
            
            uint32_t kk = hash_sign[i].run((char *)&flow_id, sizeof(uint32_t));
            int sign = HASH_TO_SIGN(kk);
            values.push_back(countercpy[i][pos]*sign);
            values_from_changed_counters.push_back(counter[i][pos]*sign);
        }

        // find median value
        size_t median_index = values.size() / 2;
        std::nth_element(values.begin(), values.begin() + median_index, values.end());
        std::nth_element(values_from_changed_counters.begin(), values_from_changed_counters.begin() + median_index, values_from_changed_counters.end());
        int32_t median = values[median_index];
        int32_t median_from = values_from_changed_counters[median_index];

        if (values.size() % 2 == 0) {
            int32_t next_median = *std::max_element(values.begin(), values.begin() + median_index);
            median = (median + next_median) / 2;
        }

        return (int)median;
    }
    int undecoded_query_before_decoding(const char *key) override
    {
        uint32_t flow_id = *(uint32_t *)key;
        std::vector<int32_t> values;

        for (int i = 0; i < array_num; ++i)
        {
            uint32_t pos = hash[i].run((char *)&flow_id, sizeof(uint32_t)) % entry_num;
            
            uint32_t kk = hash_sign[i].run((char *)&flow_id, sizeof(uint32_t));
            int sign = HASH_TO_SIGN(kk);
            values.push_back(counter[i][pos]*sign);
            cout << counter[i][pos]*sign << " ";
        }
        
        size_t median_index = values.size() / 2;
        std::nth_element(values.begin(), values.begin() + median_index, values.end());
        int32_t median = values[median_index];

        if (values.size() % 2 == 0) {
            int32_t next_median = *std::max_element(values.begin(), values.begin() + median_index);
            median = (median + next_median) / 2;
        }

        return (int)median;
    }

    bool cpy_counters_to_pos(int32_t ***counterdst){
        *counterdst = new int32_t *[array_num];
        for (int i = 0; i < array_num; i++)
        {
            (*counterdst)[i] = new int32_t[entry_num];
            for (int j = 0; j < entry_num; j++)
                (*counterdst)[i][j] = abs(counter[i][j]);
        }
        return true;
    }
    bool cpy_counters(){
        idcpy = new uint32_t *[array_num];
        for (int i = 0; i < array_num; i++)
        {
            idcpy[i] = new uint32_t[entry_num];
            for (int j = 0; j < entry_num; j++)
                idcpy[i][j] = id[i][j];
        }
        if (use_fing)
        {
            fingcpy = new int32_t *[array_num];
            for (int i = 0; i < array_num; i++)
            {
                fingcpy[i] = new int32_t[entry_num];
                for (int j = 0; j < entry_num; j++)
                    fingcpy[i][j] = fingerprint[i][j];
            }
        }
        countercpy = new int16_t *[array_num];
        for (int i = 0; i < array_num; i++)
        {
            countercpy[i] = new int16_t[entry_num];
            for (int j = 0; j < entry_num; j++)
                countercpy[i][j] = counter[i][j];
        }
        return true;
    }
    int get_sign(int array_index, char* flow_id, int size = sizeof(uint32_t)){
        uint32_t kk = hash_sign[array_index].run(flow_id, size);
        int sign = HASH_TO_SIGN(kk);
        return sign;
    }
    bool Decode(DataVariant& data) override
    {
        uint32_t checked_id = 0;
        auto* mapPtr = std::get_if<std::unordered_map<int32_t, int>>(&data);
        if (!mapPtr) {
            return false;  // 如果类型不匹配，则直接返回 false
        }
        
        auto& result = *mapPtr;

        idcpy = new uint32_t *[array_num];
        for (int i = 0; i < array_num; i++)
        {
            idcpy[i] = new uint32_t[entry_num];
            for (int j = 0; j < entry_num; j++)
                idcpy[i][j] = id[i][j];
        }
        if (use_fing)
        {
            fingcpy = new int32_t *[array_num];
            for (int i = 0; i < array_num; i++)
            {
                fingcpy[i] = new int32_t[entry_num];
                for (int j = 0; j < entry_num; j++)
                    fingcpy[i][j] = fingerprint[i][j];
            }
        }
        countercpy = new int16_t *[array_num];
        for (int i = 0; i < array_num; i++)
        {
            countercpy[i] = new int16_t[entry_num];
            for (int j = 0; j < entry_num; j++)
                countercpy[i][j] = counter[i][j];
        }
        decodeflag = 1;
        queue<int> *candidate = new queue<int>[array_num];
        int32_t flow_id = 0;
        int32_t fing = 0;
        // first round
        for (int i = 0; i < array_num; ++i){
            int sign = 0;
            bool sign_cnt_fetch_pos = 0;
            for (int j = 0; j < entry_num; ++j)
            {
                uint32_t temp_flow_id = 0;
                uint32_t temp_fin = 0;

                if (counter[i][j] == 0)
                {
                    continue;
                }
                else if (verify(i, j, temp_flow_id, temp_fin) == 1)
                {
                    // find pure bucket
                    sign = get_sign(i, (char *)&temp_flow_id, sizeof(uint32_t));
                    flow_id = (int32_t)temp_flow_id;
                    fing = (int32_t)temp_fin;
                    if (result.count(flow_id) != 0){
                        result[flow_id] += abs(counter[i][j]);
                    }
                    else{
                        result[flow_id] = abs(counter[i][j]);
                    }
                    // delete flow from other rows
                    for (int t = 0; t < array_num; ++t){
                        if (t == i)
                            continue;
                        uint32_t pos = hash[t].run((char *)&flow_id, sizeof(uint32_t)) % entry_num;
                        int other_sign = get_sign(t, (char *)&temp_flow_id, sizeof(uint32_t));
                        
                        Delete_in_one_bucket(t, pos, i, j, other_sign);
                        candidate[t].push(pos);
                    }
                    Delete_in_one_bucket(i, j, i, j, sign);
                }
                else if (verify(i, j, temp_flow_id, temp_fin) == 2){
                    sign = get_sign(i, (char *)&temp_flow_id, sizeof(uint32_t));
                    flow_id = (int32_t)temp_flow_id;
                    fing = (int32_t)temp_fin;
                    if (result.count(flow_id) != 0){
                        result[flow_id] += sign * counter[i][j];
                    }
                    else{
                        result[flow_id] = sign * counter[i][j];
                    }
                    // delete flow from other rows
                    for (int t = 0; t < array_num; ++t)
                    {
                        if (t == i)
                            continue;
                        uint32_t pos = hash[t].run((char *)&flow_id, sizeof(uint32_t)) % entry_num;
                        int other_sign = get_sign(t, (char *)&temp_flow_id, sizeof(uint32_t));

                        Delete_in_one_bucket(t, pos, i, j, other_sign);
                        candidate[t].push(pos);
                    }
                    Delete_in_one_bucket(i, j, i, j, sign);
                }                
            }
        }

        bool pause;
        int acc = 0;
        while (true)
        {
            // for (int i = 0; i < array_num; ++i) {
            //     // 创建一个副本队列
            //     std::queue<int> q = candidate[i];

            //     std::cout << "Queue " << i << ": ";
            //     while (!q.empty()) {
            //         std::cout << q.front() << " ";
            //         q.pop();
            //     }
            //     std::cout << std::endl;
            // }
            acc++;
            pause = true;
            for (int i = 0; i < array_num; ++i)
            {
                int sign = get_sign(i, (char *)&flow_id, sizeof(uint32_t));
                if (!candidate[i].empty())
                    pause = false;
                while (!candidate[i].empty())
                {
                    int check = candidate[i].front();
                    // cout << "checking: " << check << " of array " << i << std::endl;
                    candidate[i].pop();
                    uint32_t temp_flow_id = 0;
                    uint32_t temp_fin = 0;
                    if (counter[i][check] == 0)
                    {
                        continue;
                    }
                    else if (verify(i, check, temp_flow_id, temp_fin) == 1)
                    {
                        sign = get_sign(i, (char *)&temp_flow_id, sizeof(uint32_t));
                        if(id[i][check] == checked_id){
                            cout << "(1)Find a pure bucket!" << "i: " << i << " check: " << check << " counter[i][check] = " << counter[i][check] << " id[i][check] = " << " sign = " << sign << endl;
                        }
                        // find pure bucket
                        flow_id = (int32_t)temp_flow_id;
                        fing = (int32_t)temp_fin;
                        if (result.count(flow_id) != 0)
                        {
                            result[flow_id] += abs(counter[i][check]);
                            
                        }
                        else
                        {
                            result[flow_id] = abs(counter[i][check]);
                            // if(counter[i][check] != abs(counter[i][check])) cout<<counter[i][check]<<" "<<abs(counter[i][check])<<" ";
                        }
                        // delete flow from other rows
                        for (int t = 0; t < array_num; ++t)
                        {
                            if (t == i)
                                continue;
                            uint32_t pos = hash[t].run((char *)&flow_id, sizeof(uint32_t)) % entry_num;
                            int other_sign = get_sign(t, (char *)&temp_flow_id, sizeof(uint32_t));
                            Delete_in_one_bucket(t, pos, i, check, other_sign);
                            candidate[t].push(pos);
                            // cout << "Pushing " << pos << " to " << t << std::endl;
                        }
                        Delete_in_one_bucket(i, check, i, check, sign);
                    }
                    else if (verify(i, check, temp_flow_id, temp_fin) == 2)
                    {
                        sign = get_sign(i, (char *)&temp_flow_id, sizeof(uint32_t));
                        // find pure bucket
                        if(id[i][check] == checked_id){
                            cout << "(2)Find a pure bucket!" << "i: " << i << " check: " << check << " counter[i][j] = " << counter[i][check] << " id[i][check] = " << " sign = " << sign << endl;
                        }
                        flow_id = (int32_t)temp_flow_id;
                        fing = (int32_t)temp_fin;
                        if (result.count(flow_id) != 0)
                        {
                            result[flow_id] += sign * counter[i][check];
                            
                        }
                        else
                        {
                            result[flow_id] = sign * counter[i][check];
                            
                        }
                        // delete flow from other rows
                        for (int t = 0; t < array_num; ++t)
                        {
                            if (t == i)
                                continue;
                            uint32_t pos = hash[t].run((char *)&flow_id, sizeof(uint32_t)) % entry_num;
                            int other_sign = get_sign(t, (char *)&temp_flow_id, sizeof(uint32_t));
                            Delete_in_one_bucket(t, pos, i, check, other_sign);
                            // cout << "Pushing " << pos << " to " << t << std::endl;
                            candidate[t].push(pos);
                        }
                        Delete_in_one_bucket(i, check, i, check, sign);
                    }
                }
            }
            if (pause){
                printf("Break because pauce! acc = %d.\n", acc);
                break;
            }
            if (acc > 100000){
                printf("Break because acc is too big!\n");
                break;
            }
        }

        delete[] candidate;
        bool flag = true;
        for (int i = 0; i < array_num; ++i)
            for (int j = 0; j < entry_num; ++j)
                if (counter[i][j] != 0)
                {
                    flag = false;
                }
        for (auto p : result)
        {
            if (p.second == 0)
            {
                result.erase(p.first);
            }
        }
        return flag;
    }

    bool united_decode(DataVariant& data, TowerSketch* tower) override
    {
        uint32_t checked_id = 0;
        auto* mapPtr = std::get_if<std::unordered_map<int32_t, int>>(&data);
        if (!mapPtr) {
            return false;  // 如果类型不匹配，则直接返回 false
        }
        
        auto& result = *mapPtr;

        idcpy = new uint32_t *[array_num];
        for (int i = 0; i < array_num; i++)
        {
            idcpy[i] = new uint32_t[entry_num];
            for (int j = 0; j < entry_num; j++)
                idcpy[i][j] = id[i][j];
        }
        if (use_fing)
        {
            fingcpy = new int32_t *[array_num];
            for (int i = 0; i < array_num; i++)
            {
                fingcpy[i] = new int32_t[entry_num];
                for (int j = 0; j < entry_num; j++)
                    fingcpy[i][j] = fingerprint[i][j];
            }
        }
        countercpy = new int16_t *[array_num];
        for (int i = 0; i < array_num; i++)
        {
            countercpy[i] = new int16_t[entry_num];
            for (int j = 0; j < entry_num; j++)
                countercpy[i][j] = counter[i][j];
        }
        decodeflag = 1;
        queue<int> *candidate = new queue<int>[array_num];
        int32_t flow_id = 0;
        int32_t fing = 0;
        // first round
        for (int i = 0; i < array_num; ++i){
            int sign = 0;
            bool sign_cnt_fetch_pos = 0;
            for (int j = 0; j < entry_num; ++j)
            {
                uint32_t temp_flow_id = 0;
                uint32_t temp_fin = 0;

                if (counter[i][j] == 0)
                {
                    continue;
                }
                else if (united_verify(i, j, temp_flow_id, temp_fin, tower) == 1)
                {
                    // find pure bucket
                    sign = get_sign(i, (char *)&temp_flow_id, sizeof(uint32_t));
                    flow_id = (int32_t)temp_flow_id;
                    fing = (int32_t)temp_fin;
                    if (result.count(flow_id) != 0){
                        result[flow_id] += abs(counter[i][j]);
                    }
                    else{
                        result[flow_id] = abs(counter[i][j]);
                    }
                    // delete flow from other rows
                    for (int t = 0; t < array_num; ++t){
                        if (t == i)
                            continue;
                        uint32_t pos = hash[t].run((char *)&flow_id, sizeof(uint32_t)) % entry_num;
                        int other_sign = get_sign(t, (char *)&temp_flow_id, sizeof(uint32_t));
                        
                        Delete_in_one_bucket(t, pos, i, j, other_sign);
                        candidate[t].push(pos);
                    }
                    Delete_in_one_bucket(i, j, i, j, sign);
                }
                else if (united_verify(i, j, temp_flow_id, temp_fin, tower) == 2){
                    sign = get_sign(i, (char *)&temp_flow_id, sizeof(uint32_t));
                    flow_id = (int32_t)temp_flow_id;
                    fing = (int32_t)temp_fin;
                    if (result.count(flow_id) != 0){
                        result[flow_id] += sign * counter[i][j];
                    }
                    else{
                        result[flow_id] = sign * counter[i][j];
                    }
                    // delete flow from other rows
                    for (int t = 0; t < array_num; ++t)
                    {
                        if (t == i)
                            continue;
                        uint32_t pos = hash[t].run((char *)&flow_id, sizeof(uint32_t)) % entry_num;
                        int other_sign = get_sign(t, (char *)&temp_flow_id, sizeof(uint32_t));

                        Delete_in_one_bucket(t, pos, i, j, other_sign);
                        candidate[t].push(pos);
                    }
                    Delete_in_one_bucket(i, j, i, j, sign);
                }                
            }
        }

        bool pause;
        int acc = 0;
        while (true)
        {
            // for (int i = 0; i < array_num; ++i) {
            //     // 创建一个副本队列
            //     std::queue<int> q = candidate[i];

            //     std::cout << "Queue " << i << ": ";
            //     while (!q.empty()) {
            //         std::cout << q.front() << " ";
            //         q.pop();
            //     }
            //     std::cout << std::endl;
            // }
            acc++;
            pause = true;
            for (int i = 0; i < array_num; ++i)
            {
                int sign = get_sign(i, (char *)&flow_id, sizeof(uint32_t));
                if (!candidate[i].empty())
                    pause = false;
                while (!candidate[i].empty())
                {
                    int check = candidate[i].front();
                    // cout << "checking: " << check << " of array " << i << std::endl;
                    candidate[i].pop();
                    uint32_t temp_flow_id = 0;
                    uint32_t temp_fin = 0;
                    if (counter[i][check] == 0)
                    {
                        continue;
                    }
                    else if (united_verify(i, check, temp_flow_id, temp_fin, tower) == 1)
                    {
                        sign = get_sign(i, (char *)&temp_flow_id, sizeof(uint32_t));
                        if(id[i][check] == checked_id){
                            cout << "(1)Find a pure bucket!" << "i: " << i << " check: " << check << " counter[i][check] = " << counter[i][check] << " id[i][check] = " << " sign = " << sign << endl;
                        }
                        // find pure bucket
                        flow_id = (int32_t)temp_flow_id;
                        fing = (int32_t)temp_fin;
                        if (result.count(flow_id) != 0)
                        {
                            result[flow_id] += abs(counter[i][check]);
                            
                        }
                        else
                        {
                            result[flow_id] = abs(counter[i][check]);
                            // if(counter[i][check] != abs(counter[i][check])) cout<<counter[i][check]<<" "<<abs(counter[i][check])<<" ";
                        }
                        // delete flow from other rows
                        for (int t = 0; t < array_num; ++t)
                        {
                            if (t == i)
                                continue;
                            uint32_t pos = hash[t].run((char *)&flow_id, sizeof(uint32_t)) % entry_num;
                            int other_sign = get_sign(t, (char *)&temp_flow_id, sizeof(uint32_t));
                            Delete_in_one_bucket(t, pos, i, check, other_sign);
                            candidate[t].push(pos);
                            // cout << "Pushing " << pos << " to " << t << std::endl;
                        }
                        Delete_in_one_bucket(i, check, i, check, sign);
                    }
                    else if (united_verify(i, check, temp_flow_id, temp_fin, tower) == 2)
                    {
                        sign = get_sign(i, (char *)&temp_flow_id, sizeof(uint32_t));
                        // find pure bucket
                        if(id[i][check] == checked_id){
                            cout << "(2)Find a pure bucket!" << "i: " << i << " check: " << check << " counter[i][j] = " << counter[i][check] << " id[i][check] = " << " sign = " << sign << endl;
                        }
                        flow_id = (int32_t)temp_flow_id;
                        fing = (int32_t)temp_fin;
                        if (result.count(flow_id) != 0)
                        {
                            result[flow_id] += sign * counter[i][check];
                            
                        }
                        else
                        {
                            result[flow_id] = sign * counter[i][check];
                            
                        }
                        // delete flow from other rows
                        for (int t = 0; t < array_num; ++t)
                        {
                            if (t == i)
                                continue;
                            uint32_t pos = hash[t].run((char *)&flow_id, sizeof(uint32_t)) % entry_num;
                            int other_sign = get_sign(t, (char *)&temp_flow_id, sizeof(uint32_t));
                            Delete_in_one_bucket(t, pos, i, check, other_sign);
                            // cout << "Pushing " << pos << " to " << t << std::endl;
                            candidate[t].push(pos);
                        }
                        Delete_in_one_bucket(i, check, i, check, sign);
                    }
                }
            }
            if (pause){
                printf("Break because pauce! acc = %d.\n", acc);
                break;
            }
            if (acc > 100000){
                printf("Break because acc is too big!\n");
                break;
            }
        }

        delete[] candidate;
        bool flag = true;
        for (int i = 0; i < array_num; ++i)
            for (int j = 0; j < entry_num; ++j)
                if (counter[i][j] != 0)
                {
                    flag = false;
                }
        for (auto p : result)
        {
            if (p.second == 0)
            {
                result.erase(p.first);
            }
        }
        return flag;
    }
    int get_id(int n_array, int n){
        if(n_array >=0 && n_array <= array_num && n >= 0 && n <= entry_num){
            return id[n_array][n];
        }
        else{
            cout << "get_id() out of range!" << endl;
            assert(0);
        }
    }
    int get_counter(int n_array, int n){
        if(n_array >=0 && n_array <= array_num && n >= 0 && n <= entry_num){
            return counter[n_array][n];
        }
        else{
            cout << "get_counter() out of range!" << endl;
            assert(0);
        }
    }

    int set_id(int n_array, int n, int value){
        if(n_array >=0 && n_array <= array_num && n >= 0 && n <= entry_num){
            id[n_array][n] = (uint32_t)value;
            return 1;
        }
        else{
            cout << "Hahahaha, set_id() out of range!" << endl;
            cout << "n_array: " << n_array << " n: " << n << " value: " << value << endl;
            assert(0);
            return 0;
        }
    }

    int set_counter(int n_array, int n, int value){
        if(n_array >=0 && n_array <= array_num && n >= 0 && n <= entry_num){
            counter[n_array][n] = value;
            return 1;
        }
        else{
            cout << "set_counter() out of range!" << endl;
            assert(0);
            return 0;
        }
    }
};

#endif