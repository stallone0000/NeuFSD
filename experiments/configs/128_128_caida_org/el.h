#ifndef ELASTIC_SKETCH_H
#define ELASTIC_SKETCH_H

#include <cstdint>
#include <algorithm>
#include <cstring>
#include "BOBHash.h"

#define ES_d 1
#define BN 8
#define lambda 8

class ElasticSketch {
private:
    struct Heavy { 
        uint32_t FP;
        uint32_t pvote;
        uint32_t nvote;
        uint32_t Flag;
    };

    struct Light { 
        uint32_t C;
    };
    
    Heavy** HK;  // 重型桶
    Light** LK;  // 轻型桶
    
    int M1, M2;
    BOBHash** bobhash;      // 主哈希函数
    BOBHash** bobhash_aux;  // 辅助哈希函数数组
    
public:
    ElasticSketch(int M1, int M2) : M1(M1), M2(M2) {
        // 分配重型桶内存
        HK = new Heavy*[M1];
        for (int i = 0; i < M1; ++i) {
            HK[i] = new Heavy[BN];
            for (int j = 0; j < BN; ++j) {
                HK[i][j].FP = 0;
                HK[i][j].pvote = 0;
                HK[i][j].nvote = 0;
                HK[i][j].Flag = 0;
            }
        }
        
        // 分配轻型桶内存
        LK = new Light*[ES_d];
        for (int i = 0; i < ES_d; ++i) {
            LK[i] = new Light[M2];
            for (int j = 0; j < M2; ++j) {
                LK[i][j].C = 0;
            }
        }
        
        // 初始化哈希函数
        bobhash = new BOBHash*[1];
        bobhash[0] = new BOBHash(995);
        
        bobhash_aux = new BOBHash*[ES_d];
        for (int i = 0; i < ES_d; ++i) {
            bobhash_aux[i] = new BOBHash(1000 + i);
        }
    }
    
    ~ElasticSketch() {
        // 释放内存
        for (int i = 0; i < M1; ++i) {
            delete[] HK[i];
        }
        delete[] HK;
        
        for (int i = 0; i < ES_d; ++i) {
            delete[] LK[i];
        }
        delete[] LK;
        
        // 释放哈希函数
        for (int i = 0; i < ES_d; ++i) {
            delete bobhash_aux[i];
        }
        delete[] bobhash_aux;
        
        delete bobhash[0];
        delete[] bobhash;
    }
    
    void clear() {
        // 重型桶重置
        for (int i = 0; i < M1; ++i) {
            for (int j = 0; j < BN; ++j) {
                HK[i][j].FP = 0;
                HK[i][j].pvote = 0;
                HK[i][j].nvote = 0;
                HK[i][j].Flag = 0;
            }
        }
        
        // 轻型桶重置
        for (int i = 0; i < ES_d; ++i) {
            for (int j = 0; j < M2; ++j) {
                LK[i][j].C = 0;
            }
        }
    }
    
    void Insert(const char* str, size_t len) {
        uint32_t FP = bobhash[0]->run(str, len);
        uint32_t H1 = FP % M1;
        
        uint32_t hash[ES_d];
        for (int i = 0; i < ES_d; ++i) {
            hash[i] = bobhash_aux[i]->run((const char*)&FP, sizeof(FP)) % M2;
        }
    
        bool found = false;
        int min_pos = -1;
        unsigned int min_size = -1;
        
        // 在重型桶中查找空槽或相同指纹
        for (int j = 0; j < BN; j++) {
            if (HK[H1][j].FP == FP) {
                HK[H1][j].pvote++;
                found = true;
                break;
            }
            else if (HK[H1][j].pvote == 0) {
                HK[H1][j].pvote = 1;
                HK[H1][j].FP = FP;
                found = true;
                break;
            }
            
            if (min_size > HK[H1][j].pvote) {
                min_pos = j;
                min_size = HK[H1][j].pvote;
            }
        }
        
        // 未找到位置，尝试驱逐
        if (!found) {
            HK[H1][min_pos].nvote++;
            
            // 检查是否满足驱逐条件
            if (HK[H1][min_pos].pvote > 0 && 
                HK[H1][min_pos].nvote / HK[H1][min_pos].pvote >= lambda) {
                
                // 迁移旧流到轻型部分
                for (int i = 0; i < ES_d; i++) {
                    uint32_t tmphash = bobhash_aux[i]->run(
                        (const char*)&HK[H1][min_pos].FP, 
                        sizeof(HK[H1][min_pos].FP)
                    ) % M2;
                    LK[i][tmphash].C += HK[H1][min_pos].pvote;
                }
                
                // 新流占据槽位
                HK[H1][min_pos].FP = FP;
                HK[H1][min_pos].Flag = 1;
                HK[H1][min_pos].nvote = 0;
                HK[H1][min_pos].pvote = 1;
                found = true;
            }
        }
        
        // 如果既未找到也未驱逐，添加到轻型部分
        if (!found) {
            for (int i = 0; i < ES_d; i++) {
                LK[i][hash[i]].C++;
            }
        }
    }
    
    // 获取重型部分数据（供外部访问）
    Heavy** get_heavy_part() const { return HK; }
    int get_M1() const { return M1; }
    int get_buckets() const { return BN; }
};

#endif // ELASTIC_SKETCH_H