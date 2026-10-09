#ifndef _COMMON_FUNC_H_
#define _COMMON_FUNC_H_

#include <iostream>
#include <utility>
#include <unistd.h>
#include <stdint.h>
#include <vector>
#include <set>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <fstream>
#include <arpa/inet.h>
#include <cstring>
#include <random>
#include <stdexcept>
#include <sstream>
#include <string>

using std::make_pair;
using std::map;
using std::pair;
using std::set;
using std::string;
using std::unordered_map;
using std::vector;

using namespace std;

int prime_seeds[] = {37, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, \
                      103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, \
                      163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, \
                      227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, \
                      281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, \
                      353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, \
                      421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, \
                      487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, \
                      569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, \
                      631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, \
                      701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, \
                      773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, \
                      857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, \
                      937, 941, 947, 953, 967, 971, 977, 983, 991, 997, 1009, \
                      1013, 1019, 1021, 1031, 1033, 1039, 1049, 1051, 1061, \
                      1063, 1069, 1087, 1091, 1093, 1097, 1103, 1109, 1117, \
                      1123, 1129, 1151, 1153, 1163, 1171, 1181, 1187, 1193, \
                      1201, 1213, 1217, 1223, 1229, 1231, 1237, 1249, 1259, \
                      1277, 1279, 1283, 1289, 1291, 1297, 1301, 1303, 1307, \
                      1319, 1321, 1327, 1361, 1367, 1373, 1381, 1399, 1409, \
                      1423, 1427, 1429, 1433, 1439, 1447, 1451, 1453, 1459, \
                      1471, 1481, 1483, 1487, 1489, 1493, 1499, 1511, 1523, \
                      1531, 1543, 1549, 1553, 1559, 1567, 1571, 1579, 1583, \
                      1597, 1601, 1607, 1609, 1613, 1619, 1621, 1627, 1637,};

/************************** Loading Traces ******************************/

#define NUM_TRACE 12               // Number of traces in DATA directory
#define TIMES 10                   // Times of each algorithm measuring the same
                                   // trace using different hash function
#define DATA_ROOT_15s "data/data" // NUM_TRACE = 1, CAIDA
#define MY_RANDOM_SEED 813

struct SRCIP_TUPLE
{
  char key[13] = {0};
};
struct REST_TUPLE
{
  char key[9];
};

typedef vector<SRCIP_TUPLE> TRACE;

TRACE traces[NUM_TRACE];

/************/

string Stringsplit(string str)
{
	istringstream iss(str);
	string token;
  int num = 0;
	while (getline(iss, token, '|'))
	{
        if (num == 0)
            break;
        num++;
	}
    return token;
}

void tochar(string s, char arr[13]) {
    int num = stoi(s);
    std::stringstream ss;  
    ss << std::hex << num;
    std::string hexStr = ss.str();
    if (hexStr.length() > 4) {  
        hexStr = hexStr.substr(hexStr.length() - 4); // 取最后4个字符  
    } 
    for (size_t i = 0; i < hexStr.length(); i++) {  
        arr[i] = hexStr[i];
    }
}

uint32_t readData()
{
  uint32_t total_pck_num = 0;
  string filenames[3] = {
    "data/tpc-ds/store_sales.dat",
    "data/tpc-ds/web_sales.dat",
    "data/tpc-ds/catalog_sales.dat"
  };
  // string filename = "./tpc-ds/web_sales.dat";
  int window = 0;
  string line;
  SRCIP_TUPLE key;

  for (const string& filename : filenames) {
  ifstream file(filename.c_str());
  if (!getline(file, line)){
    printf("[ERROR] file error!\n");
    exit(0);
  }
  while (getline(file, line)) {
    string token = Stringsplit(line);
    if (token == "")
      continue; 
    // cout << "token = " << token << endl; 
    // break;
    char tmp[13] = {0};
    tochar(token.c_str(), tmp);
    memcpy(&key, tmp, 4);
    traces[window].push_back(key);
    total_pck_num++;
  }
  // printf("[INFO] Scanned, packets number:\n");
  // int i = 0;
  // printf("[INFO] window %02d has %ld packets\n", i, traces[i].size());
  }
  printf("[INFO] Scanned, packets number:\n");
  int i = 0;
  printf("[INFO] window %02d has %ld packets\n", i, traces[i].size());
  return total_pck_num;
}

/************/

uint32_t ReadTwoWindows()
{
  TRACE all_packets;
  uint32_t total_pck_num = 0;
  string filename130000 = "data/0.dat";
  char tmp[13] = {0};
  FILE *file1 = fopen(filename130000.c_str(), "r");
  if (file1 == NULL)
  {
    printf("[ERROR] file not open\n");
    exit(0);
  }
  SRCIP_TUPLE key;
  int window = 0;
  if (!fread(tmp, 13, 1, file1)){
    printf("[ERROR] file error!\n");
    exit(0);
  }
  while (fread(tmp, 13, 1, file1))
  {
    memcpy(&key, tmp, 4);
    all_packets.push_back(key);
    total_pck_num++;
  }
  for(int i = 0; i < total_pck_num / 2; i++){
    key = all_packets[i];
    traces[0].push_back(key);
  }
  for(int i = total_pck_num / 2; i < total_pck_num; i++){
    key = all_packets[i];
    traces[1].push_back(key);
  }
  printf("[INFO] Scanned, packets number:\n");
  fclose(file1);
  printf("[INFO] window %02d has %ld packets\n", 0, traces[0].size());
  printf("[INFO] window %02d has %ld packets\n", 1, traces[1].size());
  return total_pck_num;
}

uint32_t ReadNTraces(int n)
{
    if (n < 0 || n > 10) {
        printf("[ERROR] Invalid value of n. It should be between 0 and 10.\n");
        return 0;
    }

    uint32_t total_pck_num = 0;
    char tmp[13] = {0};

    for (int i = 0; i <= n; i++) {
        string filename = "data/" + to_string(i) + ".dat";
        FILE *file = fopen(filename.c_str(), "r");
        if (file == NULL) {
            printf("[ERROR] File %s not open\n", filename.c_str());
            continue;
        }

        SRCIP_TUPLE key;
        int window = 0;

        if (!fread(tmp, 13, 1, file)) {
            printf("[ERROR] File %s error!\n", filename.c_str());
            fclose(file);
            continue;
        }

        while (fread(tmp, 13, 1, file)) {
            memcpy(&key, tmp, 4);
            traces[window].push_back(key);
            total_pck_num++;
        }

        fclose(file);
        printf("[INFO] File %s scanned, packets number: %ld\n", filename.c_str(), traces[window].size());
    }

    printf("[INFO] Total packets number: %u\n", total_pck_num);
    return total_pck_num;
}

uint32_t myReadTraces()
{
  uint32_t total_pck_num = 0;
  string filename130000 = "data/0.dat";
  char tmp[13] = {0};
  FILE *file1 = fopen(filename130000.c_str(), "r");
  if (file1 == NULL)
  {
    printf("[ERROR] file not open\n");
    exit(0);
  }
  SRCIP_TUPLE key;
  int window = 0;
  if (!fread(tmp, 13, 1, file1)){
    printf("[ERROR] file error!\n");
    exit(0);
  }
  while (fread(tmp, 13, 1, file1))
  {
    memcpy(&key, tmp, 4);
    traces[window].push_back(key);
    total_pck_num++;
  }
  printf("[INFO] Scanned, packets number:\n");
  fclose(file1);
  int i = 0;
  printf("[INFO] window %02d has %ld packets\n", i, traces[i].size());
  return total_pck_num;
}

uint32_t ReadTraces()
{
  double starttime, nowtime;
  uint32_t total_pck_num = 0;
  string filename130000 = "data/0.dat";
  char tmp[21] = {0};
  FILE *file1 = fopen(filename130000.c_str(), "r");
  if (file1 == NULL)
  {
    printf("[ERROR] file not open\n");
    exit(0);
  }
  SRCIP_TUPLE key;
  int window = 0;
  if (!fread(tmp, 21, 1, file1)){
    printf("[ERROR] file error!\n");
    exit(0);
  }
  starttime = *(double *)(tmp + 13);
  while (fread(tmp, 21, 1, file1))
  {
    nowtime = *(double *)(tmp + 13);
    if (nowtime - starttime >= 5.0)
    {
      window++;
      starttime = nowtime;
    }
    memcpy(&key, tmp, 4);
    traces[window].push_back(key);
    total_pck_num++;
  }
  printf("[INFO] 12 windows scanned, packets number:\n");
  for (int i = 0; i < 12; i++)
    printf("[INFO] window %02d has %ld packets\n", i, traces[i].size());
  printf("\n\n");
  return total_pck_num;
}
/************************** PREDEFINED NUMBERS***********************/
#define HH_THRESHOLD 500 // 20,000,000 * 0.0005 (0.05%)
#define HC_THRESHOLD 250
#define TOT_MEM 450
/************************** COMMON FUNCTIONS*************************/
#define ROUND_2_INT(f) ((int)(f >= 0.0 ? (f + 0.5) : (f - 0.5)))

/********************************************************************/

// Fermat_tower
// #define TOT_MEMORY TOT_MEM * 1024 
#define TOT_MEMORY 500 * 1024 
#define ELE_BUCKET 2500
#define ELE_THRESHOLD 250
#define USE_FING 0
#define INIT ((uint32_t)random() % 800)
#define FERMAT_EM_ITER 15
/********************************************************************/

// FCM+TopK (16-ary)
// Here, we consider the actual hardware implementation on Tofino.
// The actual register size of each bucket in each Top-K entry is (8 * 3 + 4) = 28 Byte,
// which is composed of 1 val_all (4B) + 3 key-value pairs (4 + 4) = 28 Byte.

#define JUDGE_IF_SWAP_FCMPLUS_P4(min_val, guard_val) ((guard_val >> 5) >= min_val)
#define FCMPLUS_DEPTH 2     // number of trees
#define FCMPLUS_LEVEL 3     // number of layer in trees
#if TOT_MEM >= 200
#define FCMPLUS_BUCKET 3072 // 2^12, num of entries for key-value pairs
#elif TOT_MEM == 150
#define FCMPLUS_BUCKET 1536
#elif TOT_MEM == 100
#define FCMPLUS_BUCKET 1024
#elif TOT_MEM == 50
#define FCMPLUS_BUCKET 512
#elif TOT_MEM == 25
#define FCMPLUS_BUCKET 256
#elif TOT_MEM == 75
#define FCMPLUS_BUCKET 768
#elif TOT_MEM == 125
#define FCMPLUS_BUCKET 1280
#elif TOT_MEM == 175
#define FCMPLUS_BUCKET 1792
#endif
#define FCMPLUS_K_ARY 16    // k-ary tree

#if FCMPLUS_K_ARY == 2
#define FCMPLUS_K_POW 1 // 2^1 = 2
#elif FCMPLUS_K_ARY == 4
#define FCMPLUS_K_POW 2 // 2^2 = 4
#elif FCMPLUS_K_ARY == 8
#define FCMPLUS_K_POW 3 // 2^3 = 8
#elif FCMPLUS_K_ARY == 16
#define FCMPLUS_K_POW 4 // 2^4 = 16
#elif FCMPLUS_K_ARY == 32
#define FCMPLUS_K_POW 5 // 2^5 = 32
#endif

// Config using 1.25MB
#define FCMPLUS_HEAVY_STAGE 4
#if TOT_MEM == 1000
#define FCMPLUS_WL1 384000 // width of layer 1 (number of registers)
#define FCMPLUS_WL2 24000   // width of layer 2 (number of registers)
#define FCMPLUS_WL3 1500    // width of layer 3 (number of registers)
#elif TOT_MEM == 800
#define FCMPLUS_WL1 294400
#define FCMPLUS_WL2 18400
#define FCMPLUS_WL3 1150
// #elif TOT_MEM == 600
// #define FCMPLUS_WL1 204800
// #define FCMPLUS_WL2 12800
// #define FCMPLUS_WL3 800
// #elif TOT_MEM == 500
// #define FCMPLUS_WL1 160000
// #define FCMPLUS_WL2 10000
// #define FCMPLUS_WL3 625
// #elif TOT_MEM == 400
// #define FCMPLUS_WL1 115200
// #define FCMPLUS_WL2 7200
// #define FCMPLUS_WL3 450
// #elif TOT_MEM == 300
// #define FCMPLUS_WL1 70400
// #define FCMPLUS_WL2 4400
// #define FCMPLUS_WL3 275
// #elif TOT_MEM == 200
// #define FCMPLUS_WL1 25600
// #define FCMPLUS_WL2 1600
// #define FCMPLUS_WL3 100
#elif TOT_MEM == 200
#define FCMPLUS_WL1 25600
#define FCMPLUS_WL2 1600
#define FCMPLUS_WL3 100
#elif TOT_MEM == 225
#define FCMPLUS_WL1 36864
#define FCMPLUS_WL2 2304
#define FCMPLUS_WL3 144
#elif TOT_MEM == 250
#define FCMPLUS_WL1 48128
#define FCMPLUS_WL2 3008
#define FCMPLUS_WL3 188
#elif TOT_MEM == 275
#define FCMPLUS_WL1 59136
#define FCMPLUS_WL2 3696
#define FCMPLUS_WL3 231
#elif TOT_MEM == 300
#define FCMPLUS_WL1 70400
#define FCMPLUS_WL2 4400
#define FCMPLUS_WL3 275
#elif TOT_MEM == 325
#define FCMPLUS_WL1 81664
#define FCMPLUS_WL2 5104
#define FCMPLUS_WL3 319
#elif TOT_MEM == 350
#define FCMPLUS_WL1 92672
#define FCMPLUS_WL2 5792
#define FCMPLUS_WL3 362
#elif TOT_MEM == 375
#define FCMPLUS_WL1 103936
#define FCMPLUS_WL2 6496
#define FCMPLUS_WL3 406
#elif TOT_MEM == 400
#define FCMPLUS_WL1 115200
#define FCMPLUS_WL2 7200
#define FCMPLUS_WL3 450
#elif TOT_MEM == 425
#define FCMPLUS_WL1 126464
#define FCMPLUS_WL2 7904
#define FCMPLUS_WL3 494
#elif TOT_MEM == 450
#define FCMPLUS_WL1 137728
#define FCMPLUS_WL2 8608
#define FCMPLUS_WL3 538
#elif TOT_MEM == 475
#define FCMPLUS_WL1 148736
#define FCMPLUS_WL2 9296
#define FCMPLUS_WL3 581
#elif TOT_MEM == 500
#define FCMPLUS_WL1 160000
#define FCMPLUS_WL2 10000
#define FCMPLUS_WL3 625
#elif TOT_MEM == 525
#define FCMPLUS_WL1 171264
#define FCMPLUS_WL2 10704
#define FCMPLUS_WL3 669
#elif TOT_MEM == 550
#define FCMPLUS_WL1 182272
#define FCMPLUS_WL2 11392
#define FCMPLUS_WL3 712
#elif TOT_MEM == 575
#define FCMPLUS_WL1 193536
#define FCMPLUS_WL2 12096
#define FCMPLUS_WL3 756
#elif TOT_MEM == 600
#define FCMPLUS_WL1 204800
#define FCMPLUS_WL2 12800
#define FCMPLUS_WL3 800
#elif TOT_MEM == 625
#define FCMPLUS_WL1 216064
#define FCMPLUS_WL2 13504
#define FCMPLUS_WL3 844
#elif TOT_MEM == 650
#define FCMPLUS_WL1 227328
#define FCMPLUS_WL2 14208
#define FCMPLUS_WL3 888
#elif TOT_MEM == 675
#define FCMPLUS_WL1 238336
#define FCMPLUS_WL2 14896
#define FCMPLUS_WL3 931
#elif TOT_MEM == 700
#define FCMPLUS_WL1 249600
#define FCMPLUS_WL2 15600
#define FCMPLUS_WL3 975
#elif TOT_MEM == 725
#define FCMPLUS_WL1 260864
#define FCMPLUS_WL2 16304
#define FCMPLUS_WL3 1019
#elif TOT_MEM == 750
#define FCMPLUS_WL1 271872
#define FCMPLUS_WL2 16992
#define FCMPLUS_WL3 1062
#elif TOT_MEM == 775
#define FCMPLUS_WL1 283136
#define FCMPLUS_WL2 17696
#define FCMPLUS_WL3 1106
#elif TOT_MEM == 800
#define FCMPLUS_WL1 294400
#define FCMPLUS_WL2 18400
#define FCMPLUS_WL3 1150
#elif TOT_MEM == 825
#define FCMPLUS_WL1 305664
#define FCMPLUS_WL2 19104
#define FCMPLUS_WL3 1194
#elif TOT_MEM == 850
#define FCMPLUS_WL1 316928
#define FCMPLUS_WL2 19808
#define FCMPLUS_WL3 1238
#elif TOT_MEM == 875
#define FCMPLUS_WL1 327936
#define FCMPLUS_WL2 20496
#define FCMPLUS_WL3 1281
#elif TOT_MEM == 900
#define FCMPLUS_WL1 339200
#define FCMPLUS_WL2 21200
#define FCMPLUS_WL3 1325
#elif TOT_MEM == 925
#define FCMPLUS_WL1 350464
#define FCMPLUS_WL2 21904
#define FCMPLUS_WL3 1369
#elif TOT_MEM == 950
#define FCMPLUS_WL1 361472
#define FCMPLUS_WL2 22592
#define FCMPLUS_WL3 1412
#elif TOT_MEM == 975
#define FCMPLUS_WL1 372736
#define FCMPLUS_WL2 23296
#define FCMPLUS_WL3 1456
#elif TOT_MEM == 1000
#define FCMPLUS_WL1 384000
#define FCMPLUS_WL2 24000
#define FCMPLUS_WL3 1500
#elif TOT_MEM == 1025
#define FCMPLUS_WL1 395264
#define FCMPLUS_WL2 24704
#define FCMPLUS_WL3 1544
#elif TOT_MEM == 1050
#define FCMPLUS_WL1 406528
#define FCMPLUS_WL2 25408
#define FCMPLUS_WL3 1588
#elif TOT_MEM == 1075
#define FCMPLUS_WL1 417536
#define FCMPLUS_WL2 26096
#define FCMPLUS_WL3 1631
#elif TOT_MEM == 1100
#define FCMPLUS_WL1 428800
#define FCMPLUS_WL2 26800
#define FCMPLUS_WL3 1675
#elif TOT_MEM == 1125
#define FCMPLUS_WL1 440064
#define FCMPLUS_WL2 27504
#define FCMPLUS_WL3 1719
#elif TOT_MEM == 1150
#define FCMPLUS_WL1 451072
#define FCMPLUS_WL2 28192
#define FCMPLUS_WL3 1762
#elif TOT_MEM == 1175
#define FCMPLUS_WL1 462336
#define FCMPLUS_WL2 28896
#define FCMPLUS_WL3 1806
#elif TOT_MEM == 1200
#define FCMPLUS_WL1 473600
#define FCMPLUS_WL2 29600
#define FCMPLUS_WL3 1850
#elif TOT_MEM == 1225
#define FCMPLUS_WL1 484864
#define FCMPLUS_WL2 30304
#define FCMPLUS_WL3 1894
#elif TOT_MEM == 1250
#define FCMPLUS_WL1 496128
#define FCMPLUS_WL2 31008
#define FCMPLUS_WL3 1938
#elif TOT_MEM == 1275
#define FCMPLUS_WL1 507136
#define FCMPLUS_WL2 31696
#define FCMPLUS_WL3 1981
#elif TOT_MEM == 1300
#define FCMPLUS_WL1 518400
#define FCMPLUS_WL2 32400
#define FCMPLUS_WL3 2025
#elif TOT_MEM == 1325
#define FCMPLUS_WL1 529664
#define FCMPLUS_WL2 33104
#define FCMPLUS_WL3 2069
#elif TOT_MEM == 1350
#define FCMPLUS_WL1 540672
#define FCMPLUS_WL2 33792
#define FCMPLUS_WL3 2112
#elif TOT_MEM == 1375
#define FCMPLUS_WL1 551936
#define FCMPLUS_WL2 34496
#define FCMPLUS_WL3 2156
#elif TOT_MEM == 1400
#define FCMPLUS_WL1 563200
#define FCMPLUS_WL2 35200
#define FCMPLUS_WL3 2200

#elif TOT_MEM == 175
#define FCMPLUS_WL1 40960
#define FCMPLUS_WL2 2560
#define FCMPLUS_WL3 160
#elif TOT_MEM == 150
#define FCMPLUS_WL1 35072
#define FCMPLUS_WL2 2192
#define FCMPLUS_WL3 137
#elif TOT_MEM == 125
#define FCMPLUS_WL1 29184
#define FCMPLUS_WL2 1824
#define FCMPLUS_WL3 114
#elif TOT_MEM == 100
#define FCMPLUS_WL1 23296
#define FCMPLUS_WL2 1456
#define FCMPLUS_WL3 91
#elif TOT_MEM == 75
#define FCMPLUS_WL1 17664
#define FCMPLUS_WL2 1104
#define FCMPLUS_WL3 69
#elif TOT_MEM == 50
#define FCMPLUS_WL1 11776
#define FCMPLUS_WL2 736
#define FCMPLUS_WL3 46
#elif TOT_MEM == 25
#define FCMPLUS_WL1 6144
#define FCMPLUS_WL2 384
#define FCMPLUS_WL3 23
#endif

typedef uint8_t FCMPLUS_C1; // 8-bit
#define FCMPLUS_THL1 254
typedef uint16_t FCMPLUS_C2; // 16-bit
#define FCMPLUS_THL2 65534
typedef uint32_t FCMPLUS_C3; // 32-bit
#define FCMPLUS_EM_ITER 15   // Num.iteration of EM-Algorithm. You can control.
/********************************************************************/

#define ALPHA (TOT_MEM / 500.0)
#define ELASTIC_BUCKET int(3072 * ALPHA)
#define ELASTIC_HEAVY_STAGE (int(4*ALPHA)>0?int(4*ALPHA):1)
#define ELASTIC_WL TOT_MEM * 1024 - ELASTIC_BUCKET * ELASTIC_HEAVY_STAGE * 12
#define ELASTIC_TOFINO 0 
#define JUDGE_IF_SWAP_ELASTIC_P4(min_val, guard_val) ((guard_val >> 5) >= min_val)
#define ELASTIC_EM_ITER 15 // Num.iteration of EM-Algorithm. You can control.
#if defined _FCMPLUS_H_ || defined _ELASTIC_P4_H_
struct Bucket
{
 uint32_t key;
 uint32_t val;
 uint32_t guard_val;
};
#endif
/********************************************************************/

// Count-Min
#define CM_BYTES TOT_MEM * 1024 
#define CM_DEPTH 3       // depth of CM
/********************************************************************/

// MRAC
#define MRAC_BYTES TOT_MEM * 1024 
#define MRAC_EM_ITER 15   // Num.iteration of EM-Algorithm. You can control.
/********************************************************************/

// HYPERLOGLOG
#define HLL_B 20       
#define HLL_REG_SIZE 8 // 8-bit register size
/********************************************************************/

// CUSKETCH (Count-Min + Conservative Update scheme)
#define CU_BYTES TOT_MEM * 1024 
#define CU_DEPTH 3       // depth of CU
/********************************************************************/

// PyramidSketch + Count-Min (PCMSketch)
#define MAX_HASH_NUM 20
#define LOW_HASH_NUM 4 // depth of PCM
typedef long long lint;
typedef unsigned int uint;
#define PCM_BYTES 1572864 // 1.5 * 1024 * 1024 = 1.5MB
/********************************************************************/

// UnivMon
#define UNIV_BYTES TOT_MEM * 1024 // 1.5 * 1024 * 1024 = 1.5MB
#define UNIV_BYTES_HC TOT_MEM * 1024
#define UNIV_LEVEL 14
#define UNIV_K 1000
#define UNIV_ROW 5
//#define UNIV_BYTES 1048576

/********************************************************************/

// Sieving
#define SIEVING_MEM TOT_MEM * 1024

/********************************************************************/
#endif
