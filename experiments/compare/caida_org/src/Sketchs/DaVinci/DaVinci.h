#pragma once
#include <chrono>
#include <map>
#include "HeavyPart.h"
#include "../common/EMFSD.h"
#include "tower.h"
#include "fermat.h"
#include "../../common_func.h"

#define TOT_MEMORY 500 * 1024 
#define ELE_BUCKET 2500
#define ELE_THRESHOLD 250
#define USE_FING 0
#define INIT ((uint32_t)random() % 800)
#define TOWER_EM_ITER 10


#define HEAVY_MEM_ (150 * 1024)
#define BUCKET_NUM_ (HEAVY_MEM_ / 64)

using namespace std;

void cardPrintSketchInfo(int totalMem, int _heavypartBucketNum, int towerWidth, int array_num, int entry_num, int _fermatcount, int usefing, int _init) {
    const int width = 55;
    const std::string line(width, '-');
    const std::string space(width , ' ');

    std::cout << "╭" << line << "╮" << std::endl;
    std::cout << "│" << space << "│" << std::endl;
    
    std::string title = "Fermat_Count_IDP_CNTPM version";
    int titleLength = title.length();
    int titlePadding = (width - titleLength) / 2;
    std::cout << "│" << std::string(titlePadding, ' ') << title << std::string(width - titlePadding - titleLength, ' ') << "│" << std::endl;
    
    std::cout << "│" << space << "│" << std::endl;
    std::cout << "│   Parameters:" << std::string(width - 14, ' ') << "│" << std::endl;
    
    std::printf("│     Total Memory        = %-7d%*s│\n", totalMem, width - 34, "");
    std::printf("│     HeavypartBucketNum  = %-7d%*s│\n", _heavypartBucketNum, width - 34, "");
    std::printf("│     towerWidth_2bit     = %-7d%*s│\n", towerWidth, width - 34, "");
    std::printf("│     towerWidth_4bit     = %-7d%*s│\n", towerWidth/2, width - 34, "");
    std::printf("│     towerWidth_8bit     = %-7d%*s│\n", towerWidth/4, width - 34, "");
    std::printf("│     fermat_array_num    = %-7d%*s│\n", array_num, width - 34, "");
    std::printf("│     fermat_entry_num    = %-7d%*s│\n", entry_num, width - 34, "");
    std::printf("│     Fermat Type         = %-7d%*s│\n", _fermatcount, width - 34, "");
    std::printf("│     usefing             = %-7d%*s│\n", usefing, width - 34, "");
    std::printf("│     _init               = %-7d%*s│\n", _init, width - 34, "");
    
    std::cout << "│" << space << "│" << std::endl;
    std::cout << "╰" << line << "╯" << std::endl;
}
template<int _bucket_num>
class DaVinci
{
    int light_array_num;
    int light_entry_num;
    
public:
    bool have_decoded = false;
    bool have_got_all_result = false;
    int tot_memory;
    int tot_packets;
    int fermatEleMem;
    int towerfilterMem;
    bool ifFermatCount;
    EMFSD *em_tower = NULL;
    uint32_t memory_access_counter; 
    double last_tower_copy_ms = 0.0;
    double last_tower_em_ms = 0.0;
    double last_postprocess_ms = 0.0;
    double last_get_distribution_ms = 0.0;
    int last_tower_width = 0;
    int last_tower_nonzero = 0;
    int last_tower_max_counter = 0;
    int last_tower_counter_bits = 0;
    int last_tower_cap_counter = 0;
    int last_tower_mid_width = 0;
    int last_tower_mid_nonzero = 0;
    int last_tower_mid_cap_counter = 0;

    int heavy_mem = _bucket_num * COUNTER_PER_BUCKET * 8; //this is fake, only to satisfy "template" but not really used
    int heavy_bucket_num; // This is real

    static constexpr int bucket_num = _bucket_num;
    HeavyPart<bucket_num> *heavy_part;
    TowerSketch *tower;
    Fermat *fermatEle;

    //for test track
    unordered_map<int32_t, int> Eleresult;
    unordered_map<int32_t, int> allResult;
    unordered_map<int32_t, vector<pair<int, int>>> insert_tracking;
    unordered_map<int32_t, vector<int>> decode_track;

    // (int _tot_memory = TOT_MEMORY, int _fermatEleMem = 3 * 5*ELE_BUCKET * (8 + 4 * USE_FING), int _heavypartBucketNum = BUCKET_NUM_, 
    //     int _towerMem = TOT_MEMORY - 3 * 2*ELE_BUCKET * (8 + 4 * USE_FING) - HEAVY_MEM_, int _fermatcount = 3, 
    //              bool usefing = USE_FING, uint32_t _init = INIT)
public:

    DaVinci(int _tot_memory = TOT_MEMORY, int _fermatEleMem = 3 * 2 * ELE_BUCKET * (6 + 4 * USE_FING) , int _heavypartBucketNum = 0.8*BUCKET_NUM_, 
        int _towerMem = TOT_MEMORY - 3 * 2 * ELE_BUCKET * (6 + 4 * USE_FING) - 0.8*HEAVY_MEM_, int _fermatcount = 3, 
                 bool usefing = USE_FING, uint32_t _init = 37, bool union_task = 0, int towertype = CM) : fermatEleMem(_fermatEleMem)
    {
        printf("You are running DaVinci initiated by memory.\n");
        printf("parameters: _heavypartBucketNum = %d, _towerMem = %d, _fermatEleMem = %d, _fermatcount = %d, usefing = %d, _init = %d\n", _heavypartBucketNum, _towerMem, _fermatEleMem, _fermatcount, usefing, _init);
        heavy_bucket_num = _heavypartBucketNum;
        towerfilterMem = _towerMem;
        tot_packets = 0;
        // fermatEle = new Fermat(fermatEleMem, usefing, _init)
        if(_fermatcount == 1){
            fermatEle = new Fermat_Count(fermatEleMem, usefing, _init);
        }
        else if(_fermatcount == 2){
            cout << "Running Fermat_Count_IDP_CNTPM" << endl;
            fermatEle = new Fermat_Count_IDP_CNTPM(fermatEleMem, usefing, _init);
        }
        else if(_fermatcount == 3){
            cout << "Running Fermat_Count_IDP_CNTPM_48bits" << endl;
            fermatEle = new Fermat_Count_IDP_CNTPM_48bits(fermatEleMem, usefing, _init);
        }
        else{
            fermatEle = new Fermat_Sketch(fermatEleMem, usefing, _init);
        }
        heavy_part = new HeavyPart<bucket_num>(_heavypartBucketNum);
        if(!union_task)
            tower = new TowerSketch(_towerMem, CM, 15, _init);
        else{
            tower = new TowerSketch(_towerMem, CM, 7, _init);
        }
        light_array_num = fermatEle->get_array_num();
        light_entry_num = fermatEle->get_entry_num();

        cardPrintSketchInfo(_tot_memory, heavy_part->get_bucket_num(), _towerMem*4, fermatEle->get_array_num(), fermatEle->get_entry_num(), _fermatcount, usefing, _init);

    }
    DaVinci(int _heavypartBucketNum, int array_num, int entry_num, int _fermatcount = 2, 
                 bool usefing = USE_FING, uint32_t _init = INIT) //TODO: Update to DaVinci
    {
        printf("You are running DaVinci. ");
        printf("Parameters: _heavypartBucketNum = %d, array_num = %d, entry_num = %d, _fermatcount = %d, usefing = %d, _init = %d\n", _heavypartBucketNum, array_num, entry_num, _fermatcount, usefing, _init);
        cardPrintSketchInfo(TOT_MEMORY, _heavypartBucketNum, 44444, array_num, entry_num, _fermatcount, usefing, _init);
        heavy_bucket_num = _heavypartBucketNum;
        light_array_num = array_num;
        light_entry_num = entry_num;

        ifFermatCount =  _fermatcount;
        tot_packets = 0;
        if(_fermatcount == 1){
            fermatEle = new Fermat_Count(array_num, entry_num, usefing, _init);
        }
        else if(_fermatcount == 2){
            fermatEle = new Fermat_Count_IDP_CNTPM(array_num, entry_num, usefing, _init);
        }
        else{
            fermatEle = new Fermat_Sketch(array_num, entry_num, usefing, _init);
        }
        heavy_part = new HeavyPart<bucket_num>(_heavypartBucketNum);
        tower = new TowerSketch(TOT_MEMORY - 3 * ELE_BUCKET * (8 + 4 * USE_FING) - HEAVY_MEM_, CM, 15, MY_RANDOM_SEED);
        
    }
    void insert_after_heavy(const char *key, int f = 1){
        uint32_t checking_id = 0;
        for(int i=1;i<=f;i++){
            // cout << "Inserting into tower: " << i << "/ " << swap_val << " times" << endl;
            if(!tower->insert((char*)(key))){
                // cout << "Insert into tower failed!" << endl;
                int remain_val = GetCounterVal(f) - i + 1;
                if(*(uint32_t *)key == checking_id){
                    cout << "Insert into tower failed! remain_val is " << remain_val << endl;
                }

                fermatEle->Insert(*(uint32_t*) key, remain_val); 
                break;
            }
        }
    }
    void insert(const char *key, int f = 1){
        //heavy part
        uint32_t checking_id = 1452808729;
        uint8_t swap_key[KEY_LENGTH_4];
        uint32_t swap_val = 0;
        memory_access_counter = 0;  // 每次插入前重置计数器
        //tracking 
        int result = heavy_part->insert((uint8_t *)key, swap_key, swap_val, f);
        memory_access_counter = heavy_part->hp_memory_access_counter;
        if(*(uint32_t *)key == checking_id){
            cout << "Inserting " << checking_id << " into heavy part! result is" << result << endl;
        }
        uint32_t keysfing = *(uint32_t *)key;
        if(result == 1) { // Swap out entry
            keysfing = *(uint32_t *)swap_key;
            if(*(uint32_t *)swap_key == checking_id){
                cout << "Swap out " << checking_id << " from heavy part! value = " << swap_val << endl;
            }
        }
        else if(result == 2){
            swap_val = 1;
            memcpy(swap_key, key, KEY_LENGTH_4);
        }

        // cout << "Start to insert into tower! result == " << result << endl;
        swap_val = GetCounterVal(swap_val);
        for(int i=1;i<=swap_val;i++){
            // cout << "Inserting into tower: " << i << "/ " << swap_val << " times" << endl;
            if(!tower->insert((char*)(swap_key))){
                memory_access_counter += 2;
                // cout << "Insert into tower failed!" << endl;
                int remain_val = GetCounterVal(swap_val) - i + 1;
                if(*(uint32_t *)swap_key == checking_id){
                    cout << "Insert into tower failed! remain_val is " << remain_val << endl;
                }

                pair<int, int> valuePair = std::make_pair(result, GetCounterVal(remain_val));
                insert_tracking[keysfing].push_back(valuePair);

                // Fermat part
                switch(result)
                {
                    case 0: break; // Inserted into the heavy part and nothing to do with the light part
                    case 1: fermatEle->Insert(*(uint32_t*) swap_key, GetCounterVal(remain_val)); memory_access_counter += 3; break;//
                    case 2: fermatEle->Insert(*(uint32_t*) key, 1); memory_access_counter += 3; break;
                    default:
                        printf("error return value !\n");
                        exit(1);
                }
                break;
            }
        }
        tot_packets++;
    }

    void get_distribution(vector<double> &dist, int index = 0) { //need decoed first
        auto get_start = std::chrono::high_resolution_clock::now();

        if (em_tower != NULL)
            delete em_tower;
        em_tower = new EMFSD;
        uint32_t *countercpy;
        countercpy = new uint32_t[tower->line[2].width];
        last_tower_width = this->tower->line[2].width;
        last_tower_nonzero = 0;
        last_tower_max_counter = 0;
        last_tower_counter_bits = this->tower->line[2].counter_w;
        last_tower_cap_counter = 0;
        last_tower_mid_width = this->tower->line[1].width;
        last_tower_mid_nonzero = 0;
        last_tower_mid_cap_counter = 0;
        int tower_cap = (1 << this->tower->line[2].counter_w) - 1;
        int mid_cap = (1 << this->tower->line[1].counter_w) - 1;
        for (int i = 0; i < this->tower->line[1].width; i++)
        {
            int mid_value = tower->line[1].index(i);
            if (mid_value != 0)
                last_tower_mid_nonzero++;
            if (mid_value == mid_cap)
                last_tower_mid_cap_counter++;
        }
        for (int i = 0; i < this->tower->line[2].width; i++)
        {
            countercpy[i] = tower->line[2].index(i);
            if (countercpy[i] != 0) {
                last_tower_nonzero++;
                if ((int)countercpy[i] > last_tower_max_counter)
                    last_tower_max_counter = (int)countercpy[i];
            }
            if ((int)countercpy[i] == tower_cap)
                last_tower_cap_counter++;
        }
        em_tower->set_counters(this->tower->line[2].width, countercpy);
        auto em_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < TOWER_EM_ITER; i++)
        {
            em_tower->next_epoch();
        }
        auto post_start = std::chrono::high_resolution_clock::now();
        dist.resize(em_tower->ns.size());
        for (int i = 1; i < em_tower->ns.size(); i++)
        {
            dist[i] = em_tower->ns[i];
        }

        int tower_max = tower->maximum;
        for(auto i : Eleresult){
            if(dist.size() <= abs(i.second)+tower_max){
                dist.resize(abs(i.second) + tower_max + 1);

            }
            dist[abs(i.second)+tower_max]++;
            if(dist[tower_max])
                dist[tower_max]-=1;
        }
        
        for(int i = 0; i < heavy_bucket_num; ++i)
            for(int j = 0; j < MAX_VALID_COUNTER; ++j) {
                uint8_t key[KEY_LENGTH_4];
                *(uint32_t*)key = heavy_part->buckets[i].key[j];
                int val = heavy_part->buckets[i].val[j];
                int tower_val = 0;
                int fermat_val = 0;
                if(HIGHEST_BIT_IS_1(val)){
                    tower_val = tower->query((char*)key);
                    if(Eleresult.count(*(uint32_t*)key))
                        fermat_val = Eleresult[*(uint32_t*)key];
                }
                if(HIGHEST_BIT_IS_1(val) && fermat_val + tower_val != 0) {
                    val += (fermat_val + tower_val);
                    dist[fermat_val + tower_val]--;
                }
                val = GetCounterVal(val);
                if(val) {
                    if(val + 1 > dist.size())
                        dist.resize(val + 1);
                    dist[val]++;
                }
            }
        
        delete[] countercpy;
        auto get_end = std::chrono::high_resolution_clock::now();
        last_tower_copy_ms = std::chrono::duration<double, std::milli>(em_start - get_start).count();
        last_tower_em_ms = std::chrono::duration<double, std::milli>(post_start - em_start).count();
        last_postprocess_ms = std::chrono::duration<double, std::milli>(get_end - post_start).count();
        last_get_distribution_ms = std::chrono::duration<double, std::milli>(get_end - get_start).count();
    }
    int decode(bool use_united = 0)
    {
        // printf("Decoding...... Eleresult.size() = %d\n", Eleresult.size());
        // 创建 DataVariant 类型的实例
        cout << "Size of insertedflow is " << fermatEle->insertedflows.size() << endl;
        DataVariant variantEleresult = Eleresult;

        // 将 variantEleresult 传递给 Decode 函数
        if(use_united){
            cout << "United decoding!" << endl;
            if (fermatEle->united_decode(variantEleresult, tower)) 
                printf("United decode Successfully!\n");
            else
                printf("Decode Fail!\n");
        }
        else{
            if (fermatEle->Decode(variantEleresult)) 
                printf("Decode Successfully!\n");
            else
                printf("Decode Fail!\n");
        }
        Eleresult = std::get<std::unordered_map<int, int>>(variantEleresult);

        printf("Eleresult: %lu\n", Eleresult.size());
        printf("Lightpart-inserted num: %lu\n", fermatEle->insertedflows.size());
        printf("Decoded rate: %f\n", (double)Eleresult.size() / fermatEle->insertedflows.size());
        
        printf("-----------------------------------------------------------------------------\n");
        have_decoded = true;
        return Eleresult.size();
    }
    uint32_t query(const char *key, bool add_undecoded = 1, bool ifprint = 0)
    {
        uint32_t checking_id = 0;
        uint32_t hp_cnt = heavy_part->query((uint8_t *)key);
        uint32_t id = *(uint32_t*) key;
        uint32_t checked_id = 0;

        if(id == checking_id){
            cout << checking_id << "'s heavy_part query result: " << hp_cnt << endl;
        }

        if(hp_cnt == 0 || HIGHEST_BIT_IS_1(hp_cnt))
        {      
            if (Eleresult.count(*(uint32_t *)key))
            {
                if(id == checking_id){
                    cout << checking_id << " exists in Eleresult! value is " << Eleresult[*(uint32_t *)key] << endl;
                }
                decode_track[*(uint32_t *)key] = vector<int>{(int)GetCounterVal(hp_cnt), tower->maximum, Eleresult[*(uint32_t *)key], 0};
                return (int)GetCounterVal(hp_cnt) + Eleresult[*(uint32_t *)key] + tower->maximum;

            }
            // else if(add_undecoded){
            //     int cm_query = fermatEle->undecoded_query(key);
            //     if(checked_id && id == checked_id){
            //         if(fermatEle->insertedflows.find(id) != fermatEle->insertedflows.end()){
            //             cout << checked_id << " exists in insertedflows!" << endl;
            //         }
            //         else{
            //             cout << checked_id << " does not exist in insertedflows!" << endl;
            //         }
            //     }
            //     decode_track[*(uint32_t *)key] = vector<int>{(int)GetCounterVal(hp_cnt), 0, cm_query};
            //     return (int)GetCounterVal(hp_cnt) + cm_query;
            // }
            else{
                int tower_est = tower->query((char *)key);
                if(id == checking_id){
                    cout << checking_id << " does not exist in Eleresult! tower value is " << tower_est << endl;
                }
                decode_track[*(uint32_t *)key] = vector<int>{(int)GetCounterVal(hp_cnt), tower_est, 0, 0};
                return (int)GetCounterVal(hp_cnt) + tower_est;
            }
        }
        if(id == checked_id){
            cout << checked_id << " is not in Eleresult!" << endl;
        }
        decode_track[*(uint32_t *)key] = vector<int>{(int)GetCounterVal(hp_cnt),0, 0, 0};
        return (int)GetCounterVal(hp_cnt);
    }

    uint32_t query_only_light_part(const char *key, bool add_undecoded = 1)
    {
        uint32_t hp_cnt = heavy_part->query((uint8_t *)key);
        uint32_t id = *(uint32_t*) key;
        uint32_t checked_id = 4153222982;

        if(hp_cnt == 0 || HIGHEST_BIT_IS_1(hp_cnt)){
            if (Eleresult.count(*(uint32_t *)key))
            {
                if(id == checked_id){
                    cout << checked_id << " exists in Eleresult! value is " << Eleresult[*(uint32_t *)key] << endl;
                }
                decode_track[*(uint32_t *)key] = vector<int>{(int)GetCounterVal(hp_cnt), Eleresult[*(uint32_t *)key], 0};
                return Eleresult[*(uint32_t *)key];
            }
            else if(add_undecoded){
                int cm_query = fermatEle->undecoded_query(key);
                if(id == checked_id){
                    if(fermatEle->insertedflows.find(id) != fermatEle->insertedflows.end()){
                        cout << checked_id << " exists in insertedflows!" << endl;
                    }
                    else{
                        cout << checked_id << " does not exist in insertedflows!" << endl;
                    }
                    cout << "cm_query is " << cm_query << endl;
                }
                decode_track[*(uint32_t *)key] = vector<int>{(int)GetCounterVal(hp_cnt), 0, cm_query};
                return cm_query;
            }
        }
        if(id == checked_id){
            cout << checked_id << " is not in Eleresult! query_only_light_part return 0" << endl;
        }
        decode_track[*(uint32_t *)key] = vector<int>{(int)GetCounterVal(hp_cnt), 0, 0};
        return 0;
    }
    double get_entropy(vector<double> &distribution) //Must be used after get_distribution
    {
        double entropy = 0.0;
        double tot = 0.0;
        double entr = 0.0;
        for (int i = 1; i < distribution.size(); i++)
        {
            if(distribution[i] < 1.0)
                continue;
            tot += i * (int)distribution[i];
            entr += i * distribution[i] * log2(i);
        }
        entropy = -entr / tot + log2(tot);
        return entropy;
    }

    int get_heavy_bucket_num(){
        return heavy_bucket_num;
    }
    int get_light_array_num(){
        return light_array_num;
    }
    int get_light_entry_num(){
        return light_entry_num;
    }

    bool print_sketch(){
        printf("Heavy Part:\n");
        for(int i = 0; i < heavy_bucket_num; ++i){
            printf("Bucket %d: ", i);
            for(int j = 0; j < MAX_VALID_COUNTER; ++j){
                printf("%d ", GetCounterVal(heavy_part->buckets[i].val[j]));
            }
            printf("\n");
        }
        printf("Light Part:\n");
        for(int i = 0; i < light_array_num; ++i){
            printf("Array %d: ", i);
            for(int j = 0; j < light_entry_num; ++j){
                printf("(%d, %d)", fermatEle->get_id(i,j), fermatEle->get_counter(i,j));
            }
            printf("\n");
        }
        return true;
    
    }

    bool write2file(char* filename){
        string heavyFilename = "./outputs/heavy_" + string(filename);
        string lightFilename = "./outputs/light_" + string(filename);
        FILE *fp = fopen(heavyFilename.c_str(), "w");
        if(fp == NULL){
            printf("Open file failed!\n");
            return false;
        }
        //Heavy part
        for(int i = 0; i < heavy_bucket_num; ++i){
            for(int j = 0; j < MAX_VALID_COUNTER; ++j){
                fprintf(fp, "(%u, %d)", heavy_part->buckets[i].key[j], GetCounterVal(heavy_part->buckets[i].val[j]));
            }
            fprintf(fp, "\n");
        }
        fclose(fp);
        //Light part
        FILE *fp2 = fopen(lightFilename.c_str(), "w");
        for(int j = 0; j < light_entry_num; ++j){
            bool all_zero_flag = 1;
            for(int i = 0; i < light_array_num; ++i){
                if(fermatEle->get_id(i,j) != 0){
                    all_zero_flag = 0;
                    break;
                }
            }
            if(!all_zero_flag){
                fprintf(fp2, "%d: ", j);
                for(int i = 0; i < light_array_num; ++i){
                    uint32_t id = fermatEle->get_id(i,j);
                    int counter = fermatEle->get_counter(i,j);
                    fprintf(fp2, "(%u, %d) ", id, counter);
                }
                fprintf(fp2, "\n");
            }
        }
        fclose(fp2);
        return true;
    }

    void get_all_heavy_parts(set<int32_t>& hp) {
        assert(have_decoded);
        for(int i = 0; i < heavy_bucket_num; i++){
            for(int j = 0; j < MAX_VALID_COUNTER; j++){
                if(heavy_part->buckets[i].key[j] != 0){
                    hp.insert(heavy_part->buckets[i].key[j]);
                }
            }
        }
    }

    void get_all_results(){ //Must be used after decoding
        assert(have_decoded);
        for(int i = 0; i < heavy_bucket_num; i++){
            for(int j = 0; j < MAX_VALID_COUNTER; j++){
                if(heavy_part->buckets[i].key[j] != 0){
                    if(allResult.count(heavy_part->buckets[i].key[j]) == 0){
                        allResult[heavy_part->buckets[i].key[j]] = GetCounterVal(heavy_part->buckets[i].val[j]);
                        if(HIGHEST_BIT_IS_1(heavy_part->buckets[i].val[j])){
                            int tower_est = tower->query((const char *)&heavy_part->buckets[i].key[j]);
                            if(tower_est == tower->maximum && Eleresult.count(heavy_part->buckets[i].key[j]) > 0){
                                allResult[heavy_part->buckets[i].key[j]] += Eleresult[heavy_part->buckets[i].key[j]] + tower_est;
                            }
                            else{
                                allResult[heavy_part->buckets[i].key[j]] += tower_est;
                            }
                        }
                    }
                    else{
                        assert(0);
                    }
                }
            }
        }

        for(auto i:Eleresult){
            if(allResult.count(i.first) == 0){
                allResult[i.first] = i.second + tower->maximum;
            }
        }
        have_got_all_result = 1;
    }


    void get_heavy_hitters(set<uint32_t> &hh){
        get_all_results();
        for (auto i : allResult)
        {
            if (i.second >= HH_THRESHOLD)
                hh.insert(i.first);
        }
    }

    void get_heavy_hitters(set<uint32_t> &hh, int32_t k){
        get_all_results();
        vector<pair<uint32_t, int>> freqs;
        for (auto i : allResult)
        {
            freqs.push_back(i);
        }
        sort(freqs.begin(), freqs.end(), [&](auto const& p, auto const& q) {
            if (p.second != q.second)
                return p.second > q.second;
            else
                return p.first < q.first;
        });
        if (k + 1 < freqs.size() && freqs[k - 1].second == freqs[k].second) {
            cout << "cannot find exatly " << k << "heavy hitters!" << endl;
            exit(0);
        }
        for (int i = 0; i < k; ++i) {
            hh.insert(freqs[i].first);
        }
    }

    int get_cardinality(){
        int card = tower->get_cardinality();
        for(int i = 0; i < heavy_bucket_num; ++i)
            for(int j = 0; j < MAX_VALID_COUNTER; ++j)
            {
                uint8_t key[KEY_LENGTH_4];
                *(uint32_t*)key = heavy_part->buckets[i].key[j];
                int val = heavy_part->buckets[i].val[j];

                if(HIGHEST_BIT_IS_1(val))
                {
                    card--;
                }
                if(GetCounterVal(val))
                    card++;
            }
        return card;
    }

    ~DaVinci()
    {
        delete heavy_part;
        delete fermatEle;
        delete tower;
    }
};

template<int bucket_num>
bool compareLightpart(DaVinci<bucket_num> &sketch1, DaVinci<bucket_num> &sketch2){
    int array_num_1 = sketch1.get_light_array_num();
    int array_num_2 = sketch2.get_light_array_num();
    int entry_num_1 = sketch1.get_light_entry_num();
    int entry_num_2 = sketch2.get_light_entry_num();
    bool flag = 1;
    if(array_num_1 != array_num_2){
        printf("Fermat array num is different!\n");
        return false;
    }
    if(entry_num_1 != entry_num_2){
        printf("Fermat entry num is different!\n");
        return false;
    }
    int array_num = array_num_1;
    int entry_num = entry_num_1;
    for(int i = 0; i < array_num; ++i){
        for(int j = 0; j < entry_num; ++j){
            if(sketch1.fermatEle->get_id(i,j) != sketch2.fermatEle->get_id(i,j)){
                printf("ID at (%d, %d) is different, %d, %d\n", i, j, sketch1.fermatEle->get_id(i,j), sketch2.fermatEle->get_id(i,j));
                // return false;
                flag = 0;
            }
            if(sketch1.fermatEle->get_counter(i,j) != sketch2.fermatEle->get_counter(i,j)){
                printf("Counter at (%d, %d) is different, %d, %d\n", i, j, sketch1.fermatEle->get_counter(i,j), sketch2.fermatEle->get_counter(i,j));
                // return false;
                flag = 0;
            }
        }
    }
    printf("Light parts are the same!\n");
    return true;
}


//Check whether the two sketches are the same in Heavy_part size, light part array num and entry num
template<int bucket_num>
bool check_sketches_same_size(DaVinci<bucket_num> &sketch1, DaVinci<bucket_num> &sketch2)
{
    if(sketch1.get_heavy_bucket_num() != sketch2.get_heavy_bucket_num()){
        printf("Heavy part size is different!\n");
        return false;
    }
    if(sketch1.get_light_array_num() != sketch2.get_light_array_num()){
        printf("Fermat array num is different!\n");
        return false;
    }
    if(sketch1.get_light_entry_num() != sketch2.get_light_entry_num()){
        printf("Fermat entry num is different!\n");
        return false;
    }
    return true;
}

template<int bucket_num>
void Union(DaVinci<bucket_num> &sketch1, DaVinci<bucket_num> &sketch2, DaVinci<bucket_num>& sketch3, uint32_t init_seed = 813)
{
    //Check whether the two sketches are the same in size
    if(!check_sketches_same_size(sketch1, sketch2)){
        printf("Sketches are not the same size!\n");
        exit(1);
    }
    int heavy_bucket_num = sketch1.get_heavy_bucket_num();
    int array_num = sketch1.get_light_array_num();
    int entry_num = sketch1.get_light_entry_num();
    //Heavy part
    alignas(64) Bucket* sketch1_buckets = sketch1.heavy_part->buckets;
    alignas(64) Bucket* sketch2_buckets = sketch2.heavy_part->buckets;
    alignas(64) Bucket* sketch3_buckets = sketch3.heavy_part->buckets;

    bool full1=1, full2=1;
    map<uint32_t, bool> key_sign_map;
    map<uint32_t, bool> key_sign_map_1;
    map<uint32_t, bool> key_sign_map_2;

    vector<uint32_t> kickout_keys;
    vector<uint32_t> kickout_vals;

    int print_key = 4063245528;
    for(int i = 0; i < heavy_bucket_num; ++i){
        int total_keys_num = 0;
        map<uint32_t, uint32_t> merged_keys_vals;
        
        // Merge keys and values from both buckets
        for(int j = 0; j < MAX_VALID_COUNTER; ++j){
            uint32_t key = sketch1_buckets[i].key[j];
            uint32_t originalVal = sketch1_buckets[i].val[j];
            uint32_t val = GetCounterVal(originalVal);
            if(key != 0){
                if(HIGHEST_BIT_IS_1(originalVal)){
                    key_sign_map[key] = 1;
                    key_sign_map_1[key] = 1;
                }
                else{
                    key_sign_map[key] = -1;
                    key_sign_map_1[key] = -1;
                }
                merged_keys_vals[key] += val;
            }else{
                full1 = 0;
            }

            uint32_t key2 = sketch2_buckets[i].key[j];
            uint32_t originalVal2 = sketch2_buckets[i].val[j];
            uint32_t val2 = GetCounterVal(originalVal2);
            if(key2 != 0){
                if(HIGHEST_BIT_IS_1(originalVal2)){
                    key_sign_map[key2] = 1;
                    key_sign_map_2[key2] = 1;
                }
                else{
                    key_sign_map[key2] = -1;
                    key_sign_map_2[key2] = -1;
                }
                merged_keys_vals[key2] += val2;
            }else{
                full2 = 0;
            }
        }

        total_keys_num = merged_keys_vals.size();

        vector<pair<uint32_t, uint32_t>> sorted_merged_keys_vals_vec(merged_keys_vals.begin(), merged_keys_vals.end());
        if(total_keys_num > 7){
            //sort the merged keys and values based on value
            sort(sorted_merged_keys_vals_vec.begin(), sorted_merged_keys_vals_vec.end(), [](const pair<uint32_t, uint32_t> &a, const pair<uint32_t, uint32_t> &b){
                return a.second < b.second;
            });
        
            // Remove the smallest values if there are more than 7 different keys
            for(int k = 0; k <  sorted_merged_keys_vals_vec.size() - 7; ++k){
                kickout_keys.push_back(sorted_merged_keys_vals_vec[k].first);
                kickout_vals.push_back(sorted_merged_keys_vals_vec[k].second);
            }
        }
        
        // Store the remaining keys and values in the new sketch
        int start_index = max(static_cast<int>(sorted_merged_keys_vals_vec.size()) - 7, 0);
        for(int k = start_index; k < sorted_merged_keys_vals_vec.size(); ++k){
            uint32_t key = sorted_merged_keys_vals_vec[k].first;
            uint32_t val = sorted_merged_keys_vals_vec[k].second;
            sketch3_buckets[i].key[k - start_index] = key;
            bool sure_1_not_set = (key_sign_map_1.count(key) == 0 && (!full1)) || (key_sign_map_1.count(key) > 0 && key_sign_map_1[key] == -1);
            bool sure_2_not_set = (key_sign_map_2.count(key) == 0 && (!full2)) || (key_sign_map_2.count(key) > 0 && key_sign_map_2[key] == -1);
            if((sure_1_not_set)&&(sure_2_not_set)){
                sketch3_buckets[i].val[k - start_index] = val;
            }
            else{
                sketch3_buckets[i].val[k - start_index] = val | 0x80000000;
            }
            // sketch3_buckets[i].val[k - start_index] = val;
        }
            
    }

    //Tower union

    int width0 = sketch3.tower->line[0].width;
    for(int i = 0; i < width0; ++i){
        sketch3.tower->add_val(0, i, (sketch1.tower->line[0].index(i) + sketch2.tower->line[0].index(i))%3);
    }
    int width1 = sketch3.tower->line[1].width;
    for(int i = 0; i < width1; ++i){
        sketch3.tower->add_val(1, i, sketch1.tower->line[1].index(i) + sketch2.tower->line[1].index(i));
    }


    // Fermat Union
    for(int i = 0; i < array_num; ++i){
        for(int j = 0; j < entry_num; ++j){
            uint32_t sketch3_id = ((uint64_t)(uint32_t)(sketch1.fermatEle->get_id(i,j)) + (uint64_t)(uint32_t)(sketch2.fermatEle->get_id(i,j))) % (uint64_t)PRIME_ID_IDP_CNTPM;
            int32_t sketch3_counter = sketch1.fermatEle->get_counter(i,j) + sketch2.fermatEle->get_counter(i,j);
            sketch3.fermatEle->set_id(i, j, sketch3_id);
            sketch3.fermatEle->set_counter(i, j, sketch3_counter);
            uint32_t checking = 3458834590;
            if(sketch1.fermatEle->get_id(i,j) == checking){
                cout << "Checking situation of " << checking << ":" << endl;
                cout << "sketch1: id = " << (uint32_t)(sketch1.fermatEle->get_id(i,j)) << ", counter = " << sketch1.fermatEle->get_counter(i,j) << endl;
                cout << "calculated id = " << sketch3_id << ", counter = " << sketch3_counter << endl;
                cout << "sketch2: id = " << (uint32_t)(sketch2.fermatEle->get_id(i,j)) << ", counter = " << sketch2.fermatEle->get_counter(i,j) << endl;
                cout << "sketch3: id = " << (uint32_t)sketch3_id << ", counter = " << sketch3_counter << endl;
            }

        }
    }
    //kick out
    for(int i = 0; i < kickout_keys.size(); ++i){
        uint32_t key = kickout_keys[i];
        uint32_t val = kickout_vals[i];
        sketch3.insert_after_heavy((char*)&key, val);
    }
    //get total num of keys in different buckets
}

template<int bucket_num>
void Difference(DaVinci<bucket_num> &sketch1, DaVinci<bucket_num> &sketch2, DaVinci<bucket_num> &sketch3, uint32_t init_seed = 37)
{
    // Check whether the two sketches are the same in size
    if (!check_sketches_same_size(sketch1, sketch2))
    {
        printf("Sketches are not the same size!\n");
        exit(1);
    }

    int heavy_bucket_num = sketch1.get_heavy_bucket_num();
    int array_num = sketch1.get_light_array_num();
    int entry_num = sketch1.get_light_entry_num();

    // DaVinci<bucket_num> sketch3(heavy_bucket_num, array_num, entry_num, 2, 0, init_seed);
    printf("info about sketch3: heavy_bucket_num = %d, array_num = %d, entry_num = %d, fermatkind = %d\n", sketch3.get_heavy_bucket_num(), sketch3.get_light_array_num(), sketch3.get_light_entry_num(), sketch3.ifFermatCount);

    // Heavy part
    alignas(64) Bucket* sketch1_buckets = sketch1.heavy_part->buckets;
    alignas(64) Bucket* sketch2_buckets = sketch2.heavy_part->buckets;
    alignas(64) Bucket* sketch3_buckets = sketch3.heavy_part->buckets;

    vector<uint32_t> kickout_keys;
    vector<int32_t> kickout_vals;

    int print_key = 4063245528;
    cout << "Operating heavy part\n";
    for (int i = 0; i < heavy_bucket_num; i++)
    {
        bool full1=1, full2=1;
        bool have_signed_val_1 = 0, have_signed_val_2 = 0;
        bool kicked_flag_2 = 0;
        map<uint32_t, uint32_t> merged_keys_vals;
        map<uint32_t, bool> key_sign_map;
        map<uint32_t, bool> key_sign_map_1;
        map<uint32_t, bool> key_sign_map_2;
        for(int j = 0; j < MAX_VALID_COUNTER; ++j){
            uint32_t key = sketch1_buckets[i].key[j];
            uint32_t originalVal = sketch1_buckets[i].val[j];
            uint32_t val = GetCounterVal(originalVal);
            if(key != 0){
                if(HIGHEST_BIT_IS_1(originalVal)){
                    key_sign_map[key] = 1;
                    key_sign_map_1[key] = 1;
                    have_signed_val_1 = 1;
                }
                else{
                    key_sign_map[key] = -1;
                    key_sign_map_1[key] = -1;
                }
                merged_keys_vals[key] += val;
            }else{
                full1 = 0;
            }
        }
        for(int j = 0; j < MAX_VALID_COUNTER; ++j){
            uint32_t key = sketch2_buckets[i].key[j];
            uint32_t originalVal = sketch2_buckets[i].val[j];
            uint32_t val = GetCounterVal(originalVal);
            if(HIGHEST_BIT_IS_1(originalVal)){
                kicked_flag_2 = 1;              
            }
            if(key != 0){
                if(HIGHEST_BIT_IS_1(originalVal)){
                    kicked_flag_2 = 1;
                    key_sign_map[key] = 1;
                    key_sign_map_2[key] = 1;
                    have_signed_val_2 = 1;
                }
                else if(key_sign_map.count(key) == 0){
                    key_sign_map[key] = -1;
                    key_sign_map_2[key] = -1;
                }
                else{
                    key_sign_map_2[key] = -1;
                }
            }else{
                full2 = 0;
            }

            if(key != 0 && merged_keys_vals.count(key) > 0){
                if(merged_keys_vals[key] > val){
                    merged_keys_vals[key] -= val;
                }
                else if(key_sign_map[key] == -1){
                    merged_keys_vals.erase(key);
                }
                else{
                    //kick out
                    kickout_keys.push_back(key);
                    kickout_vals.push_back(merged_keys_vals[key] - val);
                    merged_keys_vals.erase(key);
                }
            }
            else if(merged_keys_vals.count(key) == 0){
                kickout_keys.push_back(key);
                kickout_vals.push_back(0 - val);
            }
        }

        // Store the remaining keys and values in the new sketch
        int start_index = 0;
        for(auto it = merged_keys_vals.begin(); it != merged_keys_vals.end(); ++it){
            uint32_t key = it->first;
            uint32_t val = it->second;
            sketch3_buckets[i].key[start_index] = key;
            bool sure_1_not_set = (key_sign_map_1.count(key) == 0 && (!full1)) || (key_sign_map_1.count(key) > 0 && key_sign_map_1[key] == -1);
            bool sure_2_not_set = (key_sign_map_2.count(key) == 0 && (!full2)) || (key_sign_map_2.count(key) > 0 && key_sign_map_2[key] == -1);
            if((sure_1_not_set)&&(sure_2_not_set)){
                sketch3_buckets[i].val[start_index] = val;
            }
            else{
                sketch3_buckets[i].val[start_index] = val | 0x80000000;
            }
            start_index++;
        }
    }
    // Light part
    cout << "Lightpart array_num = " << array_num << ", entry_num = " << entry_num << endl;
    for (int i = 0; i < array_num; i++)
    {
        for (int j = 0; j < entry_num; j++)
        {
            // cout << "i: " << i << ", j: " << j << endl;
            uint32_t sketch1_id = sketch1.fermatEle->get_id(i, j);
            uint32_t sketch2_id = sketch2.fermatEle->get_id(i, j);
            int32_t sketch1_counter = sketch1.fermatEle->get_counter(i, j);
            int32_t sketch2_counter = sketch2.fermatEle->get_counter(i, j);
            uint32_t diff_id = ((uint64_t)PRIME_ID_IDP_CNTPM + (uint64_t)sketch1_id - (uint64_t)sketch2_id) % (uint64_t)PRIME_ID_IDP_CNTPM;
            sketch3.fermatEle->set_id(i, j, diff_id);
            sketch3.fermatEle->set_counter(i, j, sketch1_counter - sketch2_counter);
            if(sketch1_id == 3398410894){
                cout << "i: " << i << ", j: " << j << endl;
                cout << "sketch1_id: " << sketch1_id << ", sketch2_id: " << sketch2_id << ", sketch1_counter: " << sketch1_counter << ", sketch2_counter: " << sketch2_counter << endl;
                cout << "sketch1_id - sketch2_id: " << sketch1_id - sketch2_id << ", sketch1_counter - sketch2_counter: " << sketch1_counter - sketch2_counter << endl;
                cout << "diff_id: " << diff_id << " PRIME: " << PRIME_ID_IDP_CNTPM << endl;
                cout << "sketch3_id: " << (uint32_t)(sketch3.fermatEle->get_id(i, j)) << ", sketch3_counter: " << sketch3.fermatEle->get_counter(i, j) << endl;
            }
        }
    }
    //tower
    for(int line_index = 0; line_index < 2; ++line_index){
        int width0 = sketch3.tower->line[line_index].width;
        for(int i = 0; i < width0; ++i){
            uint32_t val1 = sketch1.tower->line[line_index].index(i);
            uint32_t val2 = sketch2.tower->line[line_index].index(i);
            if(val1 == 3 && line_index == 0){
                sketch3.tower->add_val(line_index, i, 3);
            }
            // else if(val2 == 15 && line_index == 1){
            //     sketch3.tower->add_val(line_index, i, 15);
            // }
            else if(val1 < val2){
                sketch3.tower->add_val(line_index, i, 0);
            }
            else{
                sketch3.tower->add_val(line_index, i, val1 - val2);
            }
        }
    }

    // Kick out
    cout << "Operating kick out, kickout_keys.size() = " << kickout_keys.size() << endl;
    for (int i = 0; i < kickout_keys.size(); i++)
    {
        uint32_t key = kickout_keys[i];
        int32_t val = kickout_vals[i];
        // sketch3.fermatEle->Insert(key, val);
        sketch3.insert_after_heavy((char*)&key, val);
    }
    cout << "Size of insertedflows in sketch1 is " << sketch1.fermatEle->insertedflows.size() << endl;
    cout << "Size of insertedflows in sketch2 is " << sketch2.fermatEle->insertedflows.size() << endl;
    cout << "Size of insertedflows in sketch3 is " << sketch3.fermatEle->insertedflows.size() << endl;
    // return sketch3;
}

template<int bucket_num>
long double InnerProduct(DaVinci<bucket_num>& sketch1, DaVinci<bucket_num>& sketch2, bool enable_fast = true)
{
    cout << "Enter InnerProduct" << endl;
    std::ofstream outFile("./outputs/innerP_result_compare.csv");
    outFile << "key, type1, type2, est_val1, est_val2, real_val1, real_val2, est_innerP, real_innerP" << endl;
    long double innerProduct_light = 0;
    long double innerProduct_tower = 0;
    long double innerProduct_heavy = 0;
    long double innerProduct_heavy_tower = 0;
    long double innerProduct_tower_heavy = 0;
    long double innerProduct_light_heavy = 0;
    long double innerProduct_heavy_light = 0;
    long double innerProduct_light_tower = 0;
    long double innerProduct_tower_light = 0;
    long double innerProduct = 0;
    int array_num = sketch1.get_light_array_num();
    int entry_num = sketch1.get_light_entry_num();
    long double res[array_num];

    // lightXlight
    std::cout << "array_num: " << array_num << ", entry_num: " << entry_num << std::endl;
    // if(enable_fast)
    {
        for (int i = 0; i < array_num; i++)
        {
            long double k = 0;
            for (int j = 0; j < entry_num; j++)
                k += 1ll * sketch1.fermatEle->get_counter(i, j) * sketch2.fermatEle->get_counter(i, j);
            res[i] = 1.0 * k;// / entry_num; //TODO: check if this is correct
        }
        long double re = 0;
        for (int i = 0; i < array_num; i++)
            re += res[i];
        innerProduct_light = 1.0 * re / array_num;
    }
    int width = sketch1.tower->line[1].width;
    for(int i = 0; i < width; ++i){
        innerProduct_tower += 1ll * sketch1.tower->line[1].index(i) * sketch2.tower->line[1].index(i);
        // cout << "TowerXtower: " << i << "th index, " << sketch1.tower->line[1].index(i) << " * " << sketch2.tower->line[1].index(i) << " = " << 1ll * sketch1.tower->line[1].index(i) * sketch2.tower->line[1].index(i) << endl;
        innerProduct_tower_light += 1ll * sketch1.tower->line[1].index(i) * sketch2.fermatEle->get_counter(0, i%entry_num);
        // cout << "TowerXlight: " << i << "th index, " << sketch1.tower->line[1].index(i) << " * " << sketch2.fermatEle->get_counter(0, i%entry_num) << " = " << 1ll * sketch1.tower->line[1].index(i) * sketch2.fermatEle->get_counter(0, i%entry_num) << endl;
        innerProduct_light_tower += 1ll * sketch1.fermatEle->get_counter(0, i%entry_num) * sketch2.tower->line[1].index(i);
    }
    std::cout << "Fast inner product with only light part involved is " << innerProduct_light << std::endl;
    if(sketch1.have_decoded == 0){
        sketch1.decode(1);
    }
    if(sketch2.have_decoded == 0){
        sketch2.decode(1);
    }
    if(!enable_fast){
        innerProduct_light = 0;
        cout << "Not using fast mode!" << endl;
        for(auto k:sketch2.Eleresult){
            uint32_t key = k.first;
            uint32_t val = k.second;
            if(sketch1.Eleresult.count(key) == 0){
                continue;
            }
            else{
                innerProduct_light += sketch1.Eleresult[key] * val;
            }
        }
        std::cout << "Slow inner product with only light part involved is " << innerProduct_light << std::endl;
    }

    // towerXtower, towerXlight, lightXtower
    // lightXtower
    // heavyXheavy, heavyXlight, lightXheavy
    for (int i = 0; i < sketch1.heavy_bucket_num; i++)
    {
        map<uint32_t, uint32_t> merged_keys_vals_1;
        map<uint32_t, uint32_t> merged_keys_vals_2;
        for(int j = 0; j < MAX_VALID_COUNTER; ++j){
            uint32_t key = sketch1.heavy_part->buckets[i].key[j];
            if(key == 50331651){
                cout << "We found "<< key << " in bucket1 at" << i << "th bucket " << j << "th key" << endl;
            }
            uint32_t originalVal = sketch1.heavy_part->buckets[i].val[j];
            uint32_t val = GetCounterVal(originalVal);
            if(key != 0){
                if(merged_keys_vals_1.count(key) == 0){
                    if(key == 50331651){
                        cout << "Insert Key into map " << key << endl;
                    }
                    merged_keys_vals_1[key] = val;
                }
                else{
                    if(key == 50331651){
                        cout << "Key " << key << " exists in merged_keys_vals_1" << endl;
                        for(int k = 0; k < MAX_VALID_COUNTER; ++k){
                            cout << sketch1.heavy_part->buckets[i].key[k] << " -> " << sketch1.heavy_part->buckets[i].val[k] << endl;
                        }
                        for(auto it = merged_keys_vals_1.begin(); it != merged_keys_vals_1.end(); ++it){
                            cout << it->first << " ----> " << it->second << endl;
                        }
                    }
                    merged_keys_vals_1[key] += val;
                }
                // 2. Calculate inner product of heavy part and light part
                uint32_t lightValEst = sketch2.fermatEle->undecoded_query((char*)&key);
                if((int)lightValEst < 0)
                    lightValEst = 0;
                int32_t lightValWithDecoding = sketch2.query_only_light_part((char*)&key);
                innerProduct_heavy_light += val * lightValEst;

                //heavyXtower
                innerProduct_heavy_tower += val * sketch2.tower->query((char*)&key);

                outFile << key << ", heavy, light, " << val << ", " << lightValEst << ", " << val << ", " << lightValWithDecoding << ", " << val * lightValEst << ", " << (int)val * lightValWithDecoding << endl;
            }
        }
        for(int j = 0; j < MAX_VALID_COUNTER; ++j){
            uint32_t key = sketch2.heavy_part->buckets[i].key[j];
            uint32_t originalVal = sketch2.heavy_part->buckets[i].val[j];
            uint32_t val = GetCounterVal(originalVal);
            if(key != 0){
                if(merged_keys_vals_1.count(key) > 0){
                    if(key == 50331651){
                        cout << "Key from bucket2 " << key << " exists in merged_keys_vals_1" << endl;
                        for(int k = 0; k < MAX_VALID_COUNTER; ++k){
                            cout << sketch1.heavy_part->buckets[i].key[k] << " -> " << sketch1.heavy_part->buckets[i].val[k] << endl;
                        }
                        for(auto it = merged_keys_vals_1.begin(); it != merged_keys_vals_1.end(); ++it){
                            cout << it->first << " ----> " << it->second << endl;
                        }
                    }
                    innerProduct_heavy += merged_keys_vals_1[key] * val;
                    if(merged_keys_vals_1[key] * val > 10000000)
                    outFile << key << ",heavy, heavy, " << merged_keys_vals_1[key] << ", " << val << ", " << merged_keys_vals_1[key] << ", " << val << ", " << merged_keys_vals_1[key] * val << ", " << (int)merged_keys_vals_1[key] * val << endl;
                    // 2. Calculate inner product of heavy part and light part
                }
                uint32_t lightValEst = sketch1.fermatEle->undecoded_query((char*)&key);
                if((int)lightValEst < 0)
                    lightValEst = 0;
                int32_t lightValWithDecoding = sketch1.query_only_light_part((char*)&key);
                innerProduct_light_heavy += val * lightValEst;

                //towerXheavy
                innerProduct_tower_heavy += sketch1.tower->query((char*)&key) * val;
                outFile << key << ", light, heavy, " << lightValEst << ", " << val << ", " << lightValWithDecoding << "," << val << ", " << val * lightValEst << ", " << (int)val * lightValWithDecoding << endl;
            }
        }

    }

    innerProduct = innerProduct_light + innerProduct_heavy + innerProduct_light_heavy + innerProduct_heavy_light + innerProduct_heavy_tower + innerProduct_tower_heavy + innerProduct_light_tower + innerProduct_tower_light + innerProduct_tower;

    cout << "Inner product with only heavy part involved is " << innerProduct_heavy << endl;
    cout << "Inner product with only tower part involved is " << innerProduct_tower << endl;
    cout << "Inner product with only light part involved is " << innerProduct_light << endl;
    cout << "Inner product with 1 heavy part and 2 light part involved is " << innerProduct_heavy_light << endl;
    cout << "Inner product with 1 light part and 2 heavy part involved is " << innerProduct_light_heavy << endl;
    cout << "Inner product with 1 heavy part and 2 tower part involved is " << innerProduct_heavy_tower << endl;
    cout << "Inner product with 1 tower part and 2 heavy part involved is " << innerProduct_tower_heavy << endl;
    cout << "Inner product with 1 light part and 2 tower part involved is " << innerProduct_light_tower << endl;
    cout << "Inner product with 1 tower part and 2 light part involved is " << innerProduct_tower_light << endl;
    cout << "Total inner product is " << innerProduct << endl;
    outFile.close();
    return innerProduct;
}
