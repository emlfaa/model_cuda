

using namespace std;

class file_process{
public:

    // 获得目录下所有文件的名字
    void GetFileNames(string path, vector<string>& filenames);

    // 判断目录是否存在 不存在创建
    int str2char(string s, char c[]);
    bool dir_file_exists(string dir, bool mkdir_flag);
    void create_file(string save_path, vector<string> fun);

};
