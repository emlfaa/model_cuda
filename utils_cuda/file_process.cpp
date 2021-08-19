#include <string>
#include <fstream>
#include <sstream>
#include <dirent.h>
#include <unistd.h>
#include <iostream>
#include <vector>
#include <cstring>


#include "file_process.h"

using namespace std;

// 获得目录下所有文件的名字
void file_process::GetFileNames(string path, vector<string>& filenames)
{
    DIR *pDir;
    struct dirent* ptr;
    if(!(pDir = opendir(path.c_str()))){
        cout<<"Folder doesn't Exist!"<<endl;
        return;
    }
    while((ptr = readdir(pDir)) != 0){
        if(strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0){
            filenames.push_back(path + "/" + ptr->d_name);
        }
    }
    closedir(pDir);
}

// https://blog.csdn.net/cy_tec/article/details/51249231
int file_process::str2char(string s, char c[])
{
    size_t l = s.length();
    int i;
    for(i = 0; i < l; i++)
        c[i] =s [i];
    c[i] = '\0';
    return i;
}

// https://blog.csdn.net/cy_tec/article/details/51249231
// 判断目录是否存在 不存在创建
bool file_process::dir_file_exists(string dir, bool mkdir_flag)
{
    char des_dir[255];
    str2char(dir, des_dir);  // 将string 写入到字符数组中
    int state = access(des_dir, R_OK|W_OK);  // 头文件 #include <unistd.h>
    if(state == 0){
        cout<<"file exist"<<endl;
        return true;
    }
    else if(mkdir_flag)
    {
        dir = "mkdir " + dir;   // 调用linux系统命令创建文件
        str2char(dir, des_dir);
        cout<<des_dir<<endl;
        system(des_dir);
        return true;
    }
}

void file_process::create_file(string save_path, vector<string> fun){

    // 判断功能类别文件夹是否创建
    for(int i = 0; i < fun.size(); i++)
    {
        dir_file_exists(save_path + "/" + fun[i], true);
    }

//    // 创建对应图片所在的文件夹
//    for(int i = 0; i < fun.size(); i++)
//    {
//        dir_file_exists(save_path + "/" + fun[i] + "/" + img_file, true);  // 判断在下采样保存的路径下是否有对应的文件夹，没有进行创建
//    }

}