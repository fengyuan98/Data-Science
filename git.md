

## Git

**分布式版本控制系统**：

- 基于C语言，由Linux创建人Linus创建
- 自动记录每次文件的改动，还可以让同伴协作编辑
- 由集中式到分布式，不再依赖中央服务器传输文件；每个人电脑里都有完整的版本库

- 安装：https://www.liaoxuefeng.com/wiki/896043488029600/896067074338496
- 创建版本库 / 仓库（Repository）

```c
cd **                     #打开某个文件夹
pwd                       #显示当前目录
git init                  #把当前目录变成Git可管理的仓库
git add **                #将当前目录下的某一个文件添加到仓库
git commit -m "**"        #将文件提交到仓库，并输入本次提交的说明
ls / dir                  #查看当前目录中的文件
```

 ```c
 git status                #查看仓库当前状态
 git diff                  #查看变动的部分
 ```

- 版本回退

```c
git log                   #查看历史版本记录，从最近到最远的提交日志
git log --pretty=oneline
HEAD                      #当前版本
HEAD^                     #上一个版本
HEAD^^                    #上上个版本
HEAD~100                  #往上100个版本
git reset --hard HEAD^.   #将当前版本退回到上一个版本
                          #只要找到某个版本的commit_id，就能到达那个版本
git reflog                #记录每一次命令，用于查询各个版本的ID
```

- 工作区和暂存区

  git add 命令实际上就是把要提交的所有修改放到暂存区（stage），然后 git commit 就可以一次性把暂存区的所有修改提交到分支。

```c
git diff HEAD -- **       #查看**文件在工作区和版本库里面最新版本的区别
```

- 撤销修改

```c
git checkout -- **       #丢弃工作区中对**文件的修改
git reset HEAD **        #把暂存区的修改撤销掉（unstage）重新放回工作区
```

- 删除文件

```c
rm **                     #删除**文件
git rm ** 之后 git commit  #将文件从版本库中删除
```

## GitHub

**远程仓库**

1. 创建SSH key：如果用户主目录下没有“id_rsa”（私钥）和"id_rsa.pub"（公钥）这两个文件，打开Shell（windows下打开Git Bash）创建SSH key

   ```c
   ssh-keygen -t rsa -C "youremail@example.com"
   ```

2. 登陆Github，打开“Account setting”，“SSH Keys”页面：

   点“Add SSH Key”，填上Title，在Key文本框里粘贴“id_rsa_pub”文件的内容

3. 点“Add Key”，查看已经添加的Key

- 把已有的本地仓库与Github上创建的仓库关联，将本地的内容推到Github上

  ```c
  git remote add origin 
  git@github.com:fengyuan98/origin.git
  # 远程库的名字是 origin
  ```

  ```c
  git push -u origin master     #将本地库内容推送到远程
  git push origin master        #把本地master分支的最新修改推送至GitHub
  ```

**⚠️SSH警告**：

当你第一次使用Git的`clone`或者`push`命令连接GitHub时，会得到一个警告：

```c
The authenticity of host 'github.com (xx.xx.xx.xx)' can't be established.
RSA key fingerprint is xx.xx.xx.xx.xx.
Are you sure you want to continue connecting (yes/no)?
```

这是因为Git使用SSH连接，而SSH连接在第一次验证GitHub服务器的Key时，需要你确认GitHub的Key的指纹信息是否真的来自GitHub的服务器，输入`yes`回车即可。

- 删除远程库

  ```c
  git remote -v             #查看远程库信息
  git remote rm origin      #根据名字删除，如删除origin
  ```

  **这里的删除只是解除了本地和远程的绑定关系，并不是物理删除**

- 从远程库克隆

  ```c
  # 将Github上的gitskills仓库克隆到本地，默认的“git://”使用SSH协议,速度快
  git clone git@github.com:fengyuan98/gitskills.git
  ```

- 分支管理

  ```c
  git checkout -b dev             #创建并切换到dev分支
  git branch dev                  #创建dev分支
  git checkout dev                #切换到dev分支
  git branch                      #查看当前分支，前面有*号那个
  git merge dev                   #合并指定分支到当前分支
  git branch -d dev               #删除dev分支
  
  git switch -c dev               #创建并切换到dev分支
  git switch master               #切换到已有的master分支
  ```

  



































