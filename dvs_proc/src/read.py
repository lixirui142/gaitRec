import os

out_file = []
file = []
def list_all_files(rootdir,outdir):
    list = os.listdir(rootdir)
    for i in range(0,len(list)):
        path = os.path.join(rootdir,list[i])
        if os.path.isdir(path):
            temp = path.split('\\')
            outpath = os.path.join(outdir,temp[-1])
            os.mkdir(outpath)
            list_all_files(path,outpath)
        if os.path.isfile(path):
            file.append(path)
            temp = path.split('\\')
            outpath = os.path.join(outdir,temp[-1][:-7])
            if not os.path.exists(outpath):
                os.mkdir(outpath)
            out_file.append(outpath)
    return file,out_file